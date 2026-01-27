import logging
import time
from functools import cache

from botocore.exceptions import ClientError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool

from redbox.chains.parser import StreamingJsonOutputParser, StreamingPlanner
from redbox.models.chain import (
    AISettings,
    MultiAgentPlanBase,
    StructuredResponseWithCitations,
    StructuredResponseWithStepsTaken,
)
from redbox.models.settings import ChatLLMBackend, Settings
from redbox.retriever import (
    AllElasticsearchRetriever,
    BasicMetadataRetriever,
    MetadataRetriever,
    OpenSearchRetriever,
    ParameterisedElasticsearchRetriever,
    TabularElasticsearchRetriever,
)
from redbox.retriever.retrievers import KnowledgeBaseMetadataRetriever, KnowledgeBaseTabularMetadataRetriever
from redbox.transform import bedrock_tokeniser

logger = logging.getLogger(__name__)
load_dotenv()


_FALLBACK_CACHE = {}

FALLBACK_COOLDOWN_SECS = 420  # 7 mins feels ok


def get_chat_llm(
    model: ChatLLMBackend,
    ai_settings: AISettings = AISettings(),
    tools: list[StructuredTool] | None = None,
):
    fallback_backend = ChatLLMBackend(
        name="anthropic.claude-3-7-sonnet-20250219-v1:0",
        provider="bedrock",
    )

    def _init_model(backend: ChatLLMBackend):
        kwargs = {}
        if backend.name.startswith("arn"):
            if not backend.provider:
                raise ValueError(
                    "When using a model ARN you must set model.provider "
                    "to the foundation-model provider (e.g., 'anthropic')."
                )
            kwargs["provider"] = "anthropic"

        chat_model = init_chat_model(
            model=backend.name,
            model_provider=backend.provider,
            max_tokens=ai_settings.llm_max_tokens,
            configurable_fields=["base_url"],
            **kwargs,
        )
        if tools:
            chat_model = chat_model.bind_tools(tools)
        return chat_model

    cache_entry = _FALLBACK_CACHE.get(model.name)
    if cache_entry and cache_entry["until"] > time.time():
        logger.debug(
            "Using cached fallback for %s until %s",
            model.name,
            time.strftime("%H:%M:%S", time.localtime(cache_entry["until"])),
        )
        return _init_model(cache_entry["backend"])

    try:
        return _init_model(model)

    except ClientError as e:
        error_code = e.response["Error"].get("Code", "")
        if error_code in (
            "ServiceUnavailableException",
            "ThrottlingException",
            "RateLimitExceeded",
            "TooManyRequestsException",
        ):
            logger.warning(
                "Rate/service limit (%s) encountered with %s. Falling back to %s",
                error_code,
                model.name,
                fallback_backend.name,
            )
            _FALLBACK_CACHE[model.name] = {
                "until": time.time() + FALLBACK_COOLDOWN_SECS,
                "backend": fallback_backend,
            }
            return _init_model(fallback_backend)
        else:
            raise e

    except (TimeoutError, ConnectionError, EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError) as e:
        logger.warning(
            "Connection issue (%s) with %s. Falling back to %s",
            str(e),
            model.name,
            fallback_backend.name,
        )
        _FALLBACK_CACHE[model.name] = {
            "until": time.time() + FALLBACK_COOLDOWN_SECS,
            "backend": fallback_backend,
        }
        return _init_model(fallback_backend)


@cache
def get_tokeniser() -> callable:
    return bedrock_tokeniser


def get_aws_embeddings(env: Settings):
    return BedrockEmbeddings(region_name=env.aws_region, model_id=env.embedding_backend)


def get_embeddings(env: Settings) -> Embeddings:
    if env.embedding_backend == "amazon.titan-embed-text-v2:0":
        return get_aws_embeddings(env)
    if env.embedding_backend == "fake":
        return FakeEmbeddings(
            size=1024
        )  # set embedding size to 1024 to match bedrock model amazon.titan-embed-text-v2:0 default embedding size
    raise Exception("No configured embedding model")


def get_all_chunks_retriever(env: Settings) -> OpenSearchRetriever:
    return AllElasticsearchRetriever(
        es_client=env.elasticsearch_client(),
        index_name=env.elastic_chunk_alias,
    )


def get_tabular_chunks_retriever(env: Settings) -> OpenSearchRetriever:
    return TabularElasticsearchRetriever(
        es_client=env.elasticsearch_client(),
        index_name=env.elastic_chunk_alias,
    )


def get_parameterised_retriever(env: Settings, embeddings: Embeddings | None = None):
    """Creates an Elasticsearch retriever runnable.

    Runnable takes input of a dict keyed to question, file_uuids and user_uuid.

    Runnable returns a list of Chunks.
    """
    return ParameterisedElasticsearchRetriever(
        es_client=env.elasticsearch_client(),
        index_name=env.elastic_chunk_alias,
        embedding_model=embeddings or get_embeddings(env),
        embedding_field_name=env.embedding_document_field_name,
    )


def get_metadata_retriever(env: Settings):
    return MetadataRetriever(
        es_client=env.elasticsearch_client(),
        index_name=env.elastic_chunk_alias,
    )


def get_basic_metadata_retriever(env: Settings):
    return BasicMetadataRetriever(
        es_client=env.elasticsearch_client(),
        index_name=env.elastic_chunk_alias,
    )


def get_knowledge_base_metadata_retriever(env: Settings):
    return KnowledgeBaseMetadataRetriever(
        es_client=env.elasticsearch_client(),
        index_name=env.elastic_schematised_chunk_index,
    )


def get_knowledge_base_tabular_metadata_retriever(env: Settings):
    return KnowledgeBaseTabularMetadataRetriever(
        es_client=env.elasticsearch_client(),
        index_name=env.elastic_schematised_chunk_index,
    )


def get_structured_response_with_citations_parser() -> tuple[Runnable, str]:
    """
    Returns the output parser (as a runnable) for creating the StructuredResponseWithCitations object
    while streaming the answer tokens
    Also returns the format instructions for this structure for use in the prompt
    """
    # pydantic_parser = PydanticOutputParser(pydantic_object=StructuredResponseWithCitations)
    parser = StreamingJsonOutputParser(
        name_of_streamed_field="answer", pydantic_schema_object=StructuredResponseWithCitations
    )
    return (parser, parser.get_format_instructions())


def get_structured_response_with_planner_parser() -> tuple[Runnable, str]:
    parser = StreamingPlanner(
        name_of_streamed_field="tasks",
        pydantic_schema_object=MultiAgentPlanBase,
        sub_streamed_field="task",
        suffix_texts=[
            "\n\n" + item
            for item in [
                "Please let me know if you want me to go ahead with the plan, or make any changes.",
                "Let me know if you would like to proceed, or you can also ask me to make changes.",
                "If you're happy with this approach let me know, or you can change the approach also.",
                "Let me know if you'd like me to proceed, or if you want to amend or change the plan.",
            ]
        ],
        prefix_texts=[
            "Here is the plan I will execute:",
            "Here is my proposed plan:",
            "I can look into this for you, here's my current plan:",
            "Sure, here's my current plan:",
        ],
    )
    return (parser, parser.get_format_instructions())


def get_structured_response_with_steps_taken_parser() -> tuple[Runnable, str]:
    parser = StreamingJsonOutputParser(
        name_of_streamed_field="output", pydantic_schema_object=StructuredResponseWithStepsTaken
    )
    return parser, parser.get_format_instructions()
