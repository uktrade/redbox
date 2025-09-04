import logging
from functools import cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool

from redbox.chains.parser import StreamingJsonOutputParser, StreamingPlanner
from redbox.models.chain import (
    AISettings,
    MultiAgentPlan,
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
from redbox.transform import bedrock_tokeniser

logger = logging.getLogger(__name__)
load_dotenv()


def get_chat_llm(
    model: ChatLLMBackend, ai_settings: AISettings = AISettings(), tools: list[StructuredTool] | None = None
):
    logger.debug(
        "initialising model=%s model_provider=%s tools=%s max_tokens=%s",
        model.name,
        model.provider,
        tools,
        ai_settings.llm_max_tokens,
    )
    chat_model = init_chat_model(
        model=model.name,
        model_provider=model.provider,
        max_tokens=ai_settings.llm_max_tokens,
        configurable_fields=["base_url"],
    )
    if tools:
        chat_model = chat_model.bind_tools(tools)
    return chat_model


def get_base_chat_llm(model: ChatLLMBackend, ai_settings: AISettings = AISettings()):
    """Initialises a chat llm that returns a BaseChatModel object i.e. a model without tools that is not configurable."""
    logger.debug(
        "initialising model=%s model_provider=%s max_tokens=%s",
        model.name,
        model.provider,
        ai_settings.llm_max_tokens,
    )
    return init_chat_model(
        model=model.name,
        model_provider=model.provider,
        max_tokens=ai_settings.llm_max_tokens,
    )


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
        pydantic_schema_object=MultiAgentPlan,
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
