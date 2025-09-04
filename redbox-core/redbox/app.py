import os
from asyncio import CancelledError
from logging import getLogger
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from redbox.chains.components import (
    get_all_chunks_retriever,
    get_embeddings,
    get_metadata_retriever,
    get_parameterised_retriever,
    get_tabular_chunks_retriever,
)
from redbox.graph.nodes.tools import (
    build_govuk_search_tool,
    build_legislation_search_tool,
    build_search_documents_tool,
    build_search_wikipedia_tool,
)
from redbox.graph.root import build_root_graph, get_agentic_search_graph, get_summarise_graph
from redbox.models.chain import RedboxState
from redbox.models.chat import ChatRoute
from redbox.models.file import ChunkResolution
from redbox.models.graph import (
    FINAL_RESPONSE_TAG,
    ROUTABLE_KEYWORDS,
    ROUTE_NAME_TAG,
    SUMMARY_MULTIAGENT_TAG,
    RedboxEventType,
)
from redbox.models.settings import Settings, get_settings


async def _default_callback(*args, **kwargs):
    return None


logger = getLogger(__name__)


class Redbox:
    def __init__(
        self,
        all_chunks_retriever: VectorStoreRetriever | None = None,
        parameterised_retriever: VectorStoreRetriever | None = None,
        tabular_retriever: VectorStoreRetriever | None = None,
        metadata_retriever: VectorStoreRetriever | None = None,
        embedding_model: Embeddings | None = None,
        env: Settings | None = None,
        debug: bool = False,
    ):
        _env = env or get_settings()

        # DB Loction and s3_keys from the previous request
        self.previous_db_location: str | None = None
        self.previous_s3_keys: list[str] | None = None

        # Retrievers

        self.all_chunks_retriever = all_chunks_retriever or get_all_chunks_retriever(_env)
        self.parameterised_retriever = parameterised_retriever or get_parameterised_retriever(_env)
        self.tabular_retriever = tabular_retriever or get_tabular_chunks_retriever(_env)
        self.metadata_retriever = metadata_retriever or get_metadata_retriever(_env)
        self.embedding_model = embedding_model or get_embeddings(_env)

        # Tools

        search_documents = build_search_documents_tool(
            es_client=_env.elasticsearch_client(),
            index_name=_env.elastic_chunk_alias,
            embedding_model=self.embedding_model,
            embedding_field_name=_env.embedding_document_field_name,
            chunk_resolution=ChunkResolution.normal,
        )
        search_wikipedia = build_search_wikipedia_tool()
        search_govuk = build_govuk_search_tool()
        search_legislation = build_legislation_search_tool()

        self.tools = [search_documents, search_wikipedia, search_govuk]

        self.legislation_tools = {
            "Internal_Retrieval_Agent": [search_documents],
            "External_Retrieval_Agent": [search_legislation],
        }

        self.multi_agent_tools = {
            "Internal_Retrieval_Agent": [search_documents],
            "External_Retrieval_Agent": [search_wikipedia, search_govuk],
        }

        self.graph = build_root_graph(
            all_chunks_retriever=self.all_chunks_retriever,
            parameterised_retriever=self.parameterised_retriever,
            tabular_retriever=self.tabular_retriever,
            metadata_retriever=self.metadata_retriever,
            tools=self.tools,
            legislation_tools=self.legislation_tools,
            multi_agent_tools=self.multi_agent_tools,
            debug=debug,
        )

    def run_sync(self, input: RedboxState):
        """
        Run Redbox without streaming events. This simpler, synchronous execution enables use of the graph debug logging
        """
        return self.graph.invoke(input=input)

    async def run(
        self,
        input: RedboxState,
        response_tokens_callback=_default_callback,
        route_name_callback=_default_callback,
        documents_callback=_default_callback,
        citations_callback=_default_callback,
        metadata_tokens_callback=_default_callback,
        activity_event_callback=_default_callback,
    ) -> RedboxState:
        final_state = None
        request_dict = input.request.model_dump()
        logger.info("Request: %s", {k: request_dict[k] for k in request_dict.keys() - {"ai_settings"}})
        is_summary_multiagent_streamed = False
        is_evaluator_output_streamed = False

        input = self.add_docs_and_db_to_input_state(input)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(CancelledError),
            before_sleep=lambda retry_state: logger.warning(
                f"CancelledError in astream_events, attempt {retry_state.attempt_number}/3, retrying in {retry_state.next_action.sleep}s"
            ),
        )
        async def stream_events_with_retry(
            is_summary_multiagent_streamed=False,
            is_evaluator_output_streamed=False,
        ):
            nonlocal final_state
            local_is_summary_multiagent_streamed = is_summary_multiagent_streamed
            local_is_evaluator_output_streamed = is_evaluator_output_streamed
            async for event in self.graph.astream_events(
                input=input,
                version="v2",
                config={"recursion_limit": input.request.ai_settings.recursion_limit},
            ):
                kind = event["event"]
                tags = event.get("tags", [])
                try:
                    if kind == "on_chat_model_stream" and FINAL_RESPONSE_TAG in tags:
                        content = event["data"]["chunk"].content
                        if isinstance(content, str):
                            await response_tokens_callback(content)
                    elif kind == "on_chat_model_stream" and SUMMARY_MULTIAGENT_TAG in tags:
                        if local_is_evaluator_output_streamed:
                            await response_tokens_callback("\n\n")
                            local_is_evaluator_output_streamed = False

                        content = event["data"]["chunk"].content
                        if isinstance(content, str):
                            await response_tokens_callback(content)
                        local_is_summary_multiagent_streamed = True

                    elif kind == "on_chain_end" and FINAL_RESPONSE_TAG in tags:
                        content = event["data"]["output"]
                        if isinstance(content, str):
                            await response_tokens_callback(content)
                    elif kind == "on_custom_event" and event["name"] == RedboxEventType.response_tokens.value:
                        if local_is_summary_multiagent_streamed:
                            await response_tokens_callback("\n\n")
                            local_is_summary_multiagent_streamed = False

                        await response_tokens_callback(event["data"])
                        local_is_evaluator_output_streamed = True

                    elif kind == "on_chain_end" and ROUTE_NAME_TAG in tags:
                        await route_name_callback(event["data"]["output"]["route_name"])
                    elif kind == "on_custom_event" and event["name"] == RedboxEventType.on_source_report.value:
                        await documents_callback(event["data"])
                    elif kind == "on_custom_event" and event["name"] == RedboxEventType.on_citations_report.value:
                        await citations_callback(event["data"])
                    elif kind == "on_custom_event" and event["name"] == RedboxEventType.on_metadata_generation.value:
                        await metadata_tokens_callback(event["data"])
                    elif kind == "on_custom_event" and event["name"] == RedboxEventType.activity.value:
                        await activity_event_callback(event["data"])
                    elif kind == "on_chain_end" and event["name"] == "LangGraph":
                        final_state = RedboxState(**event["data"]["output"])
                except Exception as e:
                    logger.error(f"Error processing {kind} - {str(e)}")
                    raise

        try:
            await stream_events_with_retry(
                is_summary_multiagent_streamed=is_summary_multiagent_streamed,
                is_evaluator_output_streamed=is_evaluator_output_streamed,
            )
            try:
                _ = final_state.messages[-1].content
            except Exception as _:
                logger.exception("LLM Error - Blank Response")
        except CancelledError:
            logger.error("All retries exhausted for CancelledError in the astream_events function")
            raise

        except Exception:
            logger.error("Generic error in run - {str(e)}")
            raise

        self.handle_db_file(final_state)
        return final_state

    def handle_db_file(self, final_state: RedboxState):
        """Manages database file lifecycle based on state changes.

        Note:
            - If final_state is None, all database references and S3 keys are cleared
            - When database location changes, the previous database file is removed if it exists
            - The method maintains previous_db_location and previous_s3_keys as instance variables"""
        if final_state is None:
            logger.warning("No final state")
            self.remove_db_file_if_exists(self.previous_db_location)
            self.previous_db_location = None
            self.previous_s3_keys = None
        else:
            # Delete Previous DB File Location if it has changed and it exists in memory, update the new db location and previous documentation
            new_db_loc, old_db_loc = final_state.request.db_location, self.previous_db_location
            if (new_db_loc != old_db_loc) or not (new_db_loc):
                self.remove_db_file_if_exists(old_db_loc)
                self.previous_db_location = new_db_loc
            self.previous_s3_keys = final_state.request.s3_keys

    def add_docs_and_db_to_input_state(self, input: RedboxState):
        """
        Updates the input state with previous document references and database location for tabular questions.

        Note:
            - The method specifically targets questions that start with "@tabular"
            - Previous S3 keys are only injected if they exist in the current instance
            - Any exceptions during this process are logged but not propagated
        """
        old_input = input.model_copy(deep=True)  # Copy the model as is incase an error occurs while modifying it.
        try:
            # If the question is tabular, inject the previous docs and db_location into the request
            # This will need to be updated if tabular goes into newroute
            if input.request.question.startswith("@tabular"):
                # Inject Previous s3 keys if they exist
                if self.previous_s3_keys:
                    input.request.previous_s3_keys = self.previous_s3_keys

                # Inject previous db location
                input.request.db_location = self.previous_db_location
        except Exception as e:
            logger.info(f"Exception logged while trying to access input state: \n\n{e}")
            return old_input
        return input

    def get_available_keywords(self) -> dict[ChatRoute, str]:
        return ROUTABLE_KEYWORDS

    def draw(self, output_path=None, graph_to_draw: Literal["root", "agent", "chat_with_documents"] = "root"):
        from langchain_core.runnables.graph import MermaidDrawMethod

        if graph_to_draw == "root":
            graph = self.graph.get_graph()
        elif graph_to_draw == "agent":
            graph = get_agentic_search_graph(self.tools).get_graph()
        elif graph_to_draw == "summarise":
            graph = get_summarise_graph(self.all_chunks_retriever, self.parameterised_retriever).get_graph()
        else:
            raise Exception("Invalid graph_to_draw")

        return graph.draw_mermaid_png(draw_method=MermaidDrawMethod.API, output_file_path=output_path)

    def remove_db_file_if_exists(self, db_path: str):
        try:
            if db_path and os.path.exists(db_path):
                os.remove(db_path)
        except Exception as e:
            logger.error(f"Error encountered when deleting the db file at {db_path}: {e}")
