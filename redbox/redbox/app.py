from asyncio import CancelledError
from logging import getLogger
from typing import Dict, Literal

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
from redbox.graph.agents.configs import AgentConfig, agent_configs
from redbox.graph.nodes.tools import (
    build_document_from_prompt_tool,
    build_govuk_search_tool,
    build_legislation_search_tool,
    build_retrieve_document_full_text,
    build_retrieve_knowledge_base,
    build_search_documents_tool,
    build_search_wikipedia_tool,
    build_web_search_tool,
    execute_sql_query,
)
from redbox.graph.root import build_new_route_graph, build_root_graph, get_summarise_graph
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
        agents: Dict[str, AgentConfig] | None = None,
        all_chunks_retriever: VectorStoreRetriever | None = None,
        parameterised_retriever: VectorStoreRetriever | None = None,
        tabular_retriever: VectorStoreRetriever | None = None,
        metadata_retriever: VectorStoreRetriever | None = None,
        embedding_model: Embeddings | None = None,
        env: Settings | None = None,
        debug: bool = False,
    ):
        _env = env or get_settings()

        # agents configs
        self.agent_configs = agents or agent_configs

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
        retrieve_full_text = build_retrieve_document_full_text(
            es_client=_env.elasticsearch_client(), index_name=_env.elastic_chunk_alias, loop=True
        )
        retrieve_knowledge_base = build_retrieve_knowledge_base(
            es_client=_env.elasticsearch_client(), index_name=_env.elastic_chunk_alias, loop=True
        )

        search_wikipedia = build_search_wikipedia_tool()
        search_govuk = build_govuk_search_tool()
        execute_sql = execute_sql_query()
        web_search = build_web_search_tool()
        legislation_search = build_legislation_search_tool()
        doc_from_prompt = build_document_from_prompt_tool(loop=True)

        self.agent_configs["Internal_Retrieval_Agent"].tools = [search_documents]
        self.agent_configs["External_Retrieval_Agent"].tools = [search_wikipedia, search_govuk]
        self.agent_configs["Tabular_Agent"].tools = [execute_sql]
        self.agent_configs["Web_Search_Agent"].tools = [web_search]
        self.agent_configs["Legislation_Search_Agent"].tools = [legislation_search]
        self.agent_configs["Submission_Question_Answer_Agent"].tools = [
            retrieve_full_text,
            retrieve_knowledge_base,
            doc_from_prompt,
        ]
        self.agent_configs["Submission_Checker_Agent"].tools = [
            retrieve_full_text,
            retrieve_knowledge_base,
            doc_from_prompt,
        ]

        self.graph = build_root_graph(
            all_chunks_retriever=self.all_chunks_retriever,
            parameterised_retriever=self.parameterised_retriever,
            tabular_retriever=self.tabular_retriever,
            metadata_retriever=self.metadata_retriever,
            agent_configs=self.agent_configs,
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

        return final_state

    def get_available_keywords(self) -> dict[ChatRoute, str]:
        return ROUTABLE_KEYWORDS

    def draw(
        self, output_path=None, graph_to_draw: Literal["root", "agent", "chat_with_documents", "new_route"] = "root"
    ):
        from langchain_core.runnables.graph import MermaidDrawMethod

        if graph_to_draw == "root":
            graph = self.graph.get_graph()
        elif graph_to_draw == "summarise":
            graph = get_summarise_graph(self.all_chunks_retriever, self.parameterised_retriever).get_graph()
        elif graph_to_draw == "new_route":
            graph = build_new_route_graph(
                all_chunks_retriever=self.all_chunks_retriever,
                tabular_retriever=self.tabular_retriever,
                agents=self.agents,
            ).get_graph()
        else:
            raise Exception("Invalid graph_to_draw")

        return graph.draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER, output_file_path=output_path)
