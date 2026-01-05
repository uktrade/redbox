import pytest

from redbox.chains.components import (
    AISettings,
    get_all_chunks_retriever,
    get_embeddings,
    get_metadata_retriever,
    get_parameterised_retriever,
    get_tabular_chunks_retriever,
)
from redbox.graph.root import build_new_route_graph
from redbox.models.chain import Agent
from redbox.models.settings import Settings


class TestNewRouteGraphs:
    _env = Settings()
    agents = AISettings().worker_agents
    all_chunks_retriever = get_all_chunks_retriever(_env)
    parameterised_retriever = get_parameterised_retriever(_env)
    tabular_retriever = get_tabular_chunks_retriever(_env)
    metadata_retriever = get_metadata_retriever(_env)
    embedding_model = get_embeddings(_env)

    multi_agent_tools = {agent.name: [] for agent in agents}

    @pytest.mark.parametrize(
        "agent_name, edges",
        [
            ("Internal_Retrieval_Agent", ["combine_question_evaluator"]),
            ("External_Retrieval_Agent", ["combine_question_evaluator"]),
            ("Legislation_Search_Agent", ["combine_question_evaluator"]),
            ("Web_Search_Agent", ["combine_question_evaluator"]),
            ("Tabular_Agent", ["retrieve_tabular_documents"]),
            ("Summarisation_Agent", None),
            ("Submission_Checker_Agent", ["update_submission_eval", "combine_question_evaluator"]),
            ("Submission_Question_Answer_Agent", ["update_submission_qa", "combine_question_evaluator"]),
        ],
    )
    def test_new_route_graph(self, agent_name, edges):
        graph = build_new_route_graph(
            all_chunks_retriever=self.all_chunks_retriever,
            tabular_retriever=self.tabular_retriever,
            multi_agent_tools=self.multi_agent_tools,
            agents=self.agents,
        ).get_graph()
        # check if we have this agent node in the graph
        assert agent_name in graph.nodes

        # check if the edge is correct for the agent nodes
        if edges is None:
            assert len([edge.target for edge in graph.edges if edge.source == agent_name]) == 0
        else:
            edge_list = [agent_name] + edges
            for i in range(len(edge_list) - 1):
                assert edge_list[i + 1] in [edge.target for edge in graph.edges if edge.source == edge_list[i]]

    def test_non_existent_node(self):
        with pytest.raises(ValueError):
            graph = build_new_route_graph(
                all_chunks_retriever=self.all_chunks_retriever,
                tabular_retriever=self.tabular_retriever,
                multi_agent_tools=self.multi_agent_tools,
                agents=[Agent(name="Fake_Agent")],
            ).get_graph()

            assert "Fake_Agent" not in graph.nodes
