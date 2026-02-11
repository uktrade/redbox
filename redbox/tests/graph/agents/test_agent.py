import pytest
from langchain_core.messages import AIMessage
from pytest_mock import MockerFixture

from redbox.graph.agents.configs import AgentConfig, PromptConfig, PromptVariable, agent_configs
from redbox.graph.agents.formats import ArtifactAgent
from redbox.graph.agents.workers import WorkerAgent
from redbox.graph.nodes.tools import build_search_wikipedia_tool
from redbox.models.chain import AgentTaskBase, RedboxState
from redbox.test.data import GenericFakeChatModelWithTools

WORKER_RESPONSE_WITH_TOOL = AIMessage(
    content="I will be calling a tool",
    additional_kwargs={
        "tool_calls": [
            {
                "id": "call_e4003b",
                "function": {"arguments": '{\n  "query": "ai"\n}', "name": "_some_tool"},
                "type": "function",
            }
        ]
    },
)


class TestWorkerAgent:
    propmt_config = PromptConfig(
        system="Fake system prompt",
        question="Fake question prompt",
        prompt_vars=PromptVariable(question=True, metadata=True),
    )
    tools = [build_search_wikipedia_tool()]
    config = AgentConfig(
        name="Internal_Retrieval_Agent", description="Fake description", prompt=propmt_config, tools=tools
    )
    worker = WorkerAgent(config=config)
    task = AgentTaskBase(
        task="Fake task",
        agent="Internal_Retrieval_Agent",
        expected_output="A comprehensive list of fake results",
    )

    @pytest.mark.parametrize("success, task", [("success", task.model_dump_json()), ("fail", "")])
    def test_reading_task_info(self, success, task, fake_state):
        fake_state.messages = [AIMessage(content=task)]
        _, task = self.worker.reading_task_info().invoke(fake_state)
        if success == "success":
            assert task is not None
        else:
            assert task is None

    @pytest.mark.parametrize(
        "result", [("A result"), ([AIMessage("A"), AIMessage("result")]), ([{"text": "A result"}])]
    )
    def test_post_processing(self, result, fake_state):
        task = self.task
        response = self.worker.post_processing().invoke((fake_state, result, task))
        assert (
            response["agents_results"]
            == f"<{self.worker.config.name}_Result>A result</{self.worker.config.name}_Result>"
        )
        assert response["tasks_evaluator"] == task.task + "\n" + task.expected_output

    @pytest.mark.parametrize(
        "AI_response",
        [
            ("Here is your fake response"),
            (AIMessage(content="Here is your fake response")),
            (WORKER_RESPONSE_WITH_TOOL),
        ],
    )
    def test_core_task(self, AI_response, fake_state, mocker: MockerFixture):
        # mock LLM call
        llm_mock = mocker.patch("redbox.chains.runnables.get_chat_llm")
        llm_mock.return_value = GenericFakeChatModelWithTools(messages=iter([AI_response]))
        # mock tool response
        mocker.patch(
            "redbox.graph.agents.workers.run_tools_parallel",
            return_value=[AIMessage(content="Here is your fake response")],
        )
        task = self.task
        response = self.worker.core_task().invoke((fake_state, task))
        if isinstance(response, str):
            response == "Here is your fake response"
        elif isinstance(response, list):
            assert response[0].content == "Here is your fake response"

    def test_execute(self, fake_state: RedboxState, mocker: MockerFixture):
        # mock LLM call
        llm_mock = mocker.patch("redbox.chains.runnables.get_chat_llm")
        llm_mock.return_value = GenericFakeChatModelWithTools(messages=iter([WORKER_RESPONSE_WITH_TOOL]))
        # mock tool response
        mocker.patch(
            "redbox.graph.agents.workers.run_tools_parallel",
            return_value=[AIMessage(content="Here is your fake response")],
        )
        task = self.task
        fake_state.messages = [AIMessage(content=task.model_dump_json())]
        response = self.worker.execute().invoke(fake_state)
        type(response)
        assert (
            response["agents_results"]
            == f"<{self.worker.config.name}_Result>Here is your fake response</{self.worker.config.name}_Result>"
        )
        assert response["tasks_evaluator"] == task.task + "\n" + task.expected_output


class TestArtifactAgent:
    task = AgentTaskBase(
        task="Fake task",
        agent="Artifact_Builder_Agent",
        expected_output="A comprehensive list of fake results",
    )
    agent = ArtifactAgent(config=agent_configs["Artifact_Builder_Agent"])

    @pytest.mark.parametrize(
        "result", [("A result"), ([AIMessage("A"), AIMessage("result")]), ([{"text": "A result"}])]
    )
    def test_post_processing(self, result, fake_state):
        # same post processing as Worker agent but update on different property
        task = self.task
        response = self.agent.post_processing().invoke((fake_state, result, task))
        assert response["artifact_criteria"] == "A result"
