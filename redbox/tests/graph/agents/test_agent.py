import pytest
from langchain_core.messages import AIMessage
from pytest_mock import MockerFixture

from redbox.graph.agents.configs import AgentConfig, PromptConfig, PromptVariable
from redbox.graph.agents.workers import WorkerAgent
from redbox.graph.nodes.tools import build_search_wikipedia_tool
from redbox.models.chain import RedboxState, TaskStatus
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

    @pytest.mark.parametrize(
        "success, fake_state_fixture",
        [("fail", "fake_state"), ("success", "fake_state_with_plan")],
        indirect=["fake_state_fixture"],
    )
    def test_reading_task_info(self, success, fake_state_fixture):
        self.worker.reading_task_info().invoke(fake_state_fixture)
        if success == "success":
            assert self.worker.task is not None
        else:
            assert self.worker.task is None

    @pytest.mark.parametrize(
        "result", [("A result"), ([AIMessage("A"), AIMessage("result")]), ([{"text": "A result"}])]
    )
    def test_post_processing(self, result, fake_state_with_plan):
        self.worker.task = fake_state_with_plan.agent_plans.tasks[0]
        response = self.worker.post_processing().invoke((fake_state_with_plan, result))
        assert response["agents_results"] == {
            self.worker.task.id: AIMessage(
                content=f"<{self.worker.config.name}_Result>A result</{self.worker.config.name}_Result>"
            )
        }
        assert response["tasks_evaluator"] == self.worker.task.task + "\n" + self.worker.task.expected_output

    @pytest.mark.parametrize(
        "AI_response",
        [
            ("Here is your fake response"),
            (AIMessage(content="Here is your fake response")),
            (WORKER_RESPONSE_WITH_TOOL),
        ],
    )
    def test_core_task(self, AI_response, fake_state_with_plan, mocker: MockerFixture):
        # mock LLM call
        llm_mock = mocker.patch("redbox.chains.runnables.get_chat_llm")
        llm_mock.return_value = GenericFakeChatModelWithTools(messages=iter([AI_response]))
        # mock tool response
        mocker.patch(
            "redbox.graph.agents.workers.run_tools_parallel",
            return_value=[AIMessage(content="Here is your fake response")],
        )
        self.worker.task = fake_state_with_plan.agent_plans.tasks[0]
        response = self.worker.core_task().invoke(fake_state_with_plan)
        if isinstance(response, str):
            response == "Here is your fake response"
        elif isinstance(response, list):
            assert response[0].content == "Here is your fake response"

    def test_execute(self, fake_state_with_plan: RedboxState, mocker: MockerFixture):
        # mock LLM call
        llm_mock = mocker.patch("redbox.chains.runnables.get_chat_llm")
        llm_mock.return_value = GenericFakeChatModelWithTools(messages=iter([WORKER_RESPONSE_WITH_TOOL]))
        # mock tool response
        mocker.patch(
            "redbox.graph.agents.workers.run_tools_parallel",
            return_value=[AIMessage(content="Here is your fake response")],
        )
        self.worker.task = fake_state_with_plan.agent_plans.tasks[0]
        response = self.worker.execute().invoke(fake_state_with_plan)
        assert response["agents_results"] == {
            self.worker.task.id: AIMessage(
                content=f"<{self.worker.config.name}_Result>Here is your fake response</{self.worker.config.name}_Result>"
            )
        }
        assert response["tasks_evaluator"] == self.worker.task.task + "\n" + self.worker.task.expected_output
        assert response["agent_plans"].get_task_status(self.worker.task.id) == TaskStatus.COMPLETED
