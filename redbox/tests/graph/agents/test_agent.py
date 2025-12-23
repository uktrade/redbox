import pytest
from langchain_core.messages import AIMessage
from pytest_mock import MockerFixture

from redbox.graph.agents.agents import WorkerAgent
from redbox.graph.agents.configs import AgentConfig, PromptConfig
from redbox.models.chain import AgentTaskBase, RedboxState


class TestWorkerAgent:
    propmt_config = PromptConfig(
        system="Fake system prompt", question="Fake question prompt", format=None, prompt_vars=None
    )
    config = AgentConfig(name="Fake_Agent", description="Fake description", prompt=propmt_config, tools=[])
    worker = WorkerAgent(config=config)
    task = AgentTaskBase(
        task="Analyze the user's documents to understand what recommendations are made in the AI Playbook for the UK Government",
        agent="Internal_Retrieval_Agent",
        expected_output="A comprehensive list of key recommendations made in the AI Playbook document",
    )

    @pytest.mark.parametrize("success, task", [("success", task.model_dump_json()), ("fail", "")])
    def test_reading_task_info(self, success, task, fake_state, caplog):
        fake_state.messages = [AIMessage(content=task)]
        self.worker.reading_task_info(fake_state)
        if success == "success":
            assert self.worker.task is not None
        else:
            assert self.worker.task is None

    @pytest.mark.parametrize(
        "result", [("A result"), ([AIMessage("A"), AIMessage("result")]), ([{"text": "A result"}])]
    )
    def test_post_processing(self, result):
        self.worker.task = self.task
        response = self.worker.post_processing(result)
        assert isinstance(response, dict)
        assert (
            response["agents_results"]
            == f"<{self.worker.config.name}_Result>A result</{self.worker.config.name}_Result>"
        )
        assert response["tasks_evaluator"] == self.worker.task.task + "\n" + self.worker.task.expected_output

    def test_build_worker_agent(self, fake_state: RedboxState, mocker: MockerFixture):
        # mock LLM call
        llm_mock = mocker.patch("redbox.chains.runnables.get_chat_llm")
        llm_mock.return_value = AIMessage(content="Here is your fake response")

        fake_state.messages = [AIMessage(content=self.task.model_dump_json())]

        response = self.worker.execute().invoke(fake_state)

        assert response.messages[-1] == "Here is your fake response"
