import httpx
import logging

from json import JSONDecodeError

from langchain_core.runnables import RunnableLambda, RunnableParallel

from redbox.chains.parser import ClaudeParser
from redbox.graph.agents.base import Agent
from redbox.graph.agents.configs import AgentConfig
from redbox.graph.nodes.processes import build_activity_log_node
from redbox.graph.nodes.sends import run_tools_parallel
from redbox.models.chain import RedboxState, configure_agent_task_plan
from redbox.models.graph import RedboxActivityEvent
from redbox.transform import join_result_with_token_limit
from redbox.models.settings import get_settings

settings = get_settings()


class RemoteWorkerAgent(Agent):
    """
    For the FastAPI service.
    """

    def __init__(self, agent_name: str, base_url: str):
        self.agent_name = agent_name
        self.base_url = base_url.rstrip("/")
        self.logger = logging.getLogger(f"RemoteWorkerAgent[{agent_name}]")

    def invoke(self, state: RedboxState, task=None):
        """
        Invokes remote agent.
        `task` should be the task object containing `task.task` and `task.expected_output`
        """
        url = f"{self.base_url}/invoke/{self.agent_name}"

        payload = state.model_dump()
        if task:
            payload["_task_info"] = {
                "task": task.task,
                "expected_output": task.expected_output,
            }

        self.logger.warning(f"Invoking remote agent {self.agent_name} at {url}")
        try:
            response = httpx.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            self.logger.warning(f"Remote agent {self.agent_name} returned successfully")
            return result
        except httpx.HTTPError as e:
            self.logger.exception(f"Remote agent call failed: {e}")
            raise RuntimeError(f"Remote agent call failed: {e}") from e


class WorkerAgent(Agent):
    """
    Worker Agent is defined as an agent that performs a given task and store its results in `agents_results` property which can be used by other agents.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)

    def reading_task_info(self):
        @RunnableLambda
        def _reading_task_info(state: RedboxState):
            """
            Reading in task information sent from the planner
            """
            agent_options = state.request.ai_settings.get_worker_agents_options
            ConfiguredAgentTask, _ = configure_agent_task_plan(agent_options)
            parser = ClaudeParser(pydantic_object=ConfiguredAgentTask)
            task = None
            try:
                task = parser.parse(state.last_message.content)
                self.logger.warning(f"Parsing task {state.last_message.content}")
            except JSONDecodeError as e:
                self.logger.exception(f"Cannot parse task in {self.config.name}: {e}")
            return state, task

        return _reading_task_info

    def log_agent_activity(self):
        @RunnableLambda
        def _log_agent_activity(
            input,
        ):
            """
            log what task the agent is completing
            """
            state, task = input
            self.logger.warning(f"{self.config.name} is completing task: {task.task}")
            activity_node = build_activity_log_node(
                RedboxActivityEvent(message=f"{self.config.name} is completing task: {task.task}")
            )
            activity_node.invoke(state)

        return _log_agent_activity

    def post_processing(self):
        @RunnableLambda
        def _post_processing(input):
            """
            Processing data from the agent core function.
            """
            _, result, task = input

            if isinstance(result, str):
                self.logger.warning(f"[{self.config.name}] Using raw string result.")
                result_content = result
            elif isinstance(result, list) and isinstance(result[0], dict):
                self.logger.warning(f"[{self.config.name}] Using raw string in a list as result.")
                result_content = result[0].get("text", "")
            elif isinstance(result, list):
                self.logger.warning(f"[{self.config.name}] Aggregating list of tool results...")
                result_content = join_result_with_token_limit(
                    result=result, max_tokens=self.config.agents_max_tokens, log_stub=f"[{self.config.name}]"
                )
            else:
                self.logger.error(f"[{self.config.name}] Worker agent return incompatible data type {type(result)}")
                raise TypeError("Invalid tool result type")
            self.logger.warning(f"[{self.config.name}] Completed agent run.")

            return {
                "agents_results": f"<{self.config.name}_Result>{result_content}</{self.config.name}_Result>",
                "tasks_evaluator": task.task + "\n" + task.expected_output,
            }

        return _post_processing

    def core_task(self):
        @RunnableLambda
        def _core_task(input):
            state, task = input

            self.logger.warning(f"[{self.config.name}] Preparing worker agent...")

            # if getattr(self.config, "use_remote_agent", False):
            from redbox.models.settings import get_settings

            settings = get_settings()
            worker_agent = RemoteWorkerAgent(agent_name=self.config.name, base_url=settings.ai_service_url)
            ai_msg = worker_agent.invoke(state, task=task)
            # else:
            #     worker_agent = create_chain_agent(
            #         system_prompt=self.config.prompt.get_prompt,
            #         use_metadata=self.config.prompt.prompt_vars.metadata,
            #         using_chat_history=self.config.prompt.prompt_vars.chat_history,
            #         parser=self.config.parser,
            #         tools=self.config.tools,
            #         _additional_variables={"task": task.task, "expected_output": task.expected_output},
            #         model=self.config.llm_backend,
            #         use_knowledge_base=self.config.prompt.prompt_vars.knowledge_base_metadata,
            #     )
            #     ai_msg = worker_agent.invoke(state)

            self.logger.warning(f"[{self.config.name}] Worker agent output:\n{ai_msg}")

            # --- RUN TOOLS IN PARALLEL ---
            self.logger.warning(f"[{self.config.name}] Running tools via run_tools_parallel...")
            result = run_tools_parallel(ai_msg, self.config.tools, state)

            return (state, result, task)

        return _core_task.with_retry(stop_after_attempt=3)

    def execute(self):
        """
        Execution flow of the worker agent.
        """
        return (
            self.reading_task_info()
            | RunnableParallel(_=self.log_agent_activity(), result=self.core_task() | self.post_processing())
            | (lambda x: x["result"])
        )
