from json import JSONDecodeError

from langchain_core.runnables import RunnableLambda, RunnableParallel

from redbox.chains.parser import ClaudeParser
from redbox.chains.runnables import create_chain_agent
from redbox.graph.agents.base import Agent
from redbox.graph.agents.configs import AgentConfig
from redbox.graph.nodes.processes import build_activity_log_node
from redbox.graph.nodes.sends import run_tools_parallel
from redbox.models.chain import RedboxState, configure_agent_task_plan
from redbox.models.graph import RedboxActivityEvent
from redbox.transform import join_result_with_token_limit


class WorkerAgent(Agent):
    """
    Worker Agent is defined as an agent that performs a given task and store its results in `agents_results` property which can be used by other agents.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.task = None

    def reading_task_info(self):
        @RunnableLambda
        def _reading_task_info(state: RedboxState):
            """
            Reading in task information sent from the planner
            """
            agent_options = state.request.ai_settings.get_worker_agents_options
            ConfiguredAgentTask, _ = configure_agent_task_plan(agent_options)
            parser = ClaudeParser(pydantic_object=ConfiguredAgentTask)
            try:
                self.task = parser.parse(state.last_message.content)
            except JSONDecodeError as e:
                self.logger.exception(f"Cannot parse task in {self.config.name}: {e}")
            return state

        return _reading_task_info

    def log_agent_activity(self):
        @RunnableLambda
        def _log_agent_activity(
            state: RedboxState,
        ):
            """
            log what task the agent is completing
            """
            activity_node = build_activity_log_node(
                RedboxActivityEvent(message=f"{self.config.name} is completing task: {self.task.task}")
            )
            activity_node.invoke(state)

        return _log_agent_activity

    def post_processing(self):
        @RunnableLambda
        def _post_processing(input):
            """
            Processing data from the agent core function.
            """
            _, result = input

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
                "tasks_evaluator": self.task.task + "\n" + self.task.expected_output,
            }

        return _post_processing

    def core_task(self):
        @RunnableLambda
        def _core_task(state: RedboxState):
            worker_agent = create_chain_agent(
                system_prompt=self.config.prompt.get_prompt,
                use_metadata=self.config.prompt.prompt_vars.metadata,
                using_chat_history=self.config.prompt.prompt_vars.chat_history,
                parser=self.config.parser,
                tools=self.config.tools,
                _additional_variables={"task": self.task.task, "expected_output": self.task.expected_output},
                model=self.config.llm_backend,
                use_knowledge_base=self.config.prompt.prompt_vars.knowledge_base_metadata,
            )
            # worker_agent = llm_call(agent_config=self.config)
            self.logger.warning(f"[{self.config.name}] Invoking worker agent...")
            ai_msg = worker_agent.invoke(state)

            self.logger.warning(f"[{self.config.name}] Worker agent output:\n{ai_msg}")

            # --- RUN TOOLS IN PARALLEL ---
            self.logger.warning(f"[{self.config.name}] Running tools via run_tools_parallel...")

            result = run_tools_parallel(ai_msg, self.config.tools, state)
            return (state, result)

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

    def core_only_graph(self):
        return self.reading_task_info() | self.core_task()


class ToolWorkerAgent(WorkerAgent):
    def core_task(self):
        @RunnableLambda
        def _core_task(state: RedboxState):
            worker_agent = create_chain_agent(
                system_prompt=self.config.prompt.get_prompt,
                use_metadata=self.config.prompt.prompt_vars.metadata,
                using_chat_history=self.config.prompt.prompt_vars.chat_history,
                parser=self.config.parser,
                tools=self.config.tools,
                _additional_variables={"task": self.task.task, "expected_output": self.task.expected_output},
                model=self.config.llm_backend,
                use_knowledge_base=self.config.prompt.prompt_vars.knowledge_base_metadata,
            )
            # worker_agent = llm_call(agent_config=self.config)
            self.logger.warning(f"[{self.config.name}] Invoking worker agent...")
            ai_msg = worker_agent.invoke(state)

            self.logger.warning(f"[{self.config.name}] Worker agent output:\n{ai_msg}")

            # --- RUN TOOLS IN PARALLEL ---
            self.logger.warning(f"[{self.config.name}] Running tools via run_tools_parallel...")

            result = run_tools_parallel(ai_msg, self.config.tools, state)
            return result

        return _core_task.with_retry(stop_after_attempt=3)
