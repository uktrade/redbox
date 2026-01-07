import logging
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
from redbox.transform import bedrock_tokeniser, truncate_to_tokens

log = logging.getLogger(__name__)


class WorkerAgent(Agent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.task = None

    def reading_task_info(self):
        @RunnableLambda
        def _reading_task_info(state: RedboxState):
            """
            Reading in task information sent from the planner
            """
            agent_options = {agent.name: agent.name for agent in state.request.ai_settings.worker_agents}
            ConfiguredAgentTask, _ = configure_agent_task_plan(agent_options)
            parser = ClaudeParser(pydantic_object=ConfiguredAgentTask)
            try:
                self.task = parser.parse(state.last_message.content)
            except JSONDecodeError as e:
                log.exception(f"Cannot parse task in {self.config.name}: {e}")
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
            state = input[0]
            result = input[1]
            if isinstance(result, str):
                log.warning(f"[{self.config.name}] Using raw string result.")
                result_content = result
            elif isinstance(result, list) and isinstance(result[0], dict):
                log.warning(f"[{self.config.name}] Using raw string in a list as result.")
                result_content = result[0].get("text", "")
            elif isinstance(result, list):
                log.warning(f"[{self.config.name}] Aggregating list of tool results...")
                result_content = []
                current_token_counts = 0

                for res in result:
                    token_count = bedrock_tokeniser(res.content)
                    log.warning(f"[{self.config.name}] Tool response token count: {token_count}")

                    # If adding this whole piece still fits, append normally
                    if current_token_counts + token_count <= self.config.max_tokens:
                        result_content.append(res.content)
                        current_token_counts += token_count
                    else:
                        # If no room, add only what fits
                        remaining_tokens = self.config.max_tokens - current_token_counts
                        if remaining_tokens > 0:
                            log.warning(
                                f"[{self.config.name}] Truncating tool output to fit remaining token budget ({remaining_tokens})."
                            )
                            truncated = truncate_to_tokens(res.content, remaining_tokens)
                            result_content.append(truncated)
                            current_token_counts += bedrock_tokeniser(truncated)
                        else:
                            log.warning(
                                f"[{self.config.name}] No remaining token budget ({self.config.max_tokens}). Skipping."
                            )
                        break  # Max reached — stop processing further results
                result_content = " ".join(result_content)
            else:
                log.error(f"[{self.config.name}] Worker agent return incompatible data type {type(result)}")
                raise TypeError("Invalid tool result type")
            state.agents_results = f"<{self.config.name}_Result>{result_content}</{self.config.name}_Result>"
            state.tasks_evaluator = self.task.task + "\n" + self.task.expected_output
            return state

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
            )
            # worker_agent = llm_call(agent_config=self.config)
            log.warning(f"[{self.config.name}] Invoking worker agent...")
            ai_msg = worker_agent.invoke(state)

            log.warning(f"[{self.config.name}] Worker agent output:\n{ai_msg}")

            # --- RUN TOOLS IN PARALLEL ---
            log.warning(f"[{self.config.name}] Running tools via run_tools_parallel...")

            result = run_tools_parallel(ai_msg, self.config.tools, state)
            return (state, result)

        return _core_task.with_retry(stop_after_attempt=3)

    def execute(self):
        return (
            self.reading_task_info()
            | RunnableParallel(_=self.log_agent_activity(), result=self.core_task() | self.post_processing())
            | (lambda x: x["result"])
        )
