from langchain_core.runnables import RunnableLambda

from redbox.chains.runnables import create_chain_agent
from redbox.graph.agents.workers import WorkerAgent


class ArtifactAgent(WorkerAgent):
    def core_task(self):
        @RunnableLambda
        def _core_task(input):
            state, task = input

            artifact_files = [
                kb_file
                for kb_file in state.request.knowledge_base_s3_keys
                if "artifact" in kb_file.split("/")[-1].lower()
            ]

            worker_agent = create_chain_agent(
                system_prompt=self.config.prompt.get_prompt,
                use_metadata=self.config.prompt.prompt_vars.metadata,
                using_chat_history=self.config.prompt.prompt_vars.chat_history,
                parser=self.config.parser,
                tools=self.config.tools,
                _additional_variables={
                    "task": task.task,
                    "expected_output": task.expected_output,
                    "artifact_files": artifact_files,
                },
                model=self.config.llm_backend,
                use_knowledge_base=self.config.prompt.prompt_vars.knowledge_base_metadata,
            )
            result = self._agent_invocation(agent=worker_agent, state=state)
            return (state, result, task)

        return _core_task

    def post_processing(self):
        @RunnableLambda
        def _post_processing(input):
            """
            Processing data from the agent core function.
            """
            _, result, _ = input
            result_content = self._processing(result)
            return {"artifact_criteria": f"{result_content}"}

        return _post_processing
