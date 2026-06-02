from langchain_core.runnables import RunnableLambda

from redbox.chains.runnables import create_chain_agent
from redbox.graph.agents.workers import WorkerAgent
from redbox.models.chain import TaskStatus


class ArtifactAgent(WorkerAgent):
    def core_task(self):
        @RunnableLambda
        def _core_task(input):
            state, task = input

            artifact_files = [
                kb_file
                for kb_file in state.request.knowledge_base_s3_keys
                if kb_file.split("/")[-1].lower().startswith("artifact")
            ]

            worker_agent = create_chain_agent(
                config=self.config,
                _additional_variables={
                    "task": task.task,
                    "expected_output": task.expected_output,
                    "artifact_files": artifact_files,
                },
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
            state, result, task = input
            result_content = self._processing(result)
            return {
                "artifact_criteria": f"{result_content}",
                "agent_plans": state.agent_plans.update_task_status(task.id, TaskStatus.COMPLETED),
            }

        return _post_processing
