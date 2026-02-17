from langchain_core.runnables import RunnableLambda

from redbox.graph.agents.workers import WorkerAgent


class ArtifactAgent(WorkerAgent):
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
