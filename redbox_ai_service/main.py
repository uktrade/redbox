from fastapi import FastAPI, HTTPException
from redbox.models.chain import RedboxState
from redbox.graph.agents.workers import WorkerAgent
from redbox_ai_service.config import get_agent_config

app = FastAPI(title="Redbox AI Service")


@app.post("/invoke/{agent_name}")
def invoke(agent_name: str, payload: dict):
    """
    Invoke a specific worker agent by name - gets payload of RedboxState
    If _task_info exists, then it puts it into execution
    """
    try:
        state = RedboxState(**payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid state: {e}")

    try:
        config = get_agent_config(agent_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown agent {agent_name}")

    task_info = payload.get("_task_info")
    if task_info:
        state._task_info = task_info

    agent = WorkerAgent(config)
    try:
        result = agent.execute().invoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {e}")

    return result
