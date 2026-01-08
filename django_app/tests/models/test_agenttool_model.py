from redbox_app.redbox_core.models import AgentTool


def test_create_agenttool(default_agent, default_tool):
    agent_tool = AgentTool(agent=default_agent, tool=default_tool)
    agent_tool.save()

    assert agent_tool.__str__() == f"{default_agent.name} - {default_tool.name}"
