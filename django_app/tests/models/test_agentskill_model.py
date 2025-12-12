from redbox_app.redbox_core.models import AgentSkill


def test_create_agentskill(default_agent, default_skill):
    agent_skill = AgentSkill(agent=default_agent, skill=default_skill)
    agent_skill.save()

    assert agent_skill.__str__() == f"{default_agent.name} - {default_skill.name}"
