from redbox.graph.agents.configs import agent_configs, AgentConfig


def get_agent_config(name: str) -> AgentConfig:
    """
    Fetch one of the pre-defined AgentConfig objects
    """
    if name not in agent_configs:
        raise KeyError(f"Agent {name} not found")

    return agent_configs[name]
