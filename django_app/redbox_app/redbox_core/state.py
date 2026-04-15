import asyncio
from dataclasses import dataclass, field

from redbox.graph.agents.configs import AgentConfig


@dataclass
class SharedConsumerState:
    graph: object | None = None
    graph_lock: asyncio.Lock | None = None

    all_chunks_retriever: object | None = None
    parameterised_retriever: object | None = None
    metadata_retriever: object | None = None
    embedding_model: object | None = None

    agent_configs: dict[str, AgentConfig] = field(default_factory=dict)

    def get_lock(self) -> asyncio.Lock:
        if self.graph_lock is None:
            self.graph_lock = asyncio.Lock()
        return self.graph_lock
