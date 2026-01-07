import logging
from abc import ABC, abstractmethod

from redbox.graph.agents.configs import AgentConfig

log = logging.getLogger(__name__)


class Agent(ABC):
    """
    This is abtract agent class
    """

    def __init__(self, config):
        self.config: AgentConfig = config

    @abstractmethod
    def execute(self):
        """
        This function return the steps/flow of this agent. For example core_task() | post_proecssing()
        """
        pass
