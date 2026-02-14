import logging
from abc import ABC, abstractmethod

from econ_sim.llm.parser import ResponseParser

logger = logging.getLogger(__name__)


class BaseEconomicAgent(ABC):
    """Abstract base for all economic agents (Government, Firm, Worker).

    Each agent maintains a message history that accumulates across rounds,
    giving the LLM full context of how the economy has evolved.
    """

    def __init__(self, name: str, agent_type: str, llm_client, config):
        self.name = name
        self.agent_type = agent_type
        self.llm = llm_client
        self.config = config
        self.message_history: list[dict] = []
        self.action_log: list[dict] = []

    @abstractmethod
    def build_system_prompt(self, world_state: dict) -> str:
        """Construct the system prompt defining this agent's identity and rules."""

    @abstractmethod
    def build_round_prompt(self, world_state: dict, round_num: int) -> str:
        """Construct the user message for this round's decision."""

    @abstractmethod
    def validate_action(self, action: dict) -> dict:
        """Validate and clamp the parsed action to legal values."""

    def decide(self, world_state: dict, round_num: int, max_attempts: int = 3) -> dict:
        """Send context to LLM, parse response, return validated action."""
        system_prompt = self.build_system_prompt(world_state)
        round_prompt = self.build_round_prompt(world_state, round_num)

        self.message_history.append({"role": "user", "content": round_prompt})

        for attempt in range(max_attempts):
            try:
                response_text = self.llm.call(system_prompt, self.message_history)
                self.message_history.append({"role": "assistant", "content": response_text})
                action = ResponseParser.extract_json(response_text)
                validated = self.validate_action(action)
                self.action_log.append({
                    "round": round_num,
                    "action": validated,
                    "reasoning": response_text,
                })
                return validated
            except Exception as e:
                logger.warning(f"{self.name} action parse attempt {attempt + 1} failed: {e}")
                # Remove the failed assistant message so we can retry
                if self.message_history and self.message_history[-1]["role"] == "assistant":
                    self.message_history.pop()
                if attempt == max_attempts - 1:
                    logger.error(f"{self.name} falling back to default action")
                    default = self.default_action()
                    self.action_log.append({
                        "round": round_num,
                        "action": default,
                        "reasoning": "[FALLBACK — parse failed]",
                    })
                    return default

    @abstractmethod
    def default_action(self) -> dict:
        """Return a safe default action when LLM parsing fails."""

    def receive_notification(self, message: str):
        """Append environment feedback to message history."""
        self.message_history.append({"role": "user", "content": message})
