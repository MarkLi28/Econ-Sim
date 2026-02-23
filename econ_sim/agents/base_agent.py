import logging
from abc import ABC, abstractmethod
from typing import Optional

from econ_sim.llm.parser import ResponseParser

logger = logging.getLogger(__name__)

# Keep only this many rounds of history to avoid ballooning token counts
HISTORY_WINDOW = 3


class BaseEconomicAgent(ABC):
    """Abstract base for all economic agents (Government, Firm, Worker).

    Each agent maintains a message history that accumulates across rounds,
    giving the LLM context of how the economy has evolved. History is pruned
    to the last HISTORY_WINDOW round-pairs to control token costs.

    Agents declare a model_tier ("opus"|"sonnet"|"haiku") which is passed
    through to the LLM client so different agent types use appropriately
    capable models.
    """

    model_tier: str = "opus"   # override in subclasses

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

    @abstractmethod
    def default_action(self) -> dict:
        """Return a safe default action when LLM parsing fails."""

    def _model(self) -> str:
        return self.config.model_for_tier(self.model_tier)

    def _pruned_history(self) -> list[dict]:
        """Return message history trimmed to the last HISTORY_WINDOW round-pairs."""
        # Each round contributes one user + one assistant message (2 entries)
        keep = HISTORY_WINDOW * 2
        if len(self.message_history) > keep:
            return self.message_history[-keep:]
        return self.message_history

    def decide(
        self,
        world_state: dict,
        round_num: int,
        max_attempts: int = 3,
        system_prompt_override: Optional[str] = None,
        round_prompt_override: Optional[str] = None,
    ) -> dict:
        """Send context to LLM, parse response, return validated action."""
        system_prompt = system_prompt_override or self.build_system_prompt(world_state)
        round_prompt = round_prompt_override or self.build_round_prompt(world_state, round_num)

        self.message_history.append({"role": "user", "content": round_prompt})
        history = self._pruned_history()

        for attempt in range(max_attempts):
            try:
                response_text = self.llm.call(
                    system_prompt,
                    history,
                    model_override=self._model(),
                )
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
                    # Re-append a placeholder so history stays coherent
                    self.message_history.append({
                        "role": "assistant",
                        "content": "[Used default action due to parse failure]",
                    })
                    return default

    def receive_notification(self, message: str):
        """Append environment feedback to message history (no LLM call)."""
        self.message_history.append({"role": "user", "content": f"[ENVIRONMENT] {message}"})
