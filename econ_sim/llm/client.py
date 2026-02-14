import time
import logging

import anthropic

logger = logging.getLogger(__name__)


class AnthropicLLMClient:
    """Wrapper around the Anthropic Messages API with retry/backoff."""

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def call(self, system_prompt: str, messages: list[dict]) -> str:
        """Call Claude with a system prompt and message history.

        Args:
            system_prompt: The agent's identity / rules / context.
            messages: List of {"role": "user"|"assistant", "content": str}.

        Returns:
            The assistant's response text.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=messages,
                )
                return response.content[0].text
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries")
