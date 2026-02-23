import time
import logging
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)


class AnthropicLLMClient:
    """Wrapper around the Anthropic Messages API with retry/backoff and cost tracking."""

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

        # Cost tracking (input/output tokens per model)
        self.call_log: list[dict] = []

    def call(
        self,
        system_prompt: str,
        messages: list[dict],
        model_override: Optional[str] = None,
        max_tokens_override: Optional[int] = None,
    ) -> str:
        """Call Claude with a system prompt and message history.

        Args:
            system_prompt: The agent's identity / rules / context.
            messages: List of {"role": "user"|"assistant", "content": str}.
            model_override: Use a different model for this call (e.g. haiku for workers).
            max_tokens_override: Override max tokens for this call.

        Returns:
            The assistant's response text.
        """
        model = model_override or self.model
        max_tokens = max_tokens_override or self.max_tokens

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=messages,
                )
                text = response.content[0].text
                self.call_log.append({
                    "model": model,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                })
                return text
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed ({model}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries")

    def cost_summary(self) -> dict:
        """Return token usage and estimated cost breakdown by model."""
        from collections import defaultdict
        # Approximate pricing per million tokens (input / output)
        PRICING = {
            "claude-opus-4-6":           (15.00, 75.00),
            "claude-sonnet-4-6":         (3.00,  15.00),
            "claude-haiku-4-5-20251001": (0.80,   4.00),
        }
        totals: dict = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
        for entry in self.call_log:
            m = entry["model"]
            totals[m]["calls"] += 1
            totals[m]["input_tokens"] += entry["input_tokens"]
            totals[m]["output_tokens"] += entry["output_tokens"]
            in_price, out_price = PRICING.get(m, (15.0, 75.0))
            totals[m]["cost_usd"] += (
                entry["input_tokens"] * in_price / 1_000_000
                + entry["output_tokens"] * out_price / 1_000_000
            )
        return dict(totals)
