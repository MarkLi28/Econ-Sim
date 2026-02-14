import json
import re
import logging

logger = logging.getLogger(__name__)


class ResponseParser:
    """Extract structured JSON actions from LLM free-text responses."""

    @staticmethod
    def extract_json(text: str) -> dict:
        """Find a JSON object in the response text.

        Tries ```json code blocks first, then falls back to finding
        the outermost { ... } in the text.
        """
        # Try fenced code block first
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # Fallback: find the last { ... } block (agents often reason before JSON)
        candidates = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL))
        if candidates:
            return json.loads(candidates[-1].group(0))

        raise ValueError(f"No JSON found in response: {text[:200]}...")

    @staticmethod
    def validate_government_action(action: dict, config) -> dict:
        """Validate and clamp government action fields."""
        lo_i, hi_i = config.income_tax_rate_range
        lo_c, hi_c = config.corporate_tax_rate_range

        action["income_tax_rate"] = max(lo_i, min(hi_i, float(action.get("income_tax_rate", lo_i))))
        action["corporate_tax_rate"] = max(lo_c, min(hi_c, float(action.get("corporate_tax_rate", lo_c))))

        spending = action.get("spending", {})
        for key in ("infrastructure", "transfers_to_workers", "subsidies_to_firms"):
            spending[key] = max(0.0, float(spending.get(key, 0.0)))
        action["spending"] = spending
        return action

    @staticmethod
    def validate_firm_action(action: dict, config) -> dict:
        """Validate and clamp firm action fields."""
        action["wage_offer"] = max(0.0, float(action.get("wage_offer", config.base_wage_level)))
        action["price"] = max(0.01, float(action.get("price", config.base_price_level)))
        action["hiring_target"] = max(0, int(float(action.get("hiring_target", 1))))

        if config.min_wage_enabled:
            action["wage_offer"] = max(action["wage_offer"], config.min_wage)
        if config.price_controls_enabled:
            action["price"] = min(action["price"], config.max_price)

        return action

    @staticmethod
    def validate_worker_action(action: dict, firm_names: list[str]) -> dict:
        """Validate worker action fields."""
        action["spending_fraction"] = max(0.0, min(1.0, float(action.get("spending_fraction", 0.5))))

        employer = action.get("chosen_employer", "none")
        if isinstance(employer, str):
            employer = employer.strip()
        if employer not in firm_names:
            action["chosen_employer"] = None
        else:
            action["chosen_employer"] = employer

        return action
