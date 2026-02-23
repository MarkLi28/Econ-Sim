import json
import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Keys workers can allocate discretionary budget toward
DISCRETIONARY_KEYS = ["extra_food", "extra_shelter", "technology", "savings"]


class ResponseParser:
    """Extract structured JSON actions from LLM free-text responses."""

    @staticmethod
    def extract_json(text: str) -> dict:
        """Find a JSON object in the response text.

        Tries ```json code blocks first, then falls back to the outermost { ... }.
        """
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Fallback: last { ... } block (agents often reason before outputting JSON)
        candidates = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL))
        if candidates:
            try:
                return json.loads(candidates[-1].group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"No JSON found in response: {text[:300]}")

    # ------------------------------------------------------------------
    # Government
    # ------------------------------------------------------------------

    @staticmethod
    def validate_government_action(action: dict, config) -> dict:
        """Validate and clamp government action fields."""
        # Tax rates — bounded 0-1
        action["income_tax_rate"] = max(0.0, min(1.0, float(action.get("income_tax_rate", 0.15))))
        action["corporate_tax_rate"] = max(0.0, min(1.0, float(action.get("corporate_tax_rate", 0.15))))

        # Spending buckets
        spending = action.get("spending", {})
        for key in ("infrastructure", "transfers_to_workers", "subsidies_to_firms"):
            spending[key] = max(0.0, float(spending.get(key, 0.0)))
        action["spending"] = spending

        # Per-industry policy (optional; government may choose to target specific industries)
        industry_policies = action.get("industry_policies", {})
        validated_ip = {}
        for industry_name, policy in industry_policies.items():
            if not isinstance(policy, dict):
                continue
            validated_ip[industry_name] = {
                "price_cap": float(policy.get("price_cap", -1.0)),    # -1 = no cap
                "wage_floor": float(policy.get("wage_floor", -1.0)),  # -1 = no floor
                "subsidy": max(0.0, float(policy.get("subsidy", 0.0))),
                "regulation_strength": max(0.0, min(1.0, float(policy.get("regulation_strength", 0.0)))),
            }
        action["industry_policies"] = validated_ip

        return action

    # ------------------------------------------------------------------
    # Firms — industry-specific schemas
    # ------------------------------------------------------------------

    @staticmethod
    def validate_firm_action(action: dict, config, industry_name: str, base_price: float) -> dict:
        """Validate firm action. Schema varies slightly by industry."""
        # Universal fields
        action["wage_offer"] = max(0.0, float(action.get("wage_offer", config.base_wage_level)))
        action["price"] = max(0.01, float(action.get("price", base_price)))
        action["hiring_target"] = max(0, int(float(action.get("hiring_target", 1))))

        # Apply axis-driven constraints
        # Wage floors scale with wage_control_strength
        effective_floor = config.base_wage_level * 0.5 * config.wage_control_strength
        action["wage_offer"] = max(action["wage_offer"], effective_floor)

        # Price caps scale with price_control_strength
        if config.price_control_strength > 0.5:
            cap = base_price * (2.0 + (1.0 - config.price_control_strength) * 3.0)
            action["price"] = min(action["price"], cap)

        # Lobbying fields (present in all firms; clamped by meta_game_constraint)
        lobby_raw = max(0.0, float(action.get("lobby_spending", 0.0)))
        action["lobby_spending"] = lobby_raw   # clamped against capital in environment
        action["lobby_message"] = str(action.get("lobby_message", ""))[:500]

        # Manufacturing-only: B2B pricing and allocation split
        if industry_name == "manufacturing":
            action["b2b_price"] = max(0.01, float(action.get("b2b_price", base_price * 0.8)))
            action["b2b_allocation_fraction"] = max(
                0.0, min(1.0, float(action.get("b2b_allocation_fraction", 0.5)))
            )

        # Housing/Technology: input budget
        if industry_name in ("housing", "technology"):
            action["input_budget"] = max(0.0, float(action.get("input_budget", 100.0)))

        return action

    # ------------------------------------------------------------------
    # Workers — employment POV
    # ------------------------------------------------------------------

    @staticmethod
    def validate_worker_employment(action: dict, firm_names: List[str]) -> dict:
        """Validate worker employment decision."""
        employer = action.get("chosen_employer", "")
        if isinstance(employer, str):
            employer = employer.strip()
        action["chosen_employer"] = employer if employer in firm_names else None
        return action

    # ------------------------------------------------------------------
    # Workers — consumer POV
    # ------------------------------------------------------------------

    @staticmethod
    def validate_consumer_allocation(action: dict) -> dict:
        """Validate discretionary spending allocation.

        Expected keys: extra_food, extra_shelter, technology, savings.
        Values are fractions (must sum to ~1.0; we normalize if not).
        """
        alloc = action.get("discretionary_allocation", {})
        if not isinstance(alloc, dict):
            alloc = {}

        cleaned = {}
        for key in DISCRETIONARY_KEYS:
            cleaned[key] = max(0.0, float(alloc.get(key, 0.0)))

        total = sum(cleaned.values())
        if total <= 0:
            # Default: split evenly between savings and technology
            cleaned = {"extra_food": 0.1, "extra_shelter": 0.1, "technology": 0.3, "savings": 0.5}
        else:
            # Normalize
            for key in cleaned:
                cleaned[key] /= total

        action["discretionary_allocation"] = cleaned
        return action
