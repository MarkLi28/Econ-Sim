import logging
from typing import Optional

from econ_sim.agents.base_agent import BaseEconomicAgent
from econ_sim.config import Industry, WorkerProfile
from econ_sim.llm.parser import ResponseParser, DISCRETIONARY_KEYS

logger = logging.getLogger(__name__)


class WorkerAgent(BaseEconomicAgent):
    """A representative worker archetype with two decision-making perspectives.

    One WorkerAgent represents `profile.representative_count` real workers.
    Decisions are made for the archetype and replicated (with small noise) to
    all represented workers in the environment layer.

    1. WORKER POV  — employment choice (before labor market clears)
    2. CONSUMER POV — spending allocation (after production, income known)
    """

    model_tier = "haiku"   # workers use the cheapest capable model

    def __init__(
        self,
        name: str,
        llm_client,
        config,
        profile: WorkerProfile,
    ):
        super().__init__(name, "worker", llm_client, config)
        self.profile = profile
        self.skill_level: float = profile.skill_level
        self.representative_count: int = profile.representative_count
        self.savings: float = profile.initial_savings

        # Updated each round by environment
        self.employer: Optional[str] = None
        self.industry: Optional[Industry] = None
        self.wage: float = 0.0
        self.income: float = 0.0          # gross wage earned this round

        # Necessity tracking (updated by goods market)
        self.food_consumed: float = 0.0
        self.shelter_consumed: float = 0.0
        self.suffering_score: float = 0.0   # cumulative unmet needs
        self.productivity_modifier: float = 1.0  # reduced when needs unmet

        # Discretionary allocation from last consumer decision
        self.last_allocation: dict = {k: 0.0 for k in DISCRETIONARY_KEYS}

    # ------------------------------------------------------------------
    # BaseEconomicAgent interface — routes to Worker POV
    # ------------------------------------------------------------------

    def build_system_prompt(self, world_state: dict) -> str:
        return self._worker_system_prompt(world_state)

    def build_round_prompt(self, world_state: dict, round_num: int) -> str:
        return self._worker_round_prompt(world_state, round_num)

    def validate_action(self, action: dict) -> dict:
        # Worker decisions always go through decide_employment() or decide_consumption()
        # directly — this base-class method is not called in normal operation.
        return ResponseParser.validate_worker_employment(action, [])

    def default_action(self) -> dict:
        return {"chosen_employer": None}

    # ------------------------------------------------------------------
    # Worker POV — employment decision
    # ------------------------------------------------------------------

    def decide_employment(self, world_state: dict, round_num: int) -> dict:
        """LLM call for employment decision. Called before labor market clears."""
        system = self._worker_system_prompt(world_state)
        prompt = self._worker_round_prompt(world_state, round_num)
        firm_names = [f["name"] for f in world_state.get("firms", [])]

        self.message_history.append({"role": "user", "content": prompt})
        history = self._pruned_history()

        for attempt in range(3):
            try:
                text = self.llm.call(system, history, model_override=self._model())
                self.message_history.append({"role": "assistant", "content": text})
                action = ResponseParser.extract_json(text)
                validated = ResponseParser.validate_worker_employment(action, firm_names)
                self.action_log.append({"round": round_num, "phase": "worker", "action": validated, "reasoning": text})
                return validated
            except Exception as e:
                logger.warning(f"{self.name} employment parse attempt {attempt+1} failed: {e}")
                if self.message_history and self.message_history[-1]["role"] == "assistant":
                    self.message_history.pop()

        default = {"chosen_employer": None}
        self.message_history.append({"role": "assistant", "content": "[FALLBACK]"})
        self.action_log.append({"round": round_num, "phase": "worker", "action": default, "reasoning": "[FALLBACK]"})
        return default

    def _worker_system_prompt(self, world_state: dict) -> str:
        sys_label = world_state.get("system_label", "the economy") if world_state else "the economy"
        return f"""You are {self.name}, a {self.profile.name} worker in {sys_label}.

YOUR UTILITY FUNCTION (maximize across the simulation):
  U = 0.40 × consumption_satisfaction
    + 0.30 × financial_security (savings cushion)
    + 0.20 × basic_needs_met (food + shelter covered)
    + 0.10 × job_satisfaction (skill match, stability)

WORKER PROFILE:
  Skill level: {self.skill_level:.1f} (higher = more productive, preferred by employers)
  You represent {self.representative_count} workers with similar backgrounds.
  Industry preferences: {', '.join(p.value for p in self.profile.industry_preferences)}

RIGHT NOW you are making your EMPLOYMENT DECISION only.
Choose which firm to apply to. The labor market will then clear — you may or may not
get the job depending on competition and the firm's hiring target.

Reason through your options, then output JSON: {{"chosen_employer": "<firm_name or none>"}}
"""

    def _worker_round_prompt(self, world_state: dict, round_num: int) -> str:
        if not world_state:
            return f"Round {round_num}: choose employer."
        firms = world_state.get("firms", [])
        govpol = world_state.get("government_policy", {})

        # Group firms by industry for clarity
        by_industry: dict = {}
        for f in firms:
            ind = f.get("industry", "unknown")
            by_industry.setdefault(ind, []).append(f)

        firm_lines = []
        for ind, fs in by_industry.items():
            firm_lines.append(f"  [{ind.upper()}]")
            for f in fs:
                firm_lines.append(
                    f"    {f['name']}: wage=${f.get('wage', 0):.0f}, "
                    f"employees={f.get('num_employees', 0)}, "
                    f"capital=${f.get('capital', 0):.0f}"
                )

        return f"""=== ROUND {round_num} — EMPLOYMENT DECISION ({self.name}) ===

YOUR STATUS:
  Current employer: {self.employer or 'Unemployed'}
  Current wage: ${self.wage:.0f}
  Savings: ${self.savings:.0f}
  Suffering score: {self.suffering_score:.2f} (unmet needs accumulated)
  Productivity modifier: {self.productivity_modifier:.2f}

INCOME TAX: {govpol.get('income_tax_rate', 0):.1%}

AVAILABLE EMPLOYERS:
{''.join(f+chr(10) for f in firm_lines)}
Economy: unemployment {world_state.get('unemployment_rate', 0):.1%}, GDP ${world_state.get('gdp', 0):.0f}

Which firm do you want to apply to? (or "none" to remain unemployed)
Output: {{"chosen_employer": "<firm_name or none>"}}"""

    # ------------------------------------------------------------------
    # Consumer POV — spending allocation
    # ------------------------------------------------------------------

    def decide_consumption(self, world_state: dict, round_num: int) -> dict:
        """LLM call for discretionary spending allocation.

        Called AFTER necessities are automatically purchased, so the agent
        knows what needs were met and what budget remains for discretionary.
        """
        system = self._consumer_system_prompt(world_state)
        prompt = self._consumer_round_prompt(world_state, round_num)

        self.message_history.append({"role": "user", "content": prompt})
        history = self._pruned_history()

        for attempt in range(3):
            try:
                text = self.llm.call(system, history, model_override=self._model())
                self.message_history.append({"role": "assistant", "content": text})
                action = ResponseParser.extract_json(text)
                validated = ResponseParser.validate_consumer_allocation(action)
                self.last_allocation = validated["discretionary_allocation"]
                self.action_log.append({"round": round_num, "phase": "consumer", "action": validated, "reasoning": text})
                return validated
            except Exception as e:
                logger.warning(f"{self.name} consumer parse attempt {attempt+1} failed: {e}")
                if self.message_history and self.message_history[-1]["role"] == "assistant":
                    self.message_history.pop()

        default = {"discretionary_allocation": {"extra_food": 0.1, "extra_shelter": 0.1, "technology": 0.3, "savings": 0.5}}
        self.message_history.append({"role": "assistant", "content": "[FALLBACK]"})
        self.action_log.append({"round": round_num, "phase": "consumer", "action": default, "reasoning": "[FALLBACK]"})
        return default

    def _consumer_system_prompt(self, world_state: dict) -> str:
        sys_label = world_state.get("system_label", "the economy") if world_state else "the economy"
        return f"""You are {self.name}, a {self.profile.name} worker in {sys_label}.

YOUR UTILITY FUNCTION:
  U = 0.40 × consumption_satisfaction
    + 0.30 × financial_security (savings)
    + 0.20 × basic_needs_met
    + 0.10 × job_satisfaction

RIGHT NOW you are making your CONSUMER SPENDING DECISION.
Your necessities (food and shelter) have already been handled automatically.
Now allocate your REMAINING discretionary budget across:
  - extra_food: buy more food beyond the minimum
  - extra_shelter: buy more/better shelter beyond the minimum
  - technology: buy tech services
  - savings: keep the money for future rounds

All fractions must sum to 1.0.

Reason through your situation, then output JSON:
{{"discretionary_allocation": {{"extra_food": <f>, "extra_shelter": <f>, "technology": <f>, "savings": <f>}}}}
"""

    def _consumer_round_prompt(self, world_state: dict, round_num: int) -> str:
        if not world_state:
            return f"Round {round_num}: allocate discretionary budget."
        govpol = world_state.get("government_policy", {})
        transfer_per_worker = (
            govpol.get("spending", {}).get("transfers_to_workers", 0)
            / max(world_state.get("total_workers", 1), 1)
        )
        after_tax = self.income * (1 - govpol.get("income_tax_rate", 0.15)) + transfer_per_worker

        necessity_cost = world_state.get("necessity_cost_this_round", {}).get(self.name, 0.0)
        discretionary_budget = max(0.0, after_tax + self.savings * 0.0 - necessity_cost)
        # Note: savings are preserved; discretionary_budget is from income after necessities

        # Available consumer goods (non-necessity + extra necessity)
        consumer_firms = [
            f for f in world_state.get("firms", [])
            if f.get("is_consumer_good", True) and f.get("industry") != "manufacturing"
        ]
        goods_lines = "\n".join(
            f"  {f['name']} ({f.get('industry','?')}): ${f.get('price',0):.2f}/unit, "
            f"{f.get('inventory',0):.0f} units available"
            for f in consumer_firms
        )

        return f"""=== ROUND {round_num} — CONSUMER DECISION ({self.name}) ===

YOUR FINANCES:
  Gross income this round: ${self.income:.2f}
  After-tax income: ${after_tax:.2f}
  Savings: ${self.savings:.2f}
  Rounds remaining: {self.config.num_rounds - round_num}

NECESSITIES (auto-purchased before this decision):
  Food consumed: {self.food_consumed:.2f} units (need 1.0) {'✓' if self.food_consumed >= 1.0 else '✗ UNMET'}
  Shelter consumed: {self.shelter_consumed:.2f} units (need 0.5) {'✓' if self.shelter_consumed >= 0.5 else '✗ UNMET'}
  Necessity cost: ${necessity_cost:.2f}

DISCRETIONARY BUDGET: ${discretionary_budget:.2f}

AVAILABLE GOODS:
{goods_lines}

Suffering score: {self.suffering_score:.2f} (higher = more accumulated deprivation)

How do you allocate your discretionary budget?
{{"discretionary_allocation": {{"extra_food": <f>, "extra_shelter": <f>, "technology": <f>, "savings": <f>}}}}"""

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def receive_income(self, wage: float):
        self.income = wage

    def update_necessity_consumption(self, food: float, shelter: float, necessity_cost: float):
        self.food_consumed = food
        self.shelter_consumed = shelter

        # Compute suffering from unmet needs
        food_need = 1.0
        shelter_need = 0.5
        food_deficit = max(0.0, food_need - food) / food_need
        shelter_deficit = max(0.0, shelter_need - shelter) / shelter_need
        round_suffering = 0.6 * food_deficit + 0.4 * shelter_deficit
        self.suffering_score += round_suffering

        # Update productivity modifier
        food_mod = 0.70 if food < food_need * 0.5 else (0.85 if food < food_need else 1.0)
        shelter_mod = 0.85 if shelter < shelter_need * 0.5 else (0.93 if shelter < shelter_need else 1.0)
        self.productivity_modifier = food_mod * shelter_mod

    def end_round_reset(self):
        self.income = 0.0
        self.food_consumed = 0.0
        self.shelter_consumed = 0.0


