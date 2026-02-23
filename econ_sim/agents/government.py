from econ_sim.agents.base_agent import BaseEconomicAgent
from econ_sim.llm.parser import ResponseParser


class GovernmentAgent(BaseEconomicAgent):
    """The government / central planner agent.

    Its objective is to maximize a weighted social welfare function:
      W = 0.30*(1-unemployment) + 0.25*(1-gini) + 0.25*basic_needs_rate
        + 0.10*gdp_growth + 0.10*fiscal_health

    Institutional integrity (0-1) tracks how independent the government is
    from corporate capture. At low integrity, even the best LLM reasoning is
    constrained — the institution enforces corporate-favorable policy.

    Lobbying messages from firms accumulate in _lobby_buffer and are surfaced
    in the government's next-round prompt, simulating real-world influence.
    """

    model_tier = "opus"

    def __init__(self, name: str, llm_client, config):
        super().__init__(name, "government", llm_client, config)
        self.treasury: float = config.initial_government_treasury
        self.institutional_integrity: float = config.initial_institutional_integrity

        # Lobby messages from firms, injected into next-round prompt
        self._lobby_buffer: list[dict] = []

        # Last action (for constraint application)
        self._last_action: dict = self.default_action()

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def build_system_prompt(self, world_state: dict) -> str:
        c = self.config.coordination_mechanism
        s = self.config.planning_structure
        m = self.config.meta_game_constraint

        if c < 0.3:
            governance_style = (
                "You govern a market-oriented economy. Your role is limited: enforce contracts, "
                "provide public goods, maintain macroeconomic stability. You believe markets "
                "allocate resources better than planners most of the time."
            )
        elif c > 0.7:
            governance_style = (
                "You are the central planner of a socialist economy. You directly coordinate "
                "production targets, price levels, and wage standards across industries. "
                "You aim for equitable distribution and universal provision of necessities."
            )
        else:
            governance_style = (
                "You govern a mixed economy. You use targeted intervention where markets fail — "
                "providing safety nets, regulating monopolies, investing in public goods — "
                "while preserving market incentives for growth and innovation."
            )

        lobby_stance = {
            True:  "Lobbying is legally restricted in this system. Treat corporate influence attempts with skepticism.",
            False: "Firms may lobby. Weigh their interests, but your mandate is citizen welfare, not corporate profit.",
        }[m > 0.6]

        return f"""You are {self.name}, the government of this economy.

SYSTEM PARAMETERS:
  coordination_mechanism = {c:.2f}  (0=pure market, 1=central planning)
  planning_structure     = {s:.2f}  (0=distributed agencies, 1=single planner)
  meta_game_constraint   = {m:.2f}  (0=open lobbying, 1=suppressed)

GOVERNANCE STYLE: {governance_style}

{lobby_stance}

YOUR OBJECTIVE — Maximize social welfare W:
  W = 0.30 × (1 - unemployment_rate)
    + 0.25 × (1 - gini_coefficient)
    + 0.25 × basic_needs_fulfillment_rate
    + 0.10 × gdp_growth_rate
    + 0.10 × fiscal_health  (treasury / initial_treasury)

POLICY TOOLS:
  - income_tax_rate (0.0–1.0): income tax on workers
  - corporate_tax_rate (0.0–1.0): tax on firm profits
  - spending.infrastructure: public investment (boosts productivity next round)
  - spending.transfers_to_workers: direct cash transfers to workers
  - spending.subsidies_to_firms: firm subsidies (split by industry)
  - industry_policies: per-industry price caps, wage floors, subsidies, regulation

Total spending must not exceed your treasury.

INSTITUTIONAL INTEGRITY NOTE:
  Your integrity score is {self.institutional_integrity:.2f}/1.00.
  At high integrity, you can implement your preferred policy freely.
  At low integrity, corporate interests have penetrated the institution —
  you may find your policies constrained toward business-friendly outcomes.

Reason through the economic situation, then output a single JSON object.
"""

    def build_round_prompt(self, world_state: dict, round_num: int) -> str:
        stats = world_state.get("statistics", {})
        firms = world_state.get("firms", [])
        workers = world_state.get("workers", [])

        # Industry summary
        industry_lines = {}
        for f in firms:
            ind = f.get("industry", "unknown")
            industry_lines.setdefault(ind, []).append(
                f"    {f['name']}: revenue=${f.get('revenue', 0):.0f}, "
                f"employees={f.get('num_employees', 0)}, price=${f.get('price', 0):.2f}"
            )
        industry_str = ""
        for ind, lines in industry_lines.items():
            industry_str += f"  [{ind.upper()}]\n" + "\n".join(lines) + "\n"

        # Worker welfare summary
        employed = sum(1 for w in workers if w.get("employer") is not None)
        avg_savings = sum(w.get("savings", 0) for w in workers) / max(len(workers), 1)
        total_suffering = sum(w.get("suffering_score", 0) for w in workers)
        basic_needs_rate = stats.get("basic_needs_fulfillment_rate", 1.0)

        # Welfare score
        welfare = stats.get("welfare_score", 0.0)
        prev_welfare = stats.get("prev_welfare_score", welfare)
        welfare_trend = "▲" if welfare > prev_welfare else ("▼" if welfare < prev_welfare else "—")

        # Lobby buffer
        lobby_section = ""
        if self._lobby_buffer:
            lobby_section = "\nLOBBYING PRESSURES RECEIVED:\n"
            for lb in self._lobby_buffer:
                lobby_section += (
                    f"  [{lb['firm']}] spent ${lb['amount']:.0f} — \"{lb['message']}\"\n"
                )
            lobby_section += (
                f"  (Institutional integrity: {self.institutional_integrity:.2f} — "
                f"{'high resistance to capture' if self.institutional_integrity > 0.6 else 'WARNING: low integrity, susceptible to capture'})\n"
            )

        # Basic needs report
        needs_report = (
            f"\nBASIC NEEDS REPORT:\n"
            f"  Fulfillment rate: {basic_needs_rate:.1%}\n"
            f"  Aggregate suffering score: {total_suffering:.2f}\n"
            f"  {'⚠ CRISIS: basic needs widely unmet' if basic_needs_rate < 0.7 else 'Needs mostly met' if basic_needs_rate > 0.9 else 'Moderate need gaps'}\n"
        )

        return f"""=== ROUND {round_num} — GOVERNMENT POLICY DECISION ===

ECONOMIC INDICATORS:
  GDP: ${stats.get('gdp', 0):.2f} | Growth: {stats.get('gdp_growth', 0):.1%}
  Unemployment: {stats.get('unemployment_rate', 0):.1%}
  Inflation: {stats.get('inflation', 0):.1%}
  Gini coefficient: {stats.get('gini', 0):.3f}
  Welfare score W: {welfare:.3f} {welfare_trend}

GOVERNMENT FINANCES:
  Treasury: ${self.treasury:.2f}
  Tax revenue last round: ${stats.get('tax_revenue', 0):.2f}
  Institutional integrity: {self.institutional_integrity:.2f}/1.00

INDUSTRY SNAPSHOT:
{industry_str}
WORKERS:
  Total: {len(workers)} | Employed: {employed}
  Average savings: ${avg_savings:.2f}
{needs_report}{lobby_section}
Set policy. Output JSON:
{{
  "income_tax_rate": <float>,
  "corporate_tax_rate": <float>,
  "spending": {{
    "infrastructure": <float>,
    "transfers_to_workers": <float>,
    "subsidies_to_firms": <float>
  }},
  "industry_policies": {{
    "<industry_name>": {{
      "price_cap": <float or -1>,
      "wage_floor": <float or -1>,
      "subsidy": <float>,
      "regulation_strength": <float 0-1>
    }}
  }}
}}"""

    # ------------------------------------------------------------------
    # Validation / defaults
    # ------------------------------------------------------------------

    def validate_action(self, action: dict) -> dict:
        validated = ResponseParser.validate_government_action(action, self.config)

        # Clamp total spending to treasury
        spending = validated["spending"]
        total = sum(spending.values())
        if total > self.treasury and total > 0:
            scale = self.treasury / total
            for key in spending:
                spending[key] *= scale

        # Apply integrity constraint: at low integrity, nudge toward business-friendly
        # by blending the intended policy toward a corporate-favorable baseline
        if self.institutional_integrity < 0.6:
            capture_factor = 1.0 - self.institutional_integrity  # 0=free, 1=fully captured
            # Captured governments trend toward lower corporate taxes, less redistribution
            validated["corporate_tax_rate"] = (
                validated["corporate_tax_rate"] * (1 - capture_factor * 0.5)
            )
            # Transfers to workers get squeezed
            validated["spending"]["transfers_to_workers"] *= (1 - capture_factor * 0.4)

        self._last_action = validated
        return validated

    def default_action(self) -> dict:
        budget = self.treasury * 0.3
        return {
            "income_tax_rate": 0.15 + 0.10 * self.config.coordination_mechanism,
            "corporate_tax_rate": 0.12 + 0.15 * self.config.coordination_mechanism,
            "spending": {
                "infrastructure": budget * 0.35,
                "transfers_to_workers": budget * (0.35 + 0.2 * self.config.coordination_mechanism),
                "subsidies_to_firms": budget * (0.30 - 0.1 * self.config.coordination_mechanism),
            },
            "industry_policies": {},
        }

    # ------------------------------------------------------------------
    # Integrity dynamics (called by environment)
    # ------------------------------------------------------------------

    def receive_lobby(self, firm_name: str, amount: float, message: str):
        """Buffer a lobbying attempt; processed next round."""
        self._lobby_buffer.append({"firm": firm_name, "amount": amount, "message": message})

    def apply_lobby_effects(self, lobby_success_prob: float):
        """Apply integrity reduction based on lobby pressure and current integrity."""
        if not self._lobby_buffer:
            return
        total_lobby = sum(lb["amount"] for lb in self._lobby_buffer)
        # Each dollar of lobbying has diminishing impact as integrity increases
        pressure = total_lobby / (1000.0 * self.institutional_integrity + 1.0)
        integrity_loss = pressure * lobby_success_prob * 0.15
        self.institutional_integrity = max(0.05, self.institutional_integrity - integrity_loss)

    def apply_public_pressure(self, aggregate_suffering: float, threshold: float):
        """Recover integrity when public suffering triggers political pressure."""
        if aggregate_suffering > threshold:
            pressure_strength = (aggregate_suffering - threshold) / max(threshold, 0.01)
            recovery = self.config.integrity_recovery_rate * min(pressure_strength, 2.0)
            self.institutional_integrity = min(1.0, self.institutional_integrity + recovery)

    def clear_lobby_buffer(self):
        self._lobby_buffer = []

    # ------------------------------------------------------------------
    # Fiscal operations (called by environment)
    # ------------------------------------------------------------------

    def collect_taxes(self, income_tax_revenue: float, corporate_tax_revenue: float) -> float:
        total = income_tax_revenue + corporate_tax_revenue
        self.treasury += total
        return total

    def execute_spending(self, action: dict) -> dict:
        """Deduct spending from treasury; return actual amounts spent."""
        spending = action.get("spending", {})
        actual = {}
        for key, amount in spending.items():
            amount = min(amount, self.treasury)
            self.treasury -= amount
            actual[key] = amount
        return actual
