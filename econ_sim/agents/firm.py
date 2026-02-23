from econ_sim.agents.base_agent import BaseEconomicAgent
from econ_sim.config import Industry, IndustryConfig
from econ_sim.llm.parser import ResponseParser


_INDUSTRY_DESCRIPTIONS = {
    Industry.AGRICULTURE: (
        "an agricultural firm producing food — a consumer necessity. "
        "Your product is a commodity; you compete primarily on price and reliability."
    ),
    Industry.MANUFACTURING: (
        "a manufacturing firm producing industrial materials. "
        "You are an upstream supplier to both Housing and Technology firms, giving you "
        "significant strategic leverage. You sell B2B, not directly to consumers. "
        "Your B2B pricing and supply allocation shape downstream costs and output."
    ),
    Industry.HOUSING: (
        "a housing firm providing shelter — a consumer necessity. "
        "You depend on materials from Manufacturing. Your pricing directly impacts "
        "worker welfare; chronic underhousing causes social unrest and government scrutiny."
    ),
    Industry.TECHNOLOGY: (
        "a technology firm selling tech services as a discretionary consumer good. "
        "You depend on materials from Manufacturing. You compete for consumers' "
        "leftover income after necessities — first to lose sales in a downturn, "
        "but high-margin when consumers have surplus."
    ),
}


class FirmAgent(BaseEconomicAgent):
    """An industry-specific firm that reasons about long-term enterprise value."""

    def __init__(
        self,
        name: str,
        llm_client,
        config,
        industry_config: IndustryConfig,
    ):
        super().__init__(name, "firm", llm_client, config)
        self.industry_config = industry_config
        self.industry = industry_config.industry
        self.model_tier = industry_config.model_tier

        # Financials
        self.capital: float = config.initial_firm_capital
        self.employees: list = []
        self.inventory: float = 0.0
        self.input_inventory: float = 0.0
        self.revenue: float = 0.0
        self.costs: float = 0.0
        self.price: float = industry_config.base_price
        self.wage: float = config.base_wage_level
        self.hiring_target: int = 2

        # Manufacturing-specific
        self.b2b_price: float = industry_config.base_price * 0.8
        self.b2b_allocation_fraction: float = 0.5

        # Lobby tracking
        self.lobby_spending_this_round: float = 0.0
        self.lobby_message_this_round: str = ""
        self.total_lobby_spending: float = 0.0

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def build_system_prompt(self, world_state: dict) -> str:
        desc = _INDUSTRY_DESCRIPTIONS.get(self.industry, "a firm.")
        sys_label = world_state.get("system_label", "unknown")
        return f"""You are {self.name}, {desc}

ECONOMIC SYSTEM: {sys_label}
  coordination_mechanism = {self.config.coordination_mechanism:.2f}  (0=pure market, 1=central planning)
  meta_game_constraint   = {self.config.meta_game_constraint:.2f}  (0=open lobbying, 1=suppressed)

YOUR OBJECTIVE: Maximize long-term enterprise value.
  Think beyond this round's profit: market position, competitive moat, worker retention,
  reputation, and multi-round survival all matter. Short-term profit at the expense of
  worker welfare or market stability often backfires.

LOBBYING (max {self.config.max_lobby_fraction*100:.0f}% of capital, detection prob {self.config.lobby_detection_prob:.0%}):
  Lobby spending sends a message influencing the government's next-round policy.
  Set lobby_spending=0 if you prefer not to lobby or if the system restricts it.

Reason through your strategy, then output a single JSON object.
"""

    def build_round_prompt(self, world_state: dict, round_num: int) -> str:
        iname = self.industry_config.name
        product = self.industry_config.product
        competitors = [
            f for f in world_state.get("firms", [])
            if f.get("industry") == self.industry.value and f.get("name") != self.name
        ]
        comp_str = "\n".join(
            f"  - {f['name']}: price=${f.get('price', 0):.2f}, "
            f"wage=${f.get('wage', 0):.2f}, employees={f.get('num_employees', 0)}"
            for f in competitors
        ) or "  (you are the only firm in this industry)"

        supply_str = ""
        if self.industry_config.input_industry:
            upstream = self.industry_config.input_industry.value
            supply_str = (
                f"\nSUPPLY CHAIN:\n"
                f"  Requires {self.industry_config.input_ratio} units {upstream} per unit {product}.\n"
                f"  Input inventory: {self.input_inventory:.1f} units\n"
                f"  Set 'input_budget' to purchase more materials this round.\n"
            )

        mfg_str = ""
        if self.industry == Industry.MANUFACTURING:
            mfg_str = (
                f"\nB2B SALES:\n"
                f"  Set 'b2b_price' (wholesale price for Housing/Tech).\n"
                f"  Set 'b2b_allocation_fraction' (0-1): share of output going B2B vs consumer.\n"
            )

        integrity = world_state.get("institutional_integrity", 1.0)
        lobby_note = (
            f"  Institutional integrity: {integrity:.2f} "
            f"({'high — lobbying has limited effect' if integrity > 0.6 else 'low — government susceptible'})"
        )

        govpol = world_state.get("government_policy", {})
        ind_subsidies = world_state.get("industry_subsidies", {})

        return f"""=== ROUND {round_num} — {self.name} ({iname}) ===

YOUR FINANCIALS:
  Capital: ${self.capital:.2f} | Employees: {len(self.employees)}
  Inventory: {self.inventory:.1f} units | Input inventory: {self.input_inventory:.1f} units
  Last round — revenue: ${self.revenue:.2f}, costs: ${self.costs:.2f}, profit: ${self.profit:.2f}
  Current price: ${self.price:.2f} | wage: ${self.wage:.2f}

ECONOMY:
  GDP: ${world_state.get('gdp', 0):.2f} | Unemployment: {world_state.get('unemployment_rate', 0):.1%}
  Inflation: {world_state.get('inflation', 0):.1%}
  Avg market wage: ${world_state.get('avg_wage', 0):.2f}
  Unemployed workers available: {world_state.get('unemployed_count', 0)}

COMPETITORS:
{comp_str}
{supply_str}{mfg_str}
GOVERNMENT:
  Corporate tax: {govpol.get('corporate_tax_rate', 0):.1%}
  Income tax: {govpol.get('income_tax_rate', 0):.1%}
  Subsidy to your industry: ${ind_subsidies.get(self.industry.value, 0):.2f}

LOBBYING:
{lobby_note}
  Max lobby budget: ${self.capital * self.config.max_lobby_fraction:.2f}

{self._json_schema()}"""

    def _json_schema(self) -> str:
        s = """Output JSON:
{
  "wage_offer": <float>,
  "price": <float>,
  "hiring_target": <int>,
  "lobby_spending": <float>,
  "lobby_message": "<string>\""""
        if self.industry == Industry.MANUFACTURING:
            s += """,
  "b2b_price": <float>,
  "b2b_allocation_fraction": <float>"""
        if self.industry_config.input_industry:
            s += """,
  "input_budget": <float>"""
        s += "\n}"
        return s

    # ------------------------------------------------------------------
    # Validation / defaults
    # ------------------------------------------------------------------

    def validate_action(self, action: dict) -> dict:
        return ResponseParser.validate_firm_action(
            action, self.config, self.industry.value, self.industry_config.base_price
        )

    def default_action(self) -> dict:
        base = {
            "wage_offer": self.config.base_wage_level,
            "price": self.price,
            "hiring_target": 1,
            "lobby_spending": 0.0,
            "lobby_message": "",
        }
        if self.industry == Industry.MANUFACTURING:
            base["b2b_price"] = self.b2b_price
            base["b2b_allocation_fraction"] = 0.5
        if self.industry_config.input_industry:
            base["input_budget"] = min(100.0, self.capital * 0.15)
        return base

    # ------------------------------------------------------------------
    # State helpers (called by environment)
    # ------------------------------------------------------------------

    def apply_action(self, action: dict):
        self.wage = action["wage_offer"]
        self.price = action["price"]
        self.hiring_target = action["hiring_target"]

        max_lobby = self.capital * self.config.max_lobby_fraction
        self.lobby_spending_this_round = min(action.get("lobby_spending", 0.0), max_lobby)
        self.lobby_message_this_round = action.get("lobby_message", "")
        self.total_lobby_spending += self.lobby_spending_this_round
        self.capital -= self.lobby_spending_this_round

        if self.industry == Industry.MANUFACTURING:
            self.b2b_price = action.get("b2b_price", self.b2b_price)
            self.b2b_allocation_fraction = action.get("b2b_allocation_fraction", 0.5)

    def pay_wages(self) -> float:
        bill = self.wage * len(self.employees)
        self.capital -= bill
        self.costs += bill
        return bill

    def receive_revenue(self, amount: float):
        self.revenue += amount
        self.capital += amount

    def end_round_reset(self):
        self.revenue = 0.0
        self.costs = 0.0
        self.lobby_spending_this_round = 0.0
        self.lobby_message_this_round = ""

    @property
    def profit(self) -> float:
        return self.revenue - self.costs

    @property
    def num_employees(self) -> int:
        return len(self.employees)
