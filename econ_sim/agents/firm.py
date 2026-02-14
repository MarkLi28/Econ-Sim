from econ_sim.agents.base_agent import BaseEconomicAgent
from econ_sim.llm.parser import ResponseParser


class FirmAgent(BaseEconomicAgent):

    def __init__(self, name: str, llm_client, config):
        super().__init__(name, "firm", llm_client, config)
        self.capital = config.initial_firm_capital
        self.employees: list[str] = []
        self.inventory: float = 0.0
        self.revenue: float = 0.0
        self.costs: float = 0.0
        self.price: float = config.base_price_level
        self.wage: float = config.base_wage_level

    def build_system_prompt(self, world_state: dict) -> str:
        if self.config.system.value == "socialist":
            ownership = (
                "You are a state-owned enterprise. The government sets guidelines, "
                "but you still make operational decisions about wages, prices, and hiring. "
                "You should balance profitability with serving the public good."
            )
        else:
            ownership = (
                "You are a privately-owned firm competing for workers and customers. "
                "Your goal is to maximize profit while growing your business."
            )

        return (
            f"You are {self.name}, a firm in this economy.\n"
            f"{ownership}\n\n"
            f"Each round you decide:\n"
            f"- wage_offer: the wage you pay each worker (attracts talent)\n"
            f"- price: the price per unit of goods you sell (affects demand)\n"
            f"- hiring_target: how many workers you want employed\n\n"
            f"Your costs = wages * number of employees.\n"
            f"Your revenue = price * units sold.\n"
            f"Your profit = revenue - costs - corporate taxes.\n\n"
            f"Think like a real CEO: consider competition, consumer demand, labor availability, "
            f"and the tax environment.\n\n"
            f"First reason about your situation, then output:\n"
            f'```json\n'
            f'{{"wage_offer": <float>, "price": <float>, "hiring_target": <int>}}\n'
            f'```'
        )

    def build_round_prompt(self, world_state: dict, round_num: int) -> str:
        stats = world_state["statistics"]
        govt = world_state["government_policy"]

        competitors = [
            f"  - {f['name']}: price=${f['price']:.1f}, wage=${f['wage']:.1f}, employees={f['num_employees']}"
            for f in world_state["firms"]
            if f["name"] != self.name
        ]
        competitor_str = "\n".join(competitors) if competitors else "  (none)"

        subsidy_per_firm = govt["spending"].get("subsidies_to_firms", 0) / max(self.config.num_firms, 1)

        return (
            f"=== ROUND {round_num} / {self.config.num_rounds} — FIRM DECISION ===\n\n"
            f"Your Financials:\n"
            f"  Capital: ${self.capital:.0f}\n"
            f"  Current Employees: {len(self.employees)}\n"
            f"  Last Round Revenue: ${self.revenue:.0f}\n"
            f"  Last Round Costs: ${self.costs:.0f}\n"
            f"  Last Round Profit: ${self.revenue - self.costs:.0f}\n"
            f"  Inventory (unsold goods): {self.inventory:.1f} units\n\n"
            f"Market Conditions:\n"
            f"  Average Price Level: ${stats['avg_price']:.1f}\n"
            f"  Total Consumer Demand Last Round: {stats.get('total_demand', 0):.0f}\n"
            f"  Unemployment Rate: {stats['unemployment_rate']*100:.1f}%\n\n"
            f"Government Policy:\n"
            f"  Corporate Tax Rate: {govt['corporate_tax_rate']*100:.1f}%\n"
            f"  Min Wage: {'$' + str(self.config.min_wage) if self.config.min_wage_enabled else 'None'}\n"
            f"  Price Cap: {'$' + str(self.config.max_price) if self.config.price_controls_enabled else 'None'}\n"
            f"  Your Subsidy Share: ${subsidy_per_firm:.0f}\n\n"
            f"Competitors:\n{competitor_str}\n\n"
            f"Decide your strategy for this round."
        )

    def validate_action(self, action: dict) -> dict:
        return ResponseParser.validate_firm_action(action, self.config)

    def default_action(self) -> dict:
        return {
            "wage_offer": self.config.base_wage_level,
            "price": self.config.base_price_level,
            "hiring_target": 2,
        }
