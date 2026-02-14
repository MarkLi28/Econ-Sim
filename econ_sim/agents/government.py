from econ_sim.agents.base_agent import BaseEconomicAgent
from econ_sim.llm.parser import ResponseParser


SYSTEM_DESCRIPTIONS = {
    "capitalist": (
        "You govern a free-market capitalist economy. Your role is limited: "
        "maintain order, enforce contracts, and provide minimal public goods. "
        "You believe in low taxes, light regulation, and letting markets allocate resources."
    ),
    "socialist": (
        "You govern a socialist economy. Your role is expansive: ensure equality, "
        "provide universal services, and guide key industries. You believe in high "
        "redistribution, worker protections, and active state management of the economy."
    ),
    "mixed": (
        "You govern a mixed economy. You balance market freedom with social safety nets. "
        "You believe in pragmatic policy — using targeted intervention where markets fail "
        "while preserving incentives for growth."
    ),
}


class GovernmentAgent(BaseEconomicAgent):

    def __init__(self, name: str, llm_client, config):
        super().__init__(name, "government", llm_client, config)
        self.treasury = config.initial_government_treasury

    def build_system_prompt(self, world_state: dict) -> str:
        desc = SYSTEM_DESCRIPTIONS[self.config.system.value]
        lo_i, hi_i = self.config.income_tax_rate_range
        lo_c, hi_c = self.config.corporate_tax_rate_range

        return (
            f"You are {self.name}, the government of this economy.\n"
            f"{desc}\n\n"
            f"Your policy tools:\n"
            f"- Set income tax rate (allowed range: {lo_i*100:.0f}% – {hi_i*100:.0f}%)\n"
            f"- Set corporate tax rate (allowed range: {lo_c*100:.0f}% – {hi_c*100:.0f}%)\n"
            f"- Allocate your budget across: infrastructure, transfers_to_workers, subsidies_to_firms\n"
            f"  (Total spending must not exceed your treasury balance.)\n\n"
            f"Your goal: maximize overall human flourishing — consider GDP growth, low unemployment, "
            f"manageable inequality, and economic stability.\n\n"
            f"First reason about the current economic conditions, then output your decision as a JSON block:\n"
            f'```json\n'
            f'{{"income_tax_rate": <float>, "corporate_tax_rate": <float>, '
            f'"spending": {{"infrastructure": <float>, "transfers_to_workers": <float>, '
            f'"subsidies_to_firms": <float>}}}}\n'
            f'```'
        )

    def build_round_prompt(self, world_state: dict, round_num: int) -> str:
        stats = world_state["statistics"]
        firms = world_state["firms"]
        workers = world_state["workers"]

        firm_lines = "\n".join(
            f"  - {f['name']}: revenue=${f['revenue']:.0f}, employees={f['num_employees']}, price=${f['price']:.1f}, wage=${f['wage']:.1f}"
            for f in firms
        )
        employed = sum(1 for w in workers if w["employer"] is not None)
        avg_savings = sum(w["savings"] for w in workers) / max(len(workers), 1)

        return (
            f"=== ROUND {round_num} / {self.config.num_rounds} — GOVERNMENT POLICY DECISION ===\n\n"
            f"Economic Indicators:\n"
            f"  GDP: ${stats['gdp']:.0f}\n"
            f"  Unemployment Rate: {stats['unemployment_rate']*100:.1f}%\n"
            f"  Average Wage: ${stats['avg_wage']:.0f}\n"
            f"  Average Price Level: ${stats['avg_price']:.1f}\n"
            f"  Inflation: {stats['inflation']*100:.1f}%\n"
            f"  Gini Coefficient: {stats['gini']:.3f}\n\n"
            f"Government Finances:\n"
            f"  Treasury: ${self.treasury:.0f}\n"
            f"  Tax Revenue Last Round: ${stats.get('tax_revenue', 0):.0f}\n\n"
            f"Firms:\n{firm_lines}\n\n"
            f"Workers:\n"
            f"  Total: {len(workers)}, Employed: {employed}\n"
            f"  Average Savings: ${avg_savings:.0f}\n\n"
            f"Set your policy for this round."
        )

    def validate_action(self, action: dict) -> dict:
        validated = ResponseParser.validate_government_action(action, self.config)
        # Clamp total spending to treasury
        spending = validated["spending"]
        total = sum(spending.values())
        if total > self.treasury and total > 0:
            scale = self.treasury / total
            for key in spending:
                spending[key] *= scale
        return validated

    def default_action(self) -> dict:
        lo_i, hi_i = self.config.income_tax_rate_range
        lo_c, hi_c = self.config.corporate_tax_rate_range
        mid_i = (lo_i + hi_i) / 2
        mid_c = (lo_c + hi_c) / 2
        budget = self.treasury * 0.3
        return {
            "income_tax_rate": mid_i,
            "corporate_tax_rate": mid_c,
            "spending": {
                "infrastructure": budget * 0.4,
                "transfers_to_workers": budget * 0.4,
                "subsidies_to_firms": budget * 0.2,
            },
        }
