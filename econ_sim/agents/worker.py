from econ_sim.agents.base_agent import BaseEconomicAgent
from econ_sim.llm.parser import ResponseParser


class WorkerAgent(BaseEconomicAgent):

    def __init__(self, name: str, llm_client, config, skill_level: float = 1.0):
        super().__init__(name, "worker", llm_client, config)
        self.savings = config.initial_worker_savings
        self.skill_level = skill_level
        self.employer: str | None = None
        self.wage: float = 0.0
        self.income: float = 0.0

    def build_system_prompt(self, world_state: dict) -> str:
        return (
            f"You are {self.name}, a worker and consumer in this economy.\n"
            f"Your skill level: {self.skill_level:.1f} (higher = more productive, more desirable to employers).\n\n"
            f"Each round you make two decisions:\n"
            f"1. EMPLOYMENT: Which firm to work for (or choose \"none\" to be unemployed this round)\n"
            f"2. SPENDING: What fraction of your income to spend on goods (rest goes to savings)\n\n"
            f"Think like a real person: consider which firms pay the best wages, "
            f"what goods cost, how much you have saved, and your job security.\n\n"
            f"First reason about your situation, then output:\n"
            f'```json\n'
            f'{{"chosen_employer": "<firm_name or none>", "spending_fraction": <float 0-1>}}\n'
            f'```'
        )

    def build_round_prompt(self, world_state: dict, round_num: int) -> str:
        govt = world_state["government_policy"]
        firms = world_state["firms"]

        job_lines = "\n".join(
            f"  - {f['name']}: wage=${f['wage']:.1f}, price=${f['price']:.1f}, employees={f['num_employees']}"
            for f in firms
        )

        transfer_per_worker = govt["spending"].get("transfers_to_workers", 0) / max(
            world_state["statistics"]["num_workers"], 1
        )

        return (
            f"=== ROUND {round_num} / {self.config.num_rounds} — WORKER DECISION ===\n\n"
            f"Your Status:\n"
            f"  Current Employer: {self.employer or 'Unemployed'}\n"
            f"  Current Wage: ${self.wage:.0f}\n"
            f"  Savings: ${self.savings:.0f}\n"
            f"  Skill Level: {self.skill_level:.1f}\n\n"
            f"Tax & Transfers:\n"
            f"  Income Tax Rate: {govt['income_tax_rate']*100:.1f}%\n"
            f"  Government Transfer You Receive: ${transfer_per_worker:.0f}\n\n"
            f"Available Jobs & Goods:\n{job_lines}\n\n"
            f"Choose which firm to work for and what fraction of your after-tax income "
            f"to spend on goods (the rest is saved)."
        )

    def validate_action(self, action: dict) -> dict:
        # We need firm names for validation — but we stored config, not firms list.
        # The environment will re-validate with actual firm names after calling decide().
        # Here we just do basic field validation.
        action["spending_fraction"] = max(0.0, min(1.0, float(action.get("spending_fraction", 0.5))))
        employer = action.get("chosen_employer", "none")
        if isinstance(employer, str):
            employer = employer.strip()
        if employer.lower() == "none":
            employer = None
        action["chosen_employer"] = employer
        return action

    def default_action(self) -> dict:
        return {
            "chosen_employer": None,
            "spending_fraction": 0.5,
        }
