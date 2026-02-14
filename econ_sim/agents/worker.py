import logging

from econ_sim.agents.base_agent import BaseEconomicAgent
from econ_sim.llm.parser import ResponseParser

logger = logging.getLogger(__name__)


class WorkerAgent(BaseEconomicAgent):
    """A person in the economy with two distinct decision-making perspectives:

    1. WORKER POV — "Which job should I take?" (before labor market clears)
    2. CONSUMER POV — "How should I spend my money?" (after production, with real income known)
    """

    def __init__(self, name: str, llm_client, config, skill_level: float = 1.0):
        super().__init__(name, "worker", llm_client, config)
        self.savings = config.initial_worker_savings
        self.skill_level = skill_level
        self.employer: str | None = None
        self.wage: float = 0.0
        self.income: float = 0.0

    # ── Worker POV (employment decision) ──────────────────────────

    def build_system_prompt(self, world_state: dict) -> str:
        return self._worker_system_prompt(world_state)

    def build_round_prompt(self, world_state: dict, round_num: int) -> str:
        return self._worker_round_prompt(world_state, round_num)

    def validate_action(self, action: dict) -> dict:
        return self._validate_employment(action)

    def default_action(self) -> dict:
        return {"chosen_employer": None}

    # ── Consumer POV (spending decision) ──────────────────────────

    def decide_consumption(self, world_state: dict, round_num: int, max_attempts: int = 3) -> dict:
        """Separate LLM call for the consumer perspective.

        Called AFTER labor market clears and production runs, so this agent
        knows its actual wage, what goods are available, and at what prices.
        """
        system_prompt = self._consumer_system_prompt(world_state)
        round_prompt = self._consumer_round_prompt(world_state, round_num)

        self.message_history.append({"role": "user", "content": round_prompt})

        for attempt in range(max_attempts):
            try:
                response_text = self.llm.call(system_prompt, self.message_history)
                self.message_history.append({"role": "assistant", "content": response_text})
                action = ResponseParser.extract_json(response_text)
                validated = self._validate_consumption(action)
                self.action_log.append({
                    "round": round_num,
                    "phase": "consumer",
                    "action": validated,
                    "reasoning": response_text,
                })
                return validated
            except Exception as e:
                logger.warning(f"{self.name} consumer parse attempt {attempt + 1} failed: {e}")
                if self.message_history and self.message_history[-1]["role"] == "assistant":
                    self.message_history.pop()
                if attempt == max_attempts - 1:
                    logger.error(f"{self.name} falling back to default consumption")
                    default = {"spending_fraction": 0.5}
                    self.action_log.append({
                        "round": round_num,
                        "phase": "consumer",
                        "action": default,
                        "reasoning": "[FALLBACK — parse failed]",
                    })
                    return default

    # ── Worker POV prompts ────────────────────────────────────────

    def _worker_system_prompt(self, world_state: dict) -> str:
        return (
            f"You are {self.name}, a person in this economy. Right now you are making "
            f"your EMPLOYMENT decision — thinking purely as a WORKER.\n\n"
            f"Your skill level: {self.skill_level:.1f} (higher = more productive, more desirable to employers).\n\n"
            f"Consider:\n"
            f"- Which firms are hiring and what wages they offer\n"
            f"- Job stability — does the firm have revenue to sustain your job?\n"
            f"- Competition — will many other workers apply to the same firm?\n"
            f"- Your financial cushion (savings) if you risk unemployment\n\n"
            f"You are NOT deciding spending right now. Focus only on employment.\n\n"
            f"First reason about your job options, then output:\n"
            f'```json\n'
            f'{{"chosen_employer": "<firm_name or none>"}}\n'
            f'```'
        )

    def _worker_round_prompt(self, world_state: dict, round_num: int) -> str:
        govt = world_state["government_policy"]
        firms = world_state["firms"]

        job_lines = "\n".join(
            f"  - {f['name']}: wage=${f['wage']:.1f}, employees={f['num_employees']}, "
            f"revenue last round=${f['revenue']:.0f}, capital=${f.get('capital', 'N/A')}"
            for f in firms
        )

        return (
            f"=== ROUND {round_num} / {self.config.num_rounds} — EMPLOYMENT DECISION (Worker POV) ===\n\n"
            f"Your Status:\n"
            f"  Current Employer: {self.employer or 'Unemployed'}\n"
            f"  Current Wage: ${self.wage:.0f}\n"
            f"  Savings: ${self.savings:.0f}\n"
            f"  Skill Level: {self.skill_level:.1f}\n\n"
            f"Income Tax Rate: {govt['income_tax_rate']*100:.1f}%\n\n"
            f"Job Openings:\n{job_lines}\n\n"
            f"Which firm do you want to work for this round?"
        )

    def _validate_employment(self, action: dict) -> dict:
        employer = action.get("chosen_employer", "none")
        if isinstance(employer, str):
            employer = employer.strip()
            if employer.lower() == "none":
                employer = None
        action["chosen_employer"] = employer
        return action

    # ── Consumer POV prompts ──────────────────────────────────────

    def _consumer_system_prompt(self, world_state: dict) -> str:
        return (
            f"You are {self.name}, a person in this economy. Right now you are making "
            f"your SPENDING decision — thinking purely as a CONSUMER.\n\n"
            f"You already have your job for this round. Now you need to decide how much "
            f"of your after-tax income to spend on goods versus save.\n\n"
            f"Consider:\n"
            f"- What goods cost right now (are prices reasonable or inflated?)\n"
            f"- How much you have in savings (do you need a safety net?)\n"
            f"- How many rounds are left (less reason to save near the end)\n"
            f"- Whether you expect prices to rise or fall\n"
            f"- Your overall financial security\n\n"
            f"You are NOT deciding employment right now. Focus only on spending.\n\n"
            f"First reason about your financial situation, then output:\n"
            f'```json\n'
            f'{{"spending_fraction": <float 0.0 to 1.0>}}\n'
            f'```'
        )

    def _consumer_round_prompt(self, world_state: dict, round_num: int) -> str:
        govt = world_state["government_policy"]
        firms = world_state["firms"]

        transfer_per_worker = govt["spending"].get("transfers_to_workers", 0) / max(
            world_state["statistics"]["num_workers"], 1
        )

        after_tax_income = self.income * (1 - govt["income_tax_rate"]) + transfer_per_worker

        goods_lines = "\n".join(
            f"  - {f['name']}: price=${f['price']:.1f}, inventory={f['inventory']:.0f} units available"
            for f in firms
        )

        rounds_left = self.config.num_rounds - round_num

        return (
            f"=== ROUND {round_num} / {self.config.num_rounds} — SPENDING DECISION (Consumer POV) ===\n\n"
            f"Your Financial Situation:\n"
            f"  Employer: {self.employer or 'Unemployed'}\n"
            f"  Gross Wage: ${self.wage:.0f}\n"
            f"  After-Tax Income This Round: ${after_tax_income:.0f}\n"
            f"  Savings: ${self.savings:.0f}\n"
            f"  Rounds Remaining After This: {rounds_left}\n\n"
            f"Goods Available:\n{goods_lines}\n\n"
            f"What fraction of your after-tax income will you spend on goods? "
            f"(The rest goes into savings.)"
        )

    def _validate_consumption(self, action: dict) -> dict:
        action["spending_fraction"] = max(0.0, min(1.0, float(action.get("spending_fraction", 0.5))))
        return action
