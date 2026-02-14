import logging

logger = logging.getLogger(__name__)


class LaborMarket:
    """Deterministic labor market clearing.

    Workers choose a preferred employer. Firms hire up to their target,
    ranked by skill level (highest first). Rejected workers are unemployed.
    """

    def __init__(self, config):
        self.config = config

    def clear(self, firm_actions: dict, worker_actions: dict, firms: list, workers: list) -> dict:
        results = {"assignments": {}, "unemployed": [], "wages_paid": {}}

        # Reset all employment state
        for firm in firms:
            firm.employees = []
        for worker in workers:
            worker.employer = None
            worker.wage = 0.0
            worker.income = 0.0

        # Build applicant pools
        firm_map = {f.name: f for f in firms}
        applicant_pools: dict[str, list] = {f.name: [] for f in firms}

        for worker in workers:
            choice = worker_actions[worker.name]["chosen_employer"]
            if choice and choice in applicant_pools:
                applicant_pools[choice].append(worker)
            else:
                results["unemployed"].append(worker.name)

        # Resolve each firm's hiring
        for firm in firms:
            target = firm_actions[firm.name]["hiring_target"]
            wage = firm_actions[firm.name]["wage_offer"]
            applicants = sorted(
                applicant_pools[firm.name],
                key=lambda w: w.skill_level,
                reverse=True,
            )
            hired = applicants[:target]
            rejected = applicants[target:]

            for w in hired:
                w.employer = firm.name
                w.wage = wage
                w.income = wage
                firm.employees.append(w.name)
                results["assignments"][w.name] = firm.name

            for w in rejected:
                results["unemployed"].append(w.name)

            firm.wage = wage
            results["wages_paid"][firm.name] = wage * len(hired)

        return results


class GoodsMarket:
    """Deterministic goods market clearing.

    Workers spend a fraction of their after-tax income on goods.
    They buy from the cheapest firm first (rational consumer).
    """

    def __init__(self, config):
        self.config = config

    def clear(
        self,
        firms: list,
        workers: list,
        worker_actions: dict,
        income_tax_rate: float,
        transfer_per_worker: float,
    ) -> dict:
        results = {
            "firm_sales": {f.name: 0.0 for f in firms},
            "firm_revenue": {f.name: 0.0 for f in firms},
            "worker_spending": {},
            "unmet_demand": 0.0,
        }

        # Sort firms by price (cheapest first)
        sorted_firms = sorted(firms, key=lambda f: f.price)

        for worker in workers:
            after_tax_income = worker.income * (1 - income_tax_rate) + transfer_per_worker
            budget = after_tax_income * worker_actions[worker.name]["spending_fraction"]
            total_spent = 0.0

            for firm in sorted_firms:
                if budget <= 0 or firm.inventory <= 0:
                    continue
                if firm.price <= 0:
                    continue
                affordable_units = budget / firm.price
                units_bought = min(affordable_units, firm.inventory)
                cost = units_bought * firm.price
                firm.inventory -= units_bought
                budget -= cost
                total_spent += cost
                results["firm_sales"][firm.name] += units_bought
                results["firm_revenue"][firm.name] += cost

            results["worker_spending"][worker.name] = total_spent
            if budget > 0:
                results["unmet_demand"] += budget

            # Update worker savings: income minus taxes minus spending plus transfer
            worker.savings += after_tax_income - total_spent

        return results
