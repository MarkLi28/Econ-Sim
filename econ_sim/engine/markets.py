import logging
from typing import Dict, List

from econ_sim.config import Industry, NecessityRequirement

logger = logging.getLogger(__name__)


class LaborMarket:
    """Deterministic labor market clearing.

    Workers apply to a preferred firm. Firms hire up to their hiring_target,
    ranked by skill level (highest skill hired first). Rejected workers are unemployed.
    The industry of each hired worker is recorded for downstream tracking.
    """

    def __init__(self, config):
        self.config = config

    def clear(
        self,
        firm_actions: dict,
        worker_actions: dict,
        firms: list,
        workers: list,
    ) -> dict:
        results = {"assignments": {}, "unemployed": [], "wages_paid": {}}

        # Reset all employment state
        for firm in firms:
            firm.employees = []
        for worker in workers:
            worker.employer = None
            worker.industry = None
            worker.wage = 0.0
            worker.income = 0.0

        firm_map = {f.name: f for f in firms}
        applicant_pools: Dict[str, list] = {f.name: [] for f in firms}

        for worker in workers:
            choice = worker_actions[worker.name].get("chosen_employer")
            if choice and choice in applicant_pools:
                applicant_pools[choice].append(worker)
            else:
                results["unemployed"].append(worker.name)

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
                w.industry = firm.industry
                w.wage = wage
                w.income = wage
                firm.employees.append(w.name)
                results["assignments"][w.name] = firm.name

            for w in rejected:
                results["unemployed"].append(w.name)

            firm.wage = wage
            results["wages_paid"][firm.name] = wage * len(hired)

        return results


class InputMarket:
    """Clears the B2B materials market between Manufacturing and downstream firms.

    Manufacturing firms expose their inventory (less a consumer reserve) at b2b_price.
    Housing and Technology firms spend their input_budget to acquire materials.
    """

    def __init__(self, config):
        self.config = config

    def clear(
        self,
        all_firms: list,
    ) -> dict:
        results = {
            "mfg_b2b_revenue": {},
            "inputs_acquired": {},
            "input_costs": {},
        }

        mfg_firms = [f for f in all_firms if f.industry == Industry.MANUFACTURING]
        downstream_firms = [
            f for f in all_firms
            if self.config.industry_configs[f.industry].input_industry == Industry.MANUFACTURING
        ]

        if not mfg_firms or not downstream_firms:
            return results

        for firm in all_firms:
            results["inputs_acquired"][firm.name] = 0.0
            results["input_costs"][firm.name] = 0.0
        for mfg in mfg_firms:
            results["mfg_b2b_revenue"][mfg.name] = 0.0

        # Manufacturing firms set aside B2B stock (b2b_allocation_fraction of inventory)
        mfg_supply: Dict[str, float] = {}
        for mfg in mfg_firms:
            b2b_stock = mfg.inventory * mfg.b2b_allocation_fraction
            mfg_supply[mfg.name] = b2b_stock

        # Downstream firms buy from cheapest manufacturing firm first
        sorted_mfg = sorted(mfg_firms, key=lambda f: f.b2b_price)

        for buyer in downstream_firms:
            budget = getattr(buyer, "_input_budget_this_round", 0.0)
            if budget <= 0:
                continue

            for mfg in sorted_mfg:
                available = mfg_supply.get(mfg.name, 0.0)
                if available <= 0 or budget <= 0 or mfg.b2b_price <= 0:
                    continue

                units_can_buy = budget / mfg.b2b_price
                units_bought = min(units_can_buy, available)
                cost = units_bought * mfg.b2b_price

                mfg.inventory -= units_bought
                mfg_supply[mfg.name] -= units_bought
                mfg.receive_revenue(cost)
                results["mfg_b2b_revenue"][mfg.name] += cost

                buyer.input_inventory += units_bought
                buyer.capital -= cost
                buyer.costs += cost
                budget -= cost

                results["inputs_acquired"][buyer.name] += units_bought
                results["input_costs"][buyer.name] += cost

        return results


class GoodsMarket:
    """Necessity-first goods market clearing.

    Phase A (deterministic): Workers auto-buy necessities (food, shelter) up to
    minimums with their after-tax income. Unmet necessities accumulate suffering.

    Phase B (after LLM consumer decision): Workers allocate remaining discretionary
    budget according to their LLM-determined allocation fractions.
    """

    def __init__(self, config):
        self.config = config
        self._necessity_map = {
            req.industry: req for req in config.necessity_requirements
        }

    # ------------------------------------------------------------------
    # Phase A: Necessity clearing (before consumer LLM call)
    # ------------------------------------------------------------------

    def clear_necessities(
        self,
        all_firms: list,
        workers: list,
        income_tax_rate: float,
        transfer_per_worker: float,
    ) -> dict:
        """Auto-purchase necessities before any LLM consumer decision.

        Returns per-worker necessity costs and consumption amounts.
        """
        results = {
            "necessity_cost": {},      # {worker_name: total $}
            "food_consumed": {},
            "shelter_consumed": {},
            "firm_revenue": {f.name: 0.0 for f in all_firms},
            "firm_sales": {f.name: 0.0 for f in all_firms},
        }

        # Consumer goods firms, grouped by industry
        consumer_firms_by_industry = {}
        for f in all_firms:
            ind_cfg = self.config.industry_configs[f.industry]
            if ind_cfg.is_consumer_good and ind_cfg.is_necessity:
                consumer_firms_by_industry.setdefault(f.industry, []).append(f)

        # Sort each necessity industry by price (cheapest first)
        for ind in consumer_firms_by_industry:
            consumer_firms_by_industry[ind].sort(key=lambda f: f.price)

        for worker in workers:
            after_tax = worker.income * (1 - income_tax_rate) + transfer_per_worker
            budget = after_tax  # full income available for necessities first
            necessity_cost = 0.0
            food_consumed = 0.0
            shelter_consumed = 0.0

            for req in self.config.necessity_requirements:
                firms_for_need = consumer_firms_by_industry.get(req.industry, [])
                units_needed = req.min_units_per_round
                units_bought = 0.0

                for firm in firms_for_need:
                    if units_bought >= units_needed or budget <= 0 or firm.inventory <= 0:
                        break
                    still_need = units_needed - units_bought
                    affordable = budget / firm.price if firm.price > 0 else float("inf")
                    units = min(still_need, affordable, firm.inventory)
                    cost = units * firm.price

                    firm.inventory -= units
                    firm.receive_revenue(cost)
                    budget -= cost
                    necessity_cost += cost
                    units_bought += units

                    results["firm_revenue"][firm.name] += cost
                    results["firm_sales"][firm.name] += units

                if req.industry == Industry.AGRICULTURE:
                    food_consumed = units_bought
                elif req.industry == Industry.HOUSING:
                    shelter_consumed = units_bought

            # Track remaining budget for discretionary phase
            worker._post_necessity_budget = budget
            worker._post_necessity_income = after_tax
            worker.update_necessity_consumption(food_consumed, shelter_consumed, necessity_cost)

            results["necessity_cost"][worker.name] = necessity_cost
            results["food_consumed"][worker.name] = food_consumed
            results["shelter_consumed"][worker.name] = shelter_consumed

        return results

    # ------------------------------------------------------------------
    # Phase B: Discretionary clearing (after consumer LLM call)
    # ------------------------------------------------------------------

    def clear_discretionary(
        self,
        all_firms: list,
        workers: list,
        consumer_actions: dict,
        necessity_results: dict,
    ) -> dict:
        """Execute discretionary spending based on LLM consumer decisions."""
        results = {
            "firm_revenue": {f.name: 0.0 for f in all_firms},
            "firm_sales": {f.name: 0.0 for f in all_firms},
            "worker_spending": {},
            "worker_savings_change": {},
        }

        # Index consumer firms by industry
        extra_food_firms = sorted(
            [f for f in all_firms if f.industry == Industry.AGRICULTURE and self.config.industry_configs[f.industry].is_consumer_good],
            key=lambda f: f.price
        )
        extra_shelter_firms = sorted(
            [f for f in all_firms if f.industry == Industry.HOUSING and self.config.industry_configs[f.industry].is_consumer_good],
            key=lambda f: f.price
        )
        tech_firms = sorted(
            [f for f in all_firms if f.industry == Industry.TECHNOLOGY and self.config.industry_configs[f.industry].is_consumer_good],
            key=lambda f: f.price
        )

        industry_firm_map = {
            "extra_food": extra_food_firms,
            "extra_shelter": extra_shelter_firms,
            "technology": tech_firms,
        }

        for worker in workers:
            budget = getattr(worker, "_post_necessity_budget", 0.0)
            after_tax = getattr(worker, "_post_necessity_income", worker.income)
            alloc = consumer_actions.get(worker.name, {}).get(
                "discretionary_allocation",
                {"extra_food": 0.0, "extra_shelter": 0.0, "technology": 0.3, "savings": 0.7},
            )

            total_spent = 0.0
            savings_amount = alloc.get("savings", 0.5) * budget

            for key, firms_list in industry_firm_map.items():
                spend_fraction = alloc.get(key, 0.0)
                spend_budget = spend_fraction * budget

                for firm in firms_list:
                    if spend_budget <= 0 or firm.inventory <= 0 or firm.price <= 0:
                        break
                    units = min(spend_budget / firm.price, firm.inventory)
                    cost = units * firm.price
                    firm.inventory -= units
                    firm.receive_revenue(cost)
                    spend_budget -= cost
                    total_spent += cost
                    results["firm_revenue"][firm.name] += cost
                    results["firm_sales"][firm.name] += units

            # Update savings: previous savings + (remaining budget = savings allocation)
            worker.savings += savings_amount + (budget - total_spent - savings_amount)
            # Simplify: savings += remaining_after_spend; total_spent = what was spent
            # Actually: worker already saved necessity remainder; now add discretionary remainder
            results["worker_spending"][worker.name] = total_spent
            results["worker_savings_change"][worker.name] = savings_amount

        return results

    def merge_revenue(self, firms: list, necessity_results: dict, discretionary_results: dict):
        """Apply combined revenue from both clearing phases to firm financials."""
        for firm in firms:
            rev = (necessity_results["firm_revenue"].get(firm.name, 0.0)
                   + discretionary_results["firm_revenue"].get(firm.name, 0.0))
            # Revenue was already applied via firm.receive_revenue() during clearing.
            # This method computes totals for statistics only.
            pass  # Revenue already applied in-place during clearing loops

    def combined_revenue(self, firms: list, necessity_results: dict, discretionary_results: dict) -> dict:
        """Return combined revenue per firm from both phases."""
        combined = {}
        for firm in firms:
            combined[firm.name] = (
                necessity_results["firm_revenue"].get(firm.name, 0.0)
                + discretionary_results["firm_revenue"].get(firm.name, 0.0)
            )
        return combined
