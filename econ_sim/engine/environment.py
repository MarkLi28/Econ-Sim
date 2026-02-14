import logging

from econ_sim.config import EconomicConfig
from econ_sim.llm.client import AnthropicLLMClient
from econ_sim.llm.parser import ResponseParser
from econ_sim.agents.government import GovernmentAgent
from econ_sim.agents.firm import FirmAgent
from econ_sim.agents.worker import WorkerAgent
from econ_sim.engine.markets import LaborMarket, GoodsMarket
from econ_sim.engine.production import ProductionFunction
from econ_sim.engine.statistics import StatisticsTracker
from econ_sim.output.recorder import SimulationRecorder

logger = logging.getLogger(__name__)


class EconomicEnvironment:
    """Master orchestrator for the economic simulation.

    Runs the 10-phase game loop each round:
      1. Government policy
      2. Firm decisions
      3. Worker decisions
      4. Labor market clearing
      5. Production
      6. Goods market clearing
      7. Tax collection & transfers
      8. Statistics update
      9. Notify all agents
      10. Record round data
    """

    def __init__(self, config: EconomicConfig, llm_client: AnthropicLLMClient, output_dir: str = "results"):
        self.config = config
        self.round_num = 0

        # Create agents
        self.government = GovernmentAgent("Federal_Government", llm_client, config)
        self.firms = [
            FirmAgent(f"Firm_{i}", llm_client, config)
            for i in range(config.num_firms)
        ]
        self.workers = [
            WorkerAgent(
                f"Worker_{i}",
                llm_client,
                config,
                skill_level=round(0.8 + 0.4 * (i / max(config.num_workers - 1, 1)), 2),
            )
            for i in range(config.num_workers)
        ]

        # Engine components
        self.labor_market = LaborMarket(config)
        self.goods_market = GoodsMarket(config)
        self.production = ProductionFunction(config)
        self.statistics = StatisticsTracker()
        self.recorder = SimulationRecorder(output_dir, config)

        # Current government policy (updated each round)
        self._current_policy: dict = {
            "income_tax_rate": sum(config.income_tax_rate_range) / 2,
            "corporate_tax_rate": sum(config.corporate_tax_rate_range) / 2,
            "spending": {"infrastructure": 0, "transfers_to_workers": 0, "subsidies_to_firms": 0},
        }

    def get_world_state(self) -> dict:
        """Snapshot of the entire economy — passed to agents as context."""
        return {
            "round": self.round_num,
            "government_policy": self._current_policy,
            "firms": [self._firm_public_info(f) for f in self.firms],
            "workers": [self._worker_public_info(w) for w in self.workers],
            "statistics": self.statistics.current(),
        }

    def _firm_public_info(self, firm: FirmAgent) -> dict:
        return {
            "name": firm.name,
            "price": firm.price,
            "wage": firm.wage,
            "num_employees": len(firm.employees),
            "revenue": firm.revenue,
            "inventory": firm.inventory,
        }

    def _worker_public_info(self, worker: WorkerAgent) -> dict:
        return {
            "name": worker.name,
            "employer": worker.employer,
            "savings": worker.savings,
            "skill_level": worker.skill_level,
        }

    def run_simulation(self):
        """Run the full simulation for all configured rounds."""
        logger.info(
            f"Starting {self.config.system.value} simulation: "
            f"{self.config.num_rounds} rounds, {self.config.num_firms} firms, "
            f"{self.config.num_workers} workers"
        )

        for round_num in range(1, self.config.num_rounds + 1):
            self.run_single_round(round_num)

        self.recorder.save(self.government, self.firms, self.workers)
        logger.info("Simulation complete.")

    def run_single_round(self, round_num: int):
        """Execute one round of the economic simulation (10 phases)."""
        self.round_num = round_num
        logger.info(f"--- Round {round_num}/{self.config.num_rounds} ---")

        # === PHASE 1: GOVERNMENT POLICY ===
        logger.info("Phase 1: Government policy")
        world_state = self.get_world_state()
        govt_action = self.government.decide(world_state, round_num)
        self._apply_government_policy(govt_action)

        # === PHASE 2: FIRM DECISIONS ===
        logger.info("Phase 2: Firm decisions")
        world_state = self.get_world_state()
        firm_actions = {}
        for firm in self.firms:
            action = firm.decide(world_state, round_num)
            firm_actions[firm.name] = action
            firm.price = action["price"]

        # === PHASE 3: WORKER DECISIONS ===
        logger.info("Phase 3: Worker decisions")
        world_state = self.get_world_state()
        worker_actions = {}
        firm_names = [f.name for f in self.firms]
        for worker in self.workers:
            action = worker.decide(world_state, round_num)
            # Re-validate employer choice against actual firm names
            action = ResponseParser.validate_worker_action(action, firm_names)
            worker_actions[worker.name] = action

        # === PHASE 4: LABOR MARKET CLEARING ===
        logger.info("Phase 4: Labor market clearing")
        labor_results = self.labor_market.clear(firm_actions, worker_actions, self.firms, self.workers)

        # === PHASE 5: PRODUCTION ===
        logger.info("Phase 5: Production")
        production_results = {}
        for firm in self.firms:
            output = self.production.produce(firm)
            production_results[firm.name] = output

        # === PHASE 6: GOODS MARKET CLEARING ===
        logger.info("Phase 6: Goods market clearing")
        transfer_per_worker = self._current_policy["spending"].get("transfers_to_workers", 0) / max(
            len(self.workers), 1
        )
        goods_results = self.goods_market.clear(
            self.firms,
            self.workers,
            worker_actions,
            self._current_policy["income_tax_rate"],
            transfer_per_worker,
        )

        # === PHASE 7: TAX COLLECTION & FISCAL SETTLEMENT ===
        logger.info("Phase 7: Fiscal settlement")
        fiscal_results = self._fiscal_settlement(goods_results, labor_results)

        # === PHASE 8: UPDATE FIRM FINANCIALS ===
        logger.info("Phase 8: Update financials")
        self._update_firm_financials(firm_actions, goods_results, fiscal_results)

        # === PHASE 9: STATISTICS UPDATE ===
        logger.info("Phase 9: Statistics update")
        self.statistics.update(
            round_num, self.firms, self.workers, self.government,
            goods_results, fiscal_results,
        )

        # === PHASE 10: NOTIFY ALL AGENTS & RECORD ===
        logger.info("Phase 10: Notify & record")
        summary = self._build_round_summary(
            round_num, labor_results, goods_results, fiscal_results, production_results,
        )
        for agent in [self.government] + self.firms + self.workers:
            agent.receive_notification(summary)

        self.recorder.record_round(
            round_num, self.get_world_state(),
            govt_action, firm_actions, worker_actions,
            labor_results, goods_results, fiscal_results,
        )

        # Print summary to console
        stats = self.statistics.current()
        print(
            f"  Round {round_num}: GDP=${stats['gdp']:.0f}  "
            f"Unemployment={stats['unemployment_rate']*100:.0f}%  "
            f"AvgWage=${stats['avg_wage']:.0f}  "
            f"AvgPrice=${stats['avg_price']:.1f}  "
            f"Gini={stats['gini']:.3f}  "
            f"Treasury=${stats['government_treasury']:.0f}"
        )

    def _apply_government_policy(self, action: dict):
        """Apply government policy and deduct spending from treasury."""
        self._current_policy = action
        total_spending = sum(action["spending"].values())
        self.government.treasury -= total_spending

    def _fiscal_settlement(self, goods_results: dict, labor_results: dict) -> dict:
        """Collect taxes from firms and workers, update treasury."""
        income_tax_rate = self._current_policy["income_tax_rate"]
        corporate_tax_rate = self._current_policy["corporate_tax_rate"]

        # Income tax: already implicitly handled in goods market (after-tax income).
        # But we need to actually collect it for the treasury.
        income_tax_collected = sum(
            w.income * income_tax_rate for w in self.workers if w.employer is not None
        )

        # Corporate tax on revenue
        corporate_tax_collected = sum(
            goods_results["firm_revenue"].get(f.name, 0) * corporate_tax_rate
            for f in self.firms
        )

        total_tax = income_tax_collected + corporate_tax_collected
        self.government.treasury += total_tax

        # Infrastructure spending boosts productivity next round (simple bonus)
        infra_spending = self._current_policy["spending"].get("infrastructure", 0)
        if infra_spending > 0:
            # Small productivity boost proportional to infrastructure spending
            self.config.productivity_factor = max(
                0.5, 1.0 + infra_spending / 1000.0
            )

        return {
            "income_tax_collected": income_tax_collected,
            "corporate_tax_collected": corporate_tax_collected,
            "total_tax": total_tax,
        }

    def _update_firm_financials(self, firm_actions: dict, goods_results: dict, fiscal_results: dict):
        """Update firm capital, revenue, costs after market clearing."""
        corporate_tax_rate = self._current_policy["corporate_tax_rate"]
        subsidy_per_firm = self._current_policy["spending"].get("subsidies_to_firms", 0) / max(
            len(self.firms), 1
        )

        for firm in self.firms:
            firm.revenue = goods_results["firm_revenue"].get(firm.name, 0)
            wage = firm_actions[firm.name]["wage_offer"]
            firm.costs = wage * len(firm.employees)
            profit = firm.revenue - firm.costs
            tax = firm.revenue * corporate_tax_rate
            firm.capital += profit - tax + subsidy_per_firm

    def _build_round_summary(
        self, round_num, labor_results, goods_results, fiscal_results, production_results,
    ) -> str:
        stats = self.statistics.current()
        employed_list = [
            f"{name} → {employer}"
            for name, employer in labor_results["assignments"].items()
        ]
        unemployed_str = ", ".join(labor_results["unemployed"]) or "none"

        firm_lines = []
        for f in self.firms:
            sold = goods_results["firm_sales"].get(f.name, 0)
            rev = goods_results["firm_revenue"].get(f.name, 0)
            produced = production_results.get(f.name, 0)
            firm_lines.append(
                f"  {f.name}: produced {produced:.0f} units, sold {sold:.1f} units "
                f"at ${f.price:.1f}, revenue=${rev:.0f}, employees={len(f.employees)}, "
                f"capital=${f.capital:.0f}"
            )

        return (
            f"=== ROUND {round_num} RESULTS ===\n\n"
            f"Employment:\n"
            f"  Hired: {', '.join(employed_list) if employed_list else 'none'}\n"
            f"  Unemployed: {unemployed_str}\n\n"
            f"Production & Sales:\n" + "\n".join(firm_lines) + "\n\n"
            f"Fiscal:\n"
            f"  Income Tax Collected: ${fiscal_results['income_tax_collected']:.0f}\n"
            f"  Corporate Tax Collected: ${fiscal_results['corporate_tax_collected']:.0f}\n"
            f"  Government Treasury: ${self.government.treasury:.0f}\n\n"
            f"Aggregate:\n"
            f"  GDP: ${stats['gdp']:.0f}\n"
            f"  Unemployment Rate: {stats['unemployment_rate']*100:.1f}%\n"
            f"  Average Wage: ${stats['avg_wage']:.0f}\n"
            f"  Average Price: ${stats['avg_price']:.1f}\n"
            f"  Inflation: {stats['inflation']*100:.1f}%\n"
            f"  Gini Coefficient: {stats['gini']:.3f}\n"
        )
