import logging
from typing import Dict, List, Optional

from econ_sim.config import EconomicConfig, Industry
from econ_sim.agents.government import GovernmentAgent
from econ_sim.agents.firm import FirmAgent
from econ_sim.agents.worker import WorkerAgent
from econ_sim.engine.markets import LaborMarket, InputMarket, GoodsMarket
from econ_sim.engine.production import ProductionFunction
from econ_sim.engine.statistics import StatisticsTracker, ConvergenceResult
from econ_sim.llm.client import AnthropicLLMClient
from econ_sim.output.recorder import SimulationRecorder

logger = logging.getLogger(__name__)


class EconomicEnvironment:
    """Master orchestrator for the economic simulation.

    13-phase round loop:
      1.  Government policy (LLM: Opus)
      2.  Firm decisions (LLM: per industry tier)
      3.  Input market clearing (deterministic: Manufacturing → Housing/Tech)
      4.  Worker employment decisions (LLM: Haiku × archetypes)
      5.  Labor market clearing (deterministic)
      6.  Production (deterministic, input-constrained)
      7.  Necessity goods market clearing (deterministic)
      8.  Consumer spending decisions (LLM: Haiku × archetypes)
      9.  Discretionary goods market clearing (deterministic)
      10. Tax & fiscal settlement (deterministic)
      11. Firm wage payments (deterministic)
      12. Institutional integrity dynamics (deterministic)
      13. Statistics, notify agents, record
    """

    def __init__(self, config: EconomicConfig, output_dir: str = "results"):
        self.config = config
        self.output_dir = output_dir

        self.llm = AnthropicLLMClient(
            model=config.model_opus,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        self.government = self._create_government()
        self.firms: List[FirmAgent] = self._create_firms()
        self.workers: List[WorkerAgent] = self._create_workers()
        self.worker_map: Dict[str, WorkerAgent] = {w.name: w for w in self.workers}

        self.labor_market = LaborMarket(config)
        self.input_market = InputMarket(config)
        self.goods_market = GoodsMarket(config)
        self.production = ProductionFunction(config)
        self.stats = StatisticsTracker(config)
        self.recorder = SimulationRecorder(config, output_dir)

        logger.info(
            f"Environment: {config.label()} | "
            f"c={config.coordination_mechanism:.2f} s={config.planning_structure:.2f} "
            f"m={config.meta_game_constraint:.2f} | "
            f"{len(self.firms)} firms, {len(self.workers)} worker archetypes"
        )

    # ------------------------------------------------------------------
    # Agent creation
    # ------------------------------------------------------------------

    def _create_government(self) -> GovernmentAgent:
        return GovernmentAgent("Government", self.llm, self.config)

    def _create_firms(self) -> List[FirmAgent]:
        firms = []
        for industry, ind_config in self.config.industry_configs.items():
            for i in range(ind_config.num_firms):
                name = f"{ind_config.name}_Firm_{i+1}"
                firms.append(FirmAgent(name, self.llm, self.config, ind_config))
        return firms

    def _create_workers(self) -> List[WorkerAgent]:
        workers = []
        for profile in self.config.worker_profiles:
            name = f"Worker_{profile.name.title()}"
            workers.append(WorkerAgent(name, self.llm, self.config, profile))
        return workers

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        logger.info(f"Starting simulation: max {self.config.num_rounds} rounds "
                    f"(min {self.config.min_rounds} before convergence check)")
        world_state = self._build_world_state(round_num=0)

        convergence: Optional[ConvergenceResult] = None
        for round_num in range(1, self.config.num_rounds + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"ROUND {round_num} / {self.config.num_rounds}")
            logger.info(f"{'='*60}")
            self._run_round(round_num, world_state)
            world_state = self._build_world_state(round_num)

            # Check for convergence after min_rounds
            if round_num >= self.config.min_rounds:
                convergence = self.stats.check_convergence(
                    min_rounds=self.config.min_rounds
                )
                if convergence.converged:
                    print(f"\n  ✓ Convergence at round {round_num}: {convergence.message}")
                    break

        if convergence is None or not convergence.converged:
            # Ran to max_rounds without converging — classify what we observed
            convergence = self.stats.check_convergence(
                min_rounds=self.config.min_rounds, window=min(10, len(self.stats.history))
            )
            print(f"\n  ⚠ Reached max rounds ({self.config.num_rounds}) without convergence.")
            print(f"     Best trend estimate: {convergence.trend} — {convergence.message}")

        regime = self.stats.classify_regime()
        logger.info(f"\nSimulation complete. Regime: {regime} | Trend: {convergence.trend}")

        return self.recorder.finalize(
            stats_history=self.stats.history,
            regime=regime,
            integrity_history=[s.institutional_integrity for s in self.stats.history],
            cost_summary=self.llm.cost_summary(),
            convergence=convergence,
        )

    def _run_round(self, round_num: int, world_state: dict):
        for firm in self.firms:
            firm.end_round_reset()
        for worker in self.workers:
            worker.end_round_reset()

        # ── Phase 1: Government policy ──────────────────────────────
        logger.info("Phase 1: Government policy")
        gov_action = self.government.decide(world_state, round_num)
        income_tax = gov_action["income_tax_rate"]
        corp_tax = gov_action["corporate_tax_rate"]
        industry_policies = gov_action.get("industry_policies", {})

        # ── Phase 2: Firm decisions ─────────────────────────────────
        logger.info("Phase 2: Firm decisions")
        firm_actions: dict = {}
        for firm in self.firms:
            action = firm.decide(world_state, round_num)
            firm.apply_action(action)
            firm_actions[firm.name] = action
            if firm.lobby_spending_this_round > 0:
                self.government.receive_lobby(
                    firm.name,
                    firm.lobby_spending_this_round,
                    firm.lobby_message_this_round,
                )

        # ── Phase 3: Input market clearing ──────────────────────────
        logger.info("Phase 3: Input market clearing")
        for firm in self.firms:
            ind_cfg = self.config.industry_configs[firm.industry]
            firm._input_budget_this_round = (
                firm_actions[firm.name].get("input_budget", 0.0)
                if ind_cfg.input_industry is not None else 0.0
            )
        input_results = self.input_market.clear(self.firms)

        # ── Phase 4: Worker employment decisions ────────────────────
        logger.info("Phase 4: Worker employment decisions")
        updated_ws = self._build_world_state(round_num)
        worker_employment_actions: dict = {}
        for worker in self.workers:
            action = worker.decide_employment(updated_ws, round_num)
            worker_employment_actions[worker.name] = action

        # ── Phase 5: Labor market clearing ──────────────────────────
        logger.info("Phase 5: Labor market clearing")
        # Snapshot headcount before clearing so we can compute hiring/firing costs
        for firm in self.firms:
            firm.prev_num_employees = len(firm.employees)
        labor_results = self.labor_market.clear(
            firm_actions, worker_employment_actions, self.firms, self.workers
        )
        # Apply hiring/firing costs (discourages employment cycling)
        for firm in self.firms:
            new_hires = max(0, len(firm.employees) - firm.prev_num_employees)
            layoffs   = max(0, firm.prev_num_employees - len(firm.employees))
            friction_cost = (
                new_hires * self.config.hiring_cost_factor
                + layoffs * self.config.firing_cost_factor
            ) * firm.wage
            if friction_cost > 0:
                firm.capital -= friction_cost
                firm.costs   += friction_cost
                logger.debug(
                    f"  {firm.name}: +{new_hires} hires, -{layoffs} layoffs, "
                    f"friction=${friction_cost:.2f}"
                )

        # ── Phase 6: Production ─────────────────────────────────────
        logger.info("Phase 6: Production")
        for firm in self.firms:
            units = self.production.produce(firm, self.worker_map)
            logger.debug(f"  {firm.name}: {units:.1f} units produced")

        # ── Phase 7: Necessity goods market clearing ────────────────
        logger.info("Phase 7: Necessity clearing")
        transfer_per_worker = (
            gov_action["spending"].get("transfers_to_workers", 0.0)
            / max(len(self.workers), 1)
        )
        necessity_results = self.goods_market.clear_necessities(
            self.firms, self.workers, income_tax, transfer_per_worker
        )

        # ── Phase 8: Consumer spending decisions ────────────────────
        logger.info("Phase 8: Consumer decisions")
        updated_ws2 = self._build_world_state(round_num, necessity_results=necessity_results)
        consumer_actions: dict = {}
        for worker in self.workers:
            action = worker.decide_consumption(updated_ws2, round_num)
            consumer_actions[worker.name] = action

        # ── Phase 9: Discretionary goods market clearing ────────────
        logger.info("Phase 9: Discretionary clearing")
        disc_results = self.goods_market.clear_discretionary(
            self.firms, self.workers, consumer_actions, necessity_results
        )

        # ── Phase 10: Tax & fiscal settlement ───────────────────────
        logger.info("Phase 10: Fiscal settlement")
        fiscal_results = self._settle_fiscal(
            income_tax, corp_tax, gov_action, industry_policies
        )

        # ── Phase 11: Firm wage payments ────────────────────────────
        for firm in self.firms:
            firm.pay_wages()

        # ── Phase 12: Institutional integrity dynamics ───────────────
        logger.info("Phase 12: Integrity dynamics")
        self.government.apply_lobby_effects(self.config.lobby_detection_prob)
        total_suffering = sum(getattr(w, "suffering_score", 0.0) for w in self.workers)
        self.government.apply_public_pressure(
            total_suffering, self.config.suffering_pressure_threshold
        )
        self.government.clear_lobby_buffer()

        # ── Phase 13: Statistics & recording ────────────────────────
        logger.info("Phase 13: Statistics")
        goods_revenue = self.goods_market.combined_revenue(self.firms, necessity_results, disc_results)
        b2b_revenue = input_results.get("mfg_b2b_revenue", {})

        round_stats = self.stats.update(
            round_num=round_num,
            firms=self.firms,
            workers=self.workers,
            government=self.government,
            goods_revenue=goods_revenue,
            b2b_revenue=b2b_revenue,
            fiscal_results=fiscal_results,
            necessity_results=necessity_results,
        )

        self._notify_agents(round_stats)
        self.recorder.record_round(
            round_num=round_num,
            stats=round_stats,
            gov_action=gov_action,
            firm_states=self._firm_snapshots(),
            worker_states=self._worker_snapshots(),
            institutional_integrity=self.government.institutional_integrity,
            labor_results=labor_results,
        )
        self._print_round_summary(round_num, round_stats)

    # ------------------------------------------------------------------
    # Fiscal settlement
    # ------------------------------------------------------------------

    def _settle_fiscal(
        self,
        income_tax: float,
        corp_tax: float,
        gov_action: dict,
        industry_policies: dict,
    ) -> dict:
        income_tax_rev = sum(w.income * income_tax for w in self.workers)

        corp_tax_rev = 0.0
        for firm in self.firms:
            if firm.profit > 0:
                tax = firm.profit * corp_tax
                firm.capital -= tax
                corp_tax_rev += tax

        # Industry-specific subsidies
        for industry_name, policy in industry_policies.items():
            subsidy = policy.get("subsidy", 0.0)
            industry_firms = [f for f in self.firms if f.industry.value == industry_name]
            if industry_firms and subsidy > 0:
                per_firm = subsidy / len(industry_firms)
                for f in industry_firms:
                    f.capital += per_firm
                    f.revenue += per_firm

        total_tax = self.government.collect_taxes(income_tax_rev, corp_tax_rev)
        actual_spending = self.government.execute_spending(gov_action)

        return {
            "income_tax_revenue": income_tax_rev,
            "corporate_tax_revenue": corp_tax_rev,
            "total_tax": total_tax,
            "spending": actual_spending,
        }

    # ------------------------------------------------------------------
    # World state
    # ------------------------------------------------------------------

    def _build_world_state(self, round_num: int, necessity_results: Optional[dict] = None) -> dict:
        stats = self.stats.current()
        last_action = getattr(self.government, "_last_action", {})
        gov_policy = last_action if isinstance(last_action, dict) else {}

        return {
            "system_label": self.config.label(),
            "round_num": round_num,
            "coordination_mechanism": self.config.coordination_mechanism,
            "planning_structure": self.config.planning_structure,
            "meta_game_constraint": self.config.meta_game_constraint,
            "institutional_integrity": self.government.institutional_integrity,
            "government_policy": {
                "income_tax_rate": gov_policy.get("income_tax_rate", 0.15),
                "corporate_tax_rate": gov_policy.get("corporate_tax_rate", 0.12),
                "spending": gov_policy.get("spending", {}),
            },
            "industry_subsidies": {},
            "firms": self._firm_snapshots(),
            "workers": self._worker_snapshots(),
            "statistics": stats,
            "gdp": stats.get("gdp", 0.0),
            "unemployment_rate": stats.get("unemployment_rate", 0.0),
            "inflation": stats.get("inflation", 0.0),
            "avg_wage": stats.get("avg_wage", 0.0),
            "consumer_demand_index": max(0.1, 1.0 - stats.get("unemployment_rate", 0.0)),
            "unemployed_count": sum(1 for w in self.workers if w.employer is None),
            "total_workers": sum(w.representative_count for w in self.workers),
            "necessity_cost_this_round": (necessity_results or {}).get("necessity_cost", {}),
        }

    def _firm_snapshots(self) -> list:
        return [
            {
                "name": f.name,
                "industry": f.industry.value,
                "price": f.price,
                "wage": f.wage,
                "capital": f.capital,
                "inventory": f.inventory,
                "input_inventory": f.input_inventory,
                "num_employees": f.num_employees,
                "revenue": f.revenue,
                "costs": f.costs,
                "profit": f.profit,
                "lobby_spending": f.lobby_spending_this_round,
                "is_consumer_good": self.config.industry_configs[f.industry].is_consumer_good,
            }
            for f in self.firms
        ]

    def _worker_snapshots(self) -> list:
        return [
            {
                "name": w.name,
                "archetype": w.profile.name,
                "skill_level": w.skill_level,
                "representative_count": w.representative_count,
                "employer": w.employer,
                "industry": w.industry.value if w.industry else None,
                "wage": w.wage,
                "savings": w.savings,
                "suffering_score": w.suffering_score,
                "productivity_modifier": w.productivity_modifier,
                "food_consumed": w.food_consumed,
                "shelter_consumed": w.shelter_consumed,
            }
            for w in self.workers
        ]

    # ------------------------------------------------------------------
    # Notifications & output
    # ------------------------------------------------------------------

    def _notify_agents(self, stats):
        summary = (
            f"Round {stats.round_num}: GDP=${stats.gdp:.0f} ({stats.gdp_growth:+.1%}), "
            f"unemployment={stats.unemployment_rate:.1%}, W={stats.welfare_score:.3f}, "
            f"Gini={stats.gini:.3f}, needs_met={stats.basic_needs_fulfillment_rate:.1%}, "
            f"integrity={stats.institutional_integrity:.2f}"
        )
        self.government.receive_notification(summary)
        for firm in self.firms:
            firm.receive_notification(
                f"Round {stats.round_num}: GDP=${stats.gdp:.0f}, "
                f"unemployment={stats.unemployment_rate:.1%}, "
                f"your profit=${firm.profit:.0f}, capital=${firm.capital:.0f}"
            )
        for worker in self.workers:
            worker.receive_notification(
                f"Round {stats.round_num}: savings=${worker.savings:.0f}, "
                f"suffering={worker.suffering_score:.2f}, "
                f"food={worker.food_consumed:.1f}/1.0, shelter={worker.shelter_consumed:.1f}/0.5"
            )

    def _print_round_summary(self, round_num: int, stats):
        def bar(v, n=10):
            filled = int(max(0.0, min(1.0, v)) * n)
            return "█" * filled + "░" * (n - filled)

        print(f"\n── Round {round_num} ─────────────────────────────────────────")
        print(f"  GDP:          ${stats.gdp:>10.2f}  ({stats.gdp_growth:+.1%})")
        print(f"  Unemployment:  {stats.unemployment_rate:>9.1%}")
        print(f"  Gini:          {stats.gini:>9.3f}")
        print(f"  Welfare W:     {stats.welfare_score:>9.3f}  [{bar(stats.welfare_score)}]")
        print(f"  Basic needs:   {stats.basic_needs_fulfillment_rate:>9.1%}")
        print(f"  Suffering:     {stats.total_suffering:>9.2f}")
        print(f"  Integrity:     {self.government.institutional_integrity:>9.2f}  [{bar(self.government.institutional_integrity)}]")
        print(f"  Lobby $:      ${stats.total_lobby_spending:>10.2f}")
        print(f"  Treasury:     ${self.government.treasury:>10.2f}")
        if stats.industry_gdp:
            print(f"  Industry GDP:")
            for ind, rev in stats.industry_gdp.items():
                print(f"    {ind:<16} ${rev:.2f}")

        # Live trend estimate (shown once we have enough history)
        if round_num >= self.config.min_rounds:
            cr = self.stats.check_convergence(min_rounds=self.config.min_rounds)
            trend_icons = {
                "stable": "━", "rising": "▲", "falling": "▼",
                "oscillating": "↕", "collapse": "✗", "unknown": "…",
            }
            icon = trend_icons.get(cr.trend, "?")
            if cr.trend != "unknown":
                print(f"  Trend:        {icon} {cr.trend:<12} conf={cr.confidence:.0%}  {cr.message}")
            else:
                print(f"  Trend:        {icon} {cr.message}")
        print()
