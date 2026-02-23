from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


@dataclass
class RoundStatistics:
    round_num: int = 0

    # Macro
    gdp: float = 0.0
    gdp_growth: float = 0.0
    unemployment_rate: float = 0.0
    avg_wage: float = 0.0
    avg_price: float = 0.0
    inflation: float = 0.0
    gini: float = 0.0

    # Production
    total_production: float = 0.0
    total_consumer_spending: float = 0.0

    # Fiscal
    tax_revenue: float = 0.0
    government_treasury: float = 0.0
    fiscal_health: float = 1.0   # treasury / initial_treasury

    # Welfare (research outputs)
    welfare_score: float = 0.0
    basic_needs_fulfillment_rate: float = 1.0
    total_suffering: float = 0.0
    aggregate_suffering_delta: float = 0.0  # increase this round

    # Institutional
    institutional_integrity: float = 1.0
    total_lobby_spending: float = 0.0

    # Per-agent breakdowns
    num_workers: int = 0
    num_employed: int = 0
    firm_profits: Dict[str, float] = field(default_factory=dict)
    worker_savings: Dict[str, float] = field(default_factory=dict)

    # Industry-level breakdown
    industry_gdp: Dict[str, float] = field(default_factory=dict)
    industry_employment: Dict[str, int] = field(default_factory=dict)
    industry_avg_price: Dict[str, float] = field(default_factory=dict)


class StatisticsTracker:
    """Computes and tracks aggregate economic statistics each round."""

    def __init__(self, config):
        self.config = config
        self.history: list[RoundStatistics] = []
        self._prev_avg_price: Optional[float] = None
        self._prev_gdp: Optional[float] = None

    def current(self) -> dict:
        if not self.history:
            return asdict(RoundStatistics())
        return asdict(self.history[-1])

    def update(
        self,
        round_num: int,
        firms: list,
        workers: list,
        government,
        goods_revenue: dict,       # {firm_name: total revenue from goods market}
        b2b_revenue: dict,         # {firm_name: B2B revenue from input market}
        fiscal_results: dict,
        necessity_results: dict,
    ) -> RoundStatistics:

        employed = [w for w in workers if w.employer is not None]
        num_employed = len(employed)
        num_workers = len(workers)
        unemployment_rate = 1.0 - num_employed / max(num_workers, 1)

        avg_wage = sum(w.wage for w in employed) / max(num_employed, 1) if employed else 0.0

        consumer_firms = [
            f for f in firms
            if self.config.industry_configs[f.industry].is_consumer_good
        ]
        avg_price = sum(f.price for f in consumer_firms) / max(len(consumer_firms), 1) if consumer_firms else 0.0

        inflation = 0.0
        if self._prev_avg_price and self._prev_avg_price > 0:
            inflation = (avg_price - self._prev_avg_price) / self._prev_avg_price

        # GDP = total consumer + B2B revenue
        total_consumer_rev = sum(goods_revenue.values())
        total_b2b_rev = sum(b2b_revenue.values())
        gdp = total_consumer_rev + total_b2b_rev

        gdp_growth = 0.0
        if self._prev_gdp and self._prev_gdp > 0:
            gdp_growth = (gdp - self._prev_gdp) / self._prev_gdp

        # Wealth inequality: include both workers and firm owners (represented by firm capital)
        all_wealth = [w.savings for w in workers] + [f.capital for f in firms]
        gini = self._compute_gini(all_wealth)

        # Necessity fulfillment
        food_met = sum(
            1 for w in workers
            if necessity_results.get("food_consumed", {}).get(w.name, 0) >= 1.0
        )
        shelter_met = sum(
            1 for w in workers
            if necessity_results.get("shelter_consumed", {}).get(w.name, 0) >= 0.5
        )
        both_met = sum(
            1 for w in workers
            if (necessity_results.get("food_consumed", {}).get(w.name, 0) >= 1.0
                and necessity_results.get("shelter_consumed", {}).get(w.name, 0) >= 0.5)
        )
        basic_needs_rate = both_met / max(num_workers, 1)

        total_suffering = sum(getattr(w, "suffering_score", 0.0) for w in workers)
        prev_suffering = (
            self.history[-1].total_suffering if self.history else 0.0
        )
        suffering_delta = total_suffering - prev_suffering

        # Institutional
        integrity = government.institutional_integrity
        total_lobby = sum(getattr(f, "lobby_spending_this_round", 0.0) for f in firms)

        # Fiscal health
        fiscal_health = government.treasury / max(self.config.initial_government_treasury, 1.0)

        # Welfare score
        welfare = self._compute_welfare(
            unemployment_rate=unemployment_rate,
            gini=gini,
            basic_needs_rate=basic_needs_rate,
            gdp_growth=gdp_growth,
            fiscal_health=fiscal_health,
        )

        # Industry breakdown
        industry_gdp: dict = {}
        industry_employment: dict = {}
        industry_prices: dict = {}
        for f in firms:
            ind = f.industry.value
            rev = goods_revenue.get(f.name, 0.0) + b2b_revenue.get(f.name, 0.0)
            industry_gdp[ind] = industry_gdp.get(ind, 0.0) + rev
            industry_employment[ind] = industry_employment.get(ind, 0) + f.num_employees
            if self.config.industry_configs[f.industry].is_consumer_good:
                industry_prices.setdefault(ind, []).append(f.price)
        industry_avg_price = {
            ind: sum(ps) / len(ps) for ind, ps in industry_prices.items()
        }

        stats = RoundStatistics(
            round_num=round_num,
            gdp=gdp,
            gdp_growth=gdp_growth,
            unemployment_rate=unemployment_rate,
            avg_wage=avg_wage,
            avg_price=avg_price,
            inflation=inflation,
            gini=gini,
            total_production=sum(f.inventory for f in firms) + sum(goods_revenue.values()),
            total_consumer_spending=total_consumer_rev,
            tax_revenue=fiscal_results.get("total_tax", 0.0),
            government_treasury=government.treasury,
            fiscal_health=max(0.0, fiscal_health),
            welfare_score=welfare,
            basic_needs_fulfillment_rate=basic_needs_rate,
            total_suffering=total_suffering,
            aggregate_suffering_delta=suffering_delta,
            institutional_integrity=integrity,
            total_lobby_spending=total_lobby,
            num_workers=num_workers,
            num_employed=num_employed,
            firm_profits={f.name: f.profit for f in firms},
            worker_savings={w.name: w.savings for w in workers},
            industry_gdp=industry_gdp,
            industry_employment=industry_employment,
            industry_avg_price=industry_avg_price,
        )

        self.history.append(stats)
        self._prev_avg_price = avg_price
        self._prev_gdp = gdp
        return stats

    @staticmethod
    def _compute_welfare(
        unemployment_rate: float,
        gini: float,
        basic_needs_rate: float,
        gdp_growth: float,
        fiscal_health: float,
    ) -> float:
        """Social welfare function W ∈ [0, 1]."""
        gdp_term = max(-1.0, min(1.0, gdp_growth)) * 0.5 + 0.5  # map [-1,1] → [0,1]
        fiscal_term = max(0.0, min(1.0, fiscal_health))
        return (
            0.30 * (1.0 - unemployment_rate)
            + 0.25 * (1.0 - gini)
            + 0.25 * basic_needs_rate
            + 0.10 * gdp_term
            + 0.10 * fiscal_term
        )

    def classify_regime(self) -> str:
        """Classify the simulation outcome based on trajectory."""
        if len(self.history) < 3:
            return "insufficient_data"

        last = self.history[-1]
        first = self.history[0]
        mid = len(self.history) // 2

        integrity_trend = last.institutional_integrity - self.history[mid].institutional_integrity
        welfare_trend = last.welfare_score - first.welfare_score
        gini_trend = last.gini - first.gini

        # Full capture: integrity collapsed, inequality rose, welfare fell
        if last.institutional_integrity < 0.25 and gini_trend > 0.1 and welfare_trend < -0.1:
            return "firm_capture"

        # Authoritarian: integrity held via suppression but welfare still fell
        if (self.config.meta_game_constraint > 0.7
                and last.welfare_score < 0.4
                and last.basic_needs_fulfillment_rate < 0.6):
            return "authoritarian"

        # Economic collapse
        if last.unemployment_rate > 0.8 or last.gdp < first.gdp * 0.2:
            return "collapse"

        # Stable
        if last.welfare_score > 0.55 and abs(integrity_trend) < 0.2:
            return "stable_mixed"

        # Gradual capture in progress
        if integrity_trend < -0.2:
            return "capture_in_progress"

        return "mixed_unstable"

    @staticmethod
    def _compute_gini(values: list) -> float:
        if not values or all(v == 0 for v in values):
            return 0.0
        sorted_vals = sorted(max(0.0, v) for v in values)
        n = len(sorted_vals)
        total = sum(sorted_vals)
        if total == 0:
            return 0.0
        cumulative = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sorted_vals))
        return max(0.0, cumulative / (n * total))
