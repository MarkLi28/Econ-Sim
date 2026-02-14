from dataclasses import dataclass, field, asdict


@dataclass
class RoundStatistics:
    round_num: int = 0
    gdp: float = 0.0
    unemployment_rate: float = 0.0
    avg_wage: float = 0.0
    avg_price: float = 0.0
    inflation: float = 0.0
    gini: float = 0.0
    total_production: float = 0.0
    total_demand: float = 0.0
    total_supply: float = 0.0
    tax_revenue: float = 0.0
    government_treasury: float = 0.0
    num_workers: int = 0
    num_employed: int = 0
    firm_profits: dict = field(default_factory=dict)
    worker_wealth: dict = field(default_factory=dict)


class StatisticsTracker:
    """Computes and tracks aggregate economic statistics each round."""

    def __init__(self):
        self.history: list[RoundStatistics] = []
        self._prev_avg_price: float | None = None

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
        goods_results: dict,
        fiscal_results: dict,
    ):
        employed = [w for w in workers if w.employer is not None]
        num_employed = len(employed)
        num_workers = len(workers)
        unemployment_rate = 1.0 - num_employed / max(num_workers, 1)

        avg_wage = sum(w.wage for w in employed) / max(num_employed, 1) if employed else 0.0
        avg_price = sum(f.price for f in firms) / max(len(firms), 1) if firms else 0.0

        inflation = 0.0
        if self._prev_avg_price and self._prev_avg_price > 0:
            inflation = (avg_price - self._prev_avg_price) / self._prev_avg_price

        gdp = sum(goods_results["firm_revenue"].values())

        wealth_values = [w.savings for w in workers]
        gini = self._compute_gini(wealth_values)

        stats = RoundStatistics(
            round_num=round_num,
            gdp=gdp,
            unemployment_rate=unemployment_rate,
            avg_wage=avg_wage,
            avg_price=avg_price,
            inflation=inflation,
            gini=gini,
            total_production=sum(f.inventory for f in firms) + sum(goods_results["firm_sales"].values()),
            total_demand=sum(goods_results["worker_spending"].values()) + goods_results["unmet_demand"],
            total_supply=sum(f.inventory for f in firms),
            tax_revenue=fiscal_results["total_tax"],
            government_treasury=government.treasury,
            num_workers=num_workers,
            num_employed=num_employed,
            firm_profits={f.name: f.revenue - f.costs for f in firms},
            worker_wealth={w.name: w.savings for w in workers},
        )
        self.history.append(stats)
        self._prev_avg_price = avg_price

    @staticmethod
    def _compute_gini(values: list[float]) -> float:
        if not values or all(v == 0 for v in values):
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        total = sum(sorted_vals)
        if total == 0:
            return 0.0
        cumulative = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sorted_vals))
        return max(0.0, cumulative / (n * total))
