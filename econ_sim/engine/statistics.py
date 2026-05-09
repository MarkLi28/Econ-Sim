from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


@dataclass
class ConvergenceResult:
    """Result of a convergence/trend check on welfare AND integrity time-series.

    `trend` is the combined label produced by _compose_trend (e.g. "stable",
    "capture_in_progress", "decay"). `welfare_trend` and `integrity_trend`
    are the per-series classifications that produced it.

    `converged` is True only when the combined classification has held for
    `persistence_required` consecutive rounds — guards against false-positive
    convergence on transient stability windows.
    """
    converged: bool
    trend: str          # combined label (see _compose_trend)
    confidence: float   # 0.0-1.0
    rounds_observed: int
    message: str
    welfare_trend: str = "unknown"
    integrity_trend: str = "unknown"
    persistence_streak: int = 0
    persistence_required: int = 1


@dataclass
class _SeriesClassification:
    """Per-series classification produced by _classify_series."""
    label: str        # stable | rising | falling | oscillating | bottomed | unknown
    slope: float      # linear regression slope over the window
    r_sq: float       # R² of the linear fit
    variance: float   # sample variance of the window
    confidence: float # 0-1 informational confidence proxy


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
        # Persistence tracking for convergence detector
        self._last_classification: Optional[tuple[str, str]] = None
        self._classification_streak: int = 0

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

    # ------------------------------------------------------------------
    # Convergence detection (welfare + integrity, with persistence)
    # ------------------------------------------------------------------

    def check_convergence(self, min_rounds: int = 15, window: int = 7) -> ConvergenceResult:
        """Check whether the simulation has reached a classifiable end-game state.

        Each round we classify the welfare and integrity time-series
        independently into one of: stable / rising / falling / oscillating /
        bottomed / unknown. Their pair is then composed into a combined
        regime label (e.g. stable+falling-integrity → "capture_in_progress").

        Convergence requires the SAME (welfare_label, integrity_label) pair to
        hold for `convergence_persistence_rounds` consecutive rounds — this
        guards against false positives on transient stability windows. The
        streak is tracked across calls via tracker state.

        The caller should stop early only when `converged=True`.
        """
        n = len(self.history)
        K = max(1, getattr(self.config, "convergence_persistence_rounds", 5))

        if n < min_rounds:
            return ConvergenceResult(
                False, "unknown", 0.0, n,
                f"Only {n} rounds — need ≥{min_rounds} before assessing",
                persistence_required=K,
            )

        recent = self.history[-min(window, n):]
        welfare_window   = [s.welfare_score           for s in recent]
        integrity_window = [s.institutional_integrity for s in recent]
        full_welfare     = [s.welfare_score           for s in self.history]
        full_integrity   = [s.institutional_integrity for s in self.history]

        welfare_class = _classify_series(
            welfare_window, full_welfare, n_total=n,
            var_threshold=0.0015, bottom_threshold=0.10,
        )
        integrity_class = _classify_series(
            integrity_window, full_integrity, n_total=n,
            var_threshold=0.005, bottom_threshold=0.20,
        )

        combined_label = _compose_trend(welfare_class.label, integrity_class.label)
        confidence = min(welfare_class.confidence, integrity_class.confidence)

        # ── Persistence streak (tracker state) ──────────────────────────
        current_pair = (welfare_class.label, integrity_class.label)
        both_classifiable = "unknown" not in current_pair
        if both_classifiable and current_pair == self._last_classification:
            self._classification_streak += 1
        elif both_classifiable:
            self._classification_streak = 1
        else:
            self._classification_streak = 0
        self._last_classification = current_pair

        converged = both_classifiable and self._classification_streak >= K

        # ── Build human-readable message ────────────────────────────────
        msg = (
            f"welfare={welfare_class.label} (slope={welfare_class.slope:+.4f}, "
            f"R²={welfare_class.r_sq:.2f}); "
            f"integrity={integrity_class.label} (slope={integrity_class.slope:+.4f}, "
            f"R²={integrity_class.r_sq:.2f}); "
            f"combined={combined_label}; "
            f"streak={self._classification_streak}/{K}"
        )

        return ConvergenceResult(
            converged=converged,
            trend=combined_label,
            confidence=confidence,
            rounds_observed=n,
            message=msg,
            welfare_trend=welfare_class.label,
            integrity_trend=integrity_class.label,
            persistence_streak=self._classification_streak,
            persistence_required=K,
        )

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


# ---------------------------------------------------------------------------
# Module-level helpers for convergence detection
# ---------------------------------------------------------------------------

def _classify_series(
    window: list[float],
    full_history: list[float],
    *,
    n_total: int,
    var_threshold: float,
    bottom_threshold: float,
) -> _SeriesClassification:
    """Classify a single time-series window into a trend label.

    Labels:
      bottomed    — below `bottom_threshold` for 3+ consecutive rounds (full history)
      oscillating — alternating round-to-round diffs (≥70%) with amplitude > 0.02
      stable      — |slope| < 0.005 and variance < `var_threshold`
      rising      — slope > 0 with R² > 0.80, |slope| > 0.008  (or weaker if n_total ≥ 15)
      falling     — symmetric for negative slope
      unknown     — none of the above conditions met
    """
    w = len(window)

    # Rock-bottom: collapse / capture detection on full history
    if len(full_history) >= 3 and all(v < bottom_threshold for v in full_history[-3:]):
        return _SeriesClassification(
            label="bottomed",
            slope=0.0,
            r_sq=1.0,
            variance=0.0,
            confidence=0.95,
        )

    # Linear regression fit
    if w < 2:
        return _SeriesClassification("unknown", 0.0, 0.0, 0.0, 0.0)

    x_mean = (w - 1) / 2.0
    y_mean = sum(window) / w
    xy_cov = sum((i - x_mean) * (window[i] - y_mean) for i in range(w))
    x_var  = sum((i - x_mean) ** 2 for i in range(w))
    slope  = xy_cov / x_var if x_var > 0 else 0.0

    y_pred = [y_mean + slope * (i - x_mean) for i in range(w)]
    ss_res = sum((window[i] - y_pred[i]) ** 2 for i in range(w))
    ss_tot = sum((window[i] - y_mean) ** 2 for i in range(w))
    r_sq   = 1.0 - ss_res / ss_tot if ss_tot > 0.001 else 1.0
    variance = ss_tot / w

    # Oscillation: alternating diffs
    diffs = [window[i + 1] - window[i] for i in range(w - 1)]
    if len(diffs) >= 4:
        sign_changes = sum(
            1 for i in range(len(diffs) - 1)
            if diffs[i] * diffs[i + 1] < 0
        )
        osc_ratio = sign_changes / max(len(diffs) - 1, 1)
        amplitude = max(window) - min(window)
        if osc_ratio >= 0.70 and amplitude > 0.02:
            return _SeriesClassification(
                label="oscillating",
                slope=slope,
                r_sq=r_sq,
                variance=variance,
                confidence=min(0.90, osc_ratio),
            )

    # Stable: flat + low variance
    if abs(slope) < 0.005 and variance < var_threshold:
        return _SeriesClassification(
            label="stable",
            slope=slope,
            r_sq=r_sq,
            variance=variance,
            confidence=0.90,
        )

    # Strong directional trend
    if r_sq > 0.80 and abs(slope) > 0.008:
        return _SeriesClassification(
            label=("rising" if slope > 0 else "falling"),
            slope=slope,
            r_sq=r_sq,
            variance=variance,
            confidence=min(0.95, r_sq),
        )

    # Weaker but consistent trend after enough rounds
    if r_sq > 0.60 and abs(slope) > 0.005 and n_total >= 15:
        return _SeriesClassification(
            label=("rising" if slope > 0 else "falling"),
            slope=slope,
            r_sq=r_sq,
            variance=variance,
            confidence=r_sq,
        )

    return _SeriesClassification(
        label="unknown",
        slope=slope,
        r_sq=r_sq,
        variance=variance,
        confidence=0.0,
    )


# Composition table: (welfare_label, integrity_label) → combined regime label.
# The combined label is the live runtime regime; classify_regime() produces
# the post-hoc historical regime separately.
_COMBINED_TREND: Dict[tuple[str, str], str] = {
    # collapse / capture variants take precedence
    ("bottomed", "bottomed"):    "collapse",
    ("bottomed", "stable"):      "collapse",
    ("bottomed", "rising"):      "collapse",
    ("bottomed", "falling"):     "collapse",
    ("bottomed", "oscillating"): "collapse",
    ("bottomed", "unknown"):     "collapse",

    ("stable",   "bottomed"):    "captured",     # welfare propped up under captured government
    ("rising",   "bottomed"):    "captured",
    ("falling",  "bottomed"):    "captured_decay",

    # genuine end-game equilibria
    ("stable",  "stable"):  "stable",
    ("rising",  "rising"):  "improving",
    ("falling", "falling"): "decay",

    # the meta-game story: welfare looks fine, integrity is eroding
    ("stable",  "falling"): "capture_in_progress",
    ("rising",  "falling"): "capture_in_progress",

    ("stable",  "rising"):  "recovering",
    ("falling", "rising"):  "decoupling",   # rare: institutions hardening as welfare falls
    ("rising",  "stable"):  "improving",
    ("falling", "stable"):  "decay",

    # oscillation in either dimension
    ("oscillating", "stable"):      "oscillating",
    ("oscillating", "rising"):      "oscillating",
    ("oscillating", "falling"):     "oscillating",
    ("oscillating", "oscillating"): "oscillating",
    ("stable",      "oscillating"): "oscillating",
    ("rising",      "oscillating"): "oscillating",
    ("falling",     "oscillating"): "oscillating",
}


def _compose_trend(welfare: str, integrity: str) -> str:
    """Combine per-series classifications into a single regime label."""
    if welfare == "unknown" or integrity == "unknown":
        return "unknown"
    return _COMBINED_TREND.get((welfare, integrity), "diverging")
