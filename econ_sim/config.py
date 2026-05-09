from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Industry taxonomy
# ---------------------------------------------------------------------------

class Industry(Enum):
    AGRICULTURE = "agriculture"
    MANUFACTURING = "manufacturing"
    HOUSING = "housing"
    TECHNOLOGY = "technology"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class IndustryConfig:
    """Static description of one industry."""
    name: str
    industry: Industry
    product: str                        # human-readable product name
    input_industry: Optional[Industry]  # upstream industry this one depends on (or None)
    input_ratio: float                  # units of upstream input per unit of output
    num_firms: int                      # firms in this industry
    base_price: float                   # starting price agents anchor from
    productivity_factor: float          # gross output units per worker per round
    model_tier: str                     # "opus" | "sonnet" | "haiku"
    is_consumer_good: bool              # do consumers buy this directly?
    is_necessity: bool                  # must workers buy before discretionary?
    necessity_units: float              # units per worker per round needed to avoid suffering


@dataclass
class NecessityRequirement:
    """Describes survival-level consumption requirements for workers."""
    industry: Industry
    min_units_per_round: float      # units needed to avoid suffering
    productivity_penalty: float     # multiplier on productivity if fully unmet (e.g. 0.7)
    suffering_weight: float         # contribution weight to aggregate suffering score (sums to 1)


@dataclass
class WorkerProfile:
    """One representative worker archetype."""
    name: str
    skill_level: float              # multiplier on output and hiring preference
    representative_count: int       # real workers this agent speaks for
    initial_savings: float
    industry_preferences: List[Industry]  # preferred sectors in priority order


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class EconomicConfig:
    """
    Master configuration for one simulation run.

    The three research axes determine the emergent institutional dynamics:

      coordination_mechanism  (0=pure market prices, 1=central planning)
      planning_structure      (0=distributed agencies, 1=single concentrated planner)
      meta_game_constraint    (0=open/unlimited lobbying, 1=fully suppressed)

    These axes mechanically shape agent action spaces, lobbying success
    probabilities, and government constraint levels throughout the simulation.
    """

    # --- Research axes ---
    coordination_mechanism: float = 0.1   # 0.0=pure market  →  1.0=central planning
    planning_structure: float = 0.2       # 0.0=distributed  →  1.0=concentrated
    meta_game_constraint: float = 0.1     # 0.0=open lobby   →  1.0=suppressed

    # --- Simulation scale ---
    num_rounds: int = 60    # hard cap — convergence detection may stop earlier
    min_rounds: int = 15    # never stop before this many rounds (initial-condition burn-in)

    # --- Convergence detection ---
    # The same trend classification (welfare × integrity) must hold for this
    # many consecutive rounds before the simulation stops early. Trades a
    # few extra rounds of cost for much stronger end-game equilibrium claims.
    convergence_persistence_rounds: int = 5

    # --- Structure (populated by __post_init__ if left empty) ---
    industry_configs: Dict[Industry, IndustryConfig] = field(default_factory=dict)
    worker_profiles: List[WorkerProfile] = field(default_factory=list)
    necessity_requirements: List[NecessityRequirement] = field(default_factory=list)

    # --- Initial endowments ---
    initial_firm_capital: float = 1000.0
    initial_government_treasury: float = 500.0
    base_wage_level: float = 50.0

    # --- Institutional integrity dynamics ---
    # Derived from axes in __post_init__; override explicitly if needed.
    initial_institutional_integrity: float = -1.0   # sentinel → derived
    integrity_recovery_rate: float = 0.04            # per-round recovery under high public pressure
    suffering_pressure_threshold: float = 0.35       # aggregate suffering fraction that triggers pressure

    # --- Labor market friction ---
    hiring_cost_factor: float = 0.5    # cost per new hire = this × their wage
    firing_cost_factor: float = 1.0    # cost per layoff  = this × their wage

    # --- LLM settings ---
    temperature: float = 0.7
    max_tokens: int = 1024
    model_opus: str = "claude-opus-4-6"
    model_sonnet: str = "claude-sonnet-4-6"
    model_haiku: str = "claude-haiku-4-5-20251001"

    def __post_init__(self):
        if not self.industry_configs:
            self.industry_configs = _default_industry_configs()
        if not self.worker_profiles:
            self.worker_profiles = _default_worker_profiles()
        if not self.necessity_requirements:
            self.necessity_requirements = _default_necessity_requirements()
        if self.initial_institutional_integrity < 0:
            self.initial_institutional_integrity = self._derive_integrity()

    # --- Derived properties from axes ---

    def _derive_integrity(self) -> float:
        """
        Higher meta_game_constraint → harder to corrupt (higher starting integrity).
        Higher planning_structure   → single high-value target (lower starting integrity).
        """
        raw = 0.30 + 0.50 * self.meta_game_constraint - 0.20 * self.planning_structure
        return max(0.10, min(1.00, raw))

    @property
    def max_lobby_fraction(self) -> float:
        """Max lobby spend as a fraction of firm capital per round."""
        return max(0.0, 0.15 * (1.0 - self.meta_game_constraint))

    @property
    def lobby_detection_prob(self) -> float:
        """Base probability that a lobbying attempt is publicly detected."""
        return 0.05 + 0.75 * self.meta_game_constraint

    @property
    def price_control_strength(self) -> float:
        """How forcefully the government can bind prices (0=none, 1=full)."""
        return self.coordination_mechanism

    @property
    def wage_control_strength(self) -> float:
        """How forcefully the government can bind wages (0=none, 1=full)."""
        return self.coordination_mechanism

    @property
    def total_representative_workers(self) -> int:
        return sum(p.representative_count for p in self.worker_profiles)

    def label(self) -> str:
        """Short human-readable label for this parameter point."""
        c, s, m = (self.coordination_mechanism,
                   self.planning_structure,
                   self.meta_game_constraint)
        if c < 0.25 and m < 0.05:
            return "crony_capitalist"
        if c < 0.25:
            return "capitalist"
        if c >= 0.75 and s >= 0.6:
            return "socialist"
        if c >= 0.45 and m >= 0.55:
            return "nordic"
        return f"mixed_c{c:.2f}_s{s:.2f}_m{m:.2f}"

    def model_for_tier(self, tier: str) -> str:
        return {
            "opus": self.model_opus,
            "sonnet": self.model_sonnet,
            "haiku": self.model_haiku,
        }.get(tier, self.model_opus)


# ---------------------------------------------------------------------------
# Default industry / worker / necessity structures
# ---------------------------------------------------------------------------

def _default_industry_configs() -> Dict[Industry, IndustryConfig]:
    return {
        Industry.AGRICULTURE: IndustryConfig(
            name="Agriculture",
            industry=Industry.AGRICULTURE,
            product="food",
            input_industry=None,
            input_ratio=0.0,
            num_firms=2,
            base_price=8.0,
            productivity_factor=10.0,
            model_tier="haiku",
            is_consumer_good=True,
            is_necessity=True,
            necessity_units=1.0,
        ),
        Industry.MANUFACTURING: IndustryConfig(
            name="Manufacturing",
            industry=Industry.MANUFACTURING,
            product="materials",
            input_industry=None,
            input_ratio=0.0,
            num_firms=2,
            base_price=12.0,
            productivity_factor=8.0,
            model_tier="opus",     # oligopoly power as upstream supplier
            is_consumer_good=False,
            is_necessity=False,
            necessity_units=0.0,
        ),
        Industry.HOUSING: IndustryConfig(
            name="Housing",
            industry=Industry.HOUSING,
            product="shelter",
            input_industry=Industry.MANUFACTURING,
            input_ratio=0.4,       # 0.4 units of materials per unit of shelter
            num_firms=2,
            base_price=25.0,
            productivity_factor=5.0,
            model_tier="sonnet",
            is_consumer_good=True,
            is_necessity=True,
            necessity_units=0.5,
        ),
        Industry.TECHNOLOGY: IndustryConfig(
            name="Technology",
            industry=Industry.TECHNOLOGY,
            product="tech services",
            input_industry=Industry.MANUFACTURING,
            input_ratio=0.3,       # 0.3 units of materials per unit of tech output
            num_firms=2,
            base_price=15.0,
            productivity_factor=6.0,
            model_tier="opus",     # high-margin, strategic
            is_consumer_good=True,
            is_necessity=False,
            necessity_units=0.0,
        ),
    }


def _default_worker_profiles() -> List[WorkerProfile]:
    return [
        WorkerProfile(
            name="low_skill",
            skill_level=0.7,
            representative_count=3,
            initial_savings=60.0,
            industry_preferences=[
                Industry.AGRICULTURE,
                Industry.MANUFACTURING,
                Industry.HOUSING,
                Industry.TECHNOLOGY,
            ],
        ),
        WorkerProfile(
            name="mid_skill",
            skill_level=1.0,
            representative_count=3,
            initial_savings=100.0,
            industry_preferences=[
                Industry.MANUFACTURING,
                Industry.HOUSING,
                Industry.TECHNOLOGY,
                Industry.AGRICULTURE,
            ],
        ),
        WorkerProfile(
            name="high_skill",
            skill_level=1.3,
            representative_count=2,
            initial_savings=150.0,
            industry_preferences=[
                Industry.TECHNOLOGY,
                Industry.MANUFACTURING,
                Industry.HOUSING,
                Industry.AGRICULTURE,
            ],
        ),
        WorkerProfile(
            name="precarious",
            skill_level=0.5,
            representative_count=2,
            initial_savings=30.0,
            industry_preferences=[
                Industry.AGRICULTURE,
                Industry.HOUSING,
                Industry.MANUFACTURING,
                Industry.TECHNOLOGY,
            ],
        ),
    ]


def _default_necessity_requirements() -> List[NecessityRequirement]:
    return [
        NecessityRequirement(
            industry=Industry.AGRICULTURE,
            min_units_per_round=1.0,
            productivity_penalty=0.70,   # 30% productivity loss if no food
            suffering_weight=0.60,        # food deprivation weighs heavily
        ),
        NecessityRequirement(
            industry=Industry.HOUSING,
            min_units_per_round=0.5,
            productivity_penalty=0.85,   # 15% productivity loss if no shelter
            suffering_weight=0.40,
        ),
    ]


# ---------------------------------------------------------------------------
# Named presets — expressed as (coordination, structure, meta_game) tuples
# These produce EconomicConfig instances at common research points.
# ---------------------------------------------------------------------------

def make_config(
    coordination: float,
    structure: float,
    meta_game: float,
    num_rounds: int = 60,
    **kwargs,
) -> EconomicConfig:
    return EconomicConfig(
        coordination_mechanism=coordination,
        planning_structure=structure,
        meta_game_constraint=meta_game,
        num_rounds=num_rounds,
        **kwargs,
    )


# Canonical named presets
CAPITALIST_CONFIG      = make_config(coordination=0.10, structure=0.20, meta_game=0.10)
CRONY_CAPITALIST_CONFIG = make_config(coordination=0.10, structure=0.30, meta_game=0.00)
SOCIALIST_CONFIG       = make_config(coordination=0.85, structure=0.80, meta_game=0.65)
NORDIC_CONFIG          = make_config(coordination=0.50, structure=0.30, meta_game=0.80)
MIXED_CONFIG           = make_config(coordination=0.40, structure=0.40, meta_game=0.45)

SYSTEM_PRESETS: Dict[str, EconomicConfig] = {
    "capitalist":       CAPITALIST_CONFIG,
    "crony":            CRONY_CAPITALIST_CONFIG,
    "socialist":        SOCIALIST_CONFIG,
    "nordic":           NORDIC_CONFIG,
    "mixed":            MIXED_CONFIG,
}
