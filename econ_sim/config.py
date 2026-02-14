from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


class EconomicSystem(Enum):
    CAPITALIST = "capitalist"
    SOCIALIST = "socialist"
    MIXED = "mixed"


@dataclass
class EconomicConfig:
    """Master configuration for an economic simulation run.

    Swap presets (CAPITALIST / SOCIALIST / MIXED) to change
    the rules and constraints agents operate under.
    """

    system: EconomicSystem = EconomicSystem.CAPITALIST
    num_rounds: int = 20
    num_firms: int = 3
    num_workers: int = 5

    # Government policy constraints
    income_tax_rate_range: Tuple[float, float] = (0.0, 0.5)
    corporate_tax_rate_range: Tuple[float, float] = (0.0, 0.5)
    min_wage_enabled: bool = False
    min_wage: float = 30.0
    price_controls_enabled: bool = False
    max_price: float = 20.0

    # Market rules
    allow_free_pricing: bool = True
    allow_free_wages: bool = True

    # Initial conditions
    initial_firm_capital: float = 1000.0
    initial_worker_savings: float = 100.0
    initial_government_treasury: float = 500.0
    base_price_level: float = 10.0
    base_wage_level: float = 50.0

    # Production (units of goods per worker per round)
    productivity_factor: float = 10.0

    # LLM settings
    model: str = "claude-opus-4-6"
    temperature: float = 0.7
    max_tokens: int = 1024


CAPITALIST_CONFIG = EconomicConfig(
    system=EconomicSystem.CAPITALIST,
    income_tax_rate_range=(0.05, 0.25),
    corporate_tax_rate_range=(0.05, 0.20),
    allow_free_pricing=True,
    allow_free_wages=True,
    min_wage_enabled=False,
    price_controls_enabled=False,
)

SOCIALIST_CONFIG = EconomicConfig(
    system=EconomicSystem.SOCIALIST,
    income_tax_rate_range=(0.30, 0.60),
    corporate_tax_rate_range=(0.30, 0.60),
    allow_free_pricing=False,
    allow_free_wages=False,
    min_wage_enabled=True,
    price_controls_enabled=True,
    initial_government_treasury=2000.0,
)

MIXED_CONFIG = EconomicConfig(
    system=EconomicSystem.MIXED,
    income_tax_rate_range=(0.15, 0.40),
    corporate_tax_rate_range=(0.15, 0.35),
    allow_free_pricing=True,
    allow_free_wages=True,
    min_wage_enabled=True,
    price_controls_enabled=False,
)

SYSTEM_PRESETS = {
    "capitalist": CAPITALIST_CONFIG,
    "socialist": SOCIALIST_CONFIG,
    "mixed": MIXED_CONFIG,
}
