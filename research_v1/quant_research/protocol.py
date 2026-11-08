"""Pre-registered research scope and chronological split boundaries."""

from dataclasses import dataclass
from datetime import date

ETF_SYMBOLS = ("510050", "510300", "510500", "510880")
CAPITALS = (10_000, 100_000)
HORIZONS = (5, 20)
RESEARCH_START = date(2015, 1, 1)
VALIDATION_START = date(2023, 1, 1)
HOLDOUT_START = date(2025, 1, 1)
LAST_EXPLORED_SESSION = date(2026, 11, 6)
MAX_DRAWDOWN = 0.15
RESEARCH_LINES = tuple(f"{kind}-{h}d" for kind in ("etf", "stock") for h in HORIZONS)


@dataclass(frozen=True)
class ResearchPeriod:
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("end precedes start")
