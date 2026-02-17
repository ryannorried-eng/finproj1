"""Shared core math and CLV helpers."""

from .clv import compute_clv_metrics
from .math import (
    american_profit,
    american_to_decimal,
    american_total_return,
    decimal_to_american,
    ev_per_dollar,
    implied_probability,
    kelly_fraction,
    kelly_suggested,
    parlay_payout,
)

__all__ = [
    "american_profit",
    "american_to_decimal",
    "american_total_return",
    "compute_clv_metrics",
    "decimal_to_american",
    "ev_per_dollar",
    "implied_probability",
    "kelly_fraction",
    "kelly_suggested",
    "parlay_payout",
]
