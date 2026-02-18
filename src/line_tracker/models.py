"""Data models for betting lines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class BetType(Enum):
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"


@dataclass
class BettingLine:
    """A single betting line from a sportsbook."""

    sportsbook: str
    sport: str
    event: str
    bet_type: BetType
    home_team: str
    away_team: str
    home_value: float
    away_value: float
    timestamp: datetime
    home_price: float | None = None
    away_price: float | None = None
    commence_time: datetime | None = None
    api_event_id: str | None = None

    # For moneylines: home_value/away_value are the odds (e.g., -150, +130)
    # For spreads: values are the spread, prices are the juice
    # For totals: home_value is over number, away_value is under number,
    #             home_price/away_price are the juice for over/under


@dataclass
class BestBetResult:
    """Structured best-bet evaluation with explainability data.

    Provides a clean domain model for ranked recommendations, carrying
    both the key metrics and a machine-readable ``explanation`` dict
    that describes *why* this bet was ranked the way it was.
    """

    # ── Core edge / probability metrics ──────────────────────────────
    edge_pct: float  # 100 * (p - 1/d), prob-point edge %
    consensus_prob: float  # vig-free consensus probability
    best_odds_american: float  # best available American odds
    best_odds_decimal: float  # best available decimal odds

    # ── Market coverage ──────────────────────────────────────────────
    books_used: list[str]  # sportsbooks that contributed to consensus
    volatility_sigma: float  # std dev of de-vigged probs
    recency_weight: float  # average recency multiplier across books
    outliers_removed: int  # number of outlier books filtered out

    # ── Explainability ───────────────────────────────────────────────
    explanation: dict[str, Any] = field(default_factory=dict)

    # ── Optional debug payload ───────────────────────────────────────
    raw_inputs: dict[str, Any] | None = None

    # ── Carry-through fields from BetRecommendation ──────────────────
    market: str = ""
    selection: str = ""
    side: str = ""
    line: float | None = None
    confidence: str = ""
    quality_score: int = 0
    quality_tier: str = ""
    ev_roi: float = 0.0
    ev_100: float = 0.0
    edge_z: float = 0.0
    edge_ev: float = 0.0
    edge_ev_shrunk: float = 0.0
    kelly_suggested: float = 0.0
    sizing_note: str = ""
    best_sportsbook: str = ""
    books_used_count: int = 0
    total_books_count: int = 0
    outlier_filtered: bool = False
    market_unstable: bool = False
    market_hold_median: float = 0.0
    robust_sigma: float = 0.0
    n_eff: float = 0.0
    agreement_score: float = 0.0
    skipped_reason: str = ""
    bet_tier: str = ""  # "Tier 1", "Tier 2", "Tier 3", or "Stay Away"
