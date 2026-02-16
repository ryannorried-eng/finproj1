"""Tests for Tier 1A / 1B gating and calibration stats."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from line_tracker.performance import calibration_stats
from line_tracker.slate import (
    _TIER1A_BOOKS_MIN,
    _TIER1A_HOLD_MAX,
    _TIER1B_BOOKS_MIN,
    _TIER1B_HOLD_MAX,
    build_daily_slate,
    classify_rec,
    dyn_floor_1a,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    *,
    edge_pct: float = 4.0,
    confidence: str = "High",
    quality_tier: str = "Strong",
    quality_score: int = 85,
    market_volatility_sigma: float = 0.0,
    edge_z: float = 3.0,
    market_unstable: bool = False,
    books_used: int = 7,
    oldest_update_age_min: float = 15.0,
    market_hold_median: float = 4.0,
    divergence: float | None = None,
    market: str = "moneyline",
) -> dict:
    """Build an entry dict that passes Tier 1A by default."""
    return {
        "edge_pct": edge_pct,
        "confidence": confidence,
        "quality_tier": quality_tier,
        "quality_score": quality_score,
        "market_volatility_sigma": market_volatility_sigma,
        "edge_z": edge_z,
        "market_unstable": market_unstable,
        "books_used": books_used,
        "oldest_update_age_min": oldest_update_age_min,
        "market_hold_median": market_hold_median,
        "divergence": divergence,
        "market": market,
    }


# ---------------------------------------------------------------------------
# Tier 1A gating
# ---------------------------------------------------------------------------


class TestTier1AGating:
    def test_tier1a_passes_all_gates(self):
        """Entry meeting all Tier 1A gates → tier1a."""
        result = classify_rec(_entry())
        assert result["tier"] == "tier1a"

    def test_tier1a_rejects_medium_confidence(self):
        """Medium confidence fails Tier 1A → falls to 1B."""
        result = classify_rec(_entry(confidence="Medium"))
        assert result["tier"] == "tier1b"

    def test_tier1a_rejects_moderate_quality(self):
        """Moderate quality fails Tier 1A → falls to 1B."""
        result = classify_rec(_entry(quality_tier="Moderate"))
        assert result["tier"] == "tier1b"

    def test_tier1a_rejects_few_books(self):
        """books_used < 6 fails Tier 1A."""
        result = classify_rec(_entry(books_used=5))
        assert result["tier"] != "tier1a"

    def test_tier1a_rejects_high_hold(self):
        """hold > 6% fails Tier 1A."""
        result = classify_rec(_entry(market_hold_median=6.5))
        assert result["tier"] != "tier1a"

    def test_tier1a_hold_at_threshold_passes(self):
        """hold == 6% passes Tier 1A (<=)."""
        result = classify_rec(_entry(market_hold_median=_TIER1A_HOLD_MAX))
        assert result["tier"] == "tier1a"

    def test_tier1a_books_at_threshold_passes(self):
        """books == 6 passes Tier 1A (>=)."""
        result = classify_rec(_entry(books_used=_TIER1A_BOOKS_MIN))
        assert result["tier"] == "tier1a"

    def test_tier1a_edge_below_floor(self):
        """Edge below dyn_floor_1a fails Tier 1A."""
        result = classify_rec(_entry(edge_pct=2.0))
        assert result["tier"] != "tier1a"


# ---------------------------------------------------------------------------
# Tier 1B gating
# ---------------------------------------------------------------------------


class TestTier1BGating:
    def test_tier1b_medium_confidence(self):
        """Medium conf + Strong quality + books=5 → tier1b."""
        result = classify_rec(_entry(
            confidence="Medium", books_used=5,
        ))
        assert result["tier"] == "tier1b"

    def test_tier1b_moderate_quality(self):
        """High conf + Moderate quality + books=5 → tier1b."""
        result = classify_rec(_entry(
            quality_tier="Moderate", books_used=5,
        ))
        assert result["tier"] == "tier1b"

    def test_tier1b_rejects_low_confidence(self):
        """Low confidence fails Tier 1B."""
        result = classify_rec(_entry(confidence="Low"))
        assert result["tier"] not in ("tier1a", "tier1b")

    def test_tier1b_rejects_thin_quality(self):
        """Thin quality fails Tier 1B."""
        result = classify_rec(_entry(quality_tier="Thin"))
        assert result["tier"] not in ("tier1a", "tier1b")

    def test_tier1b_hold_at_threshold(self):
        """hold == 7.5% passes Tier 1B (<=)."""
        result = classify_rec(_entry(
            market_hold_median=_TIER1B_HOLD_MAX, books_used=5,
            confidence="Medium", quality_tier="Moderate",
        ))
        assert result["tier"] == "tier1b"

    def test_tier1b_hold_above_threshold(self):
        """hold > 7.5% fails Tier 1B → falls to tier2."""
        result = classify_rec(_entry(
            market_hold_median=7.6, books_used=5,
            confidence="Medium", quality_tier="Moderate",
        ))
        assert result["tier"] not in ("tier1a", "tier1b")

    def test_tier1b_books_at_threshold(self):
        """books == 5 passes Tier 1B (>=)."""
        result = classify_rec(_entry(
            books_used=_TIER1B_BOOKS_MIN,
            confidence="Medium", quality_tier="Moderate",
        ))
        assert result["tier"] == "tier1b"

    def test_tier1b_books_below_threshold(self):
        """books < 5 fails Tier 1B."""
        result = classify_rec(_entry(
            books_used=4, confidence="Medium", quality_tier="Moderate",
        ))
        assert result["tier"] not in ("tier1a", "tier1b")


# ---------------------------------------------------------------------------
# Tier 1A dynamic floor
# ---------------------------------------------------------------------------


class TestTier1ADynamicFloor:
    def test_sigma_raises_floor(self):
        """sigma=1.0 → floor_1a = 2.5 + 1.0*1.0 = 3.5; edge=3.0 < 3.5."""
        result = classify_rec(_entry(
            edge_pct=3.0, market_volatility_sigma=1.0,
        ))
        assert result["tier"] != "tier1a"
        assert result["dynamic_edge_floor"] == dyn_floor_1a(1.0)

    def test_edge_above_raised_floor(self):
        """sigma=1.0 → floor_1a = 3.5; edge=3.6 >= 3.5 → tier1a."""
        result = classify_rec(_entry(
            edge_pct=3.6, market_volatility_sigma=1.0,
        ))
        assert result["tier"] == "tier1a"


# ---------------------------------------------------------------------------
# Demotion from Tier 1A to Tier 1B
# ---------------------------------------------------------------------------


class TestTier1ADemotion:
    def test_demoted_by_books(self):
        """books=5 fails 1A (needs 6) but passes 1B → tier1b."""
        result = classify_rec(_entry(books_used=5))
        assert result["tier"] == "tier1b"

    def test_demoted_by_hold(self):
        """hold=7.0 fails 1A (needs <=6) but passes 1B (<=7.5) → tier1b."""
        result = classify_rec(_entry(market_hold_median=7.0))
        assert result["tier"] == "tier1b"


# ---------------------------------------------------------------------------
# build_daily_slate tier1a / tier1b buckets
# ---------------------------------------------------------------------------


class TestBuildSlateTiers:
    def _make_rec(self, **kwargs):
        from line_tracker.best_bets import BetRecommendation

        defaults = dict(
            market="moneyline", selection="Home", side="home",
            line=None, consensus_prob=0.55, best_sportsbook="FanDuel",
            best_odds=-110.0, breakeven_prob=0.52, ev=0.05,
            edge_pct=4.0, ev_per_100=5.0, confidence="High",
            quality_score=85, quality_tier="Strong",
            books_used_count=7, newest_update_age_min=10.0,
            oldest_update_age_min=15.0, market_unstable=False,
            market_volatility_sigma=0.0, market_hold_median=4.0,
            edge_z=3.0,
        )
        defaults.update(kwargs)
        return BetRecommendation(**defaults)

    def _lines(self, event="A @ B"):
        from datetime import datetime

        from line_tracker.models import BettingLine, BetType

        return [
            BettingLine(
                sportsbook="FanDuel", sport="basketball_nba",
                event=event, bet_type=BetType.MONEYLINE,
                home_team="B", away_team="A",
                home_value=-150, away_value=130,
                timestamp=datetime(2026, 2, 1),
            ),
            BettingLine(
                sportsbook="DraftKings", sport="basketball_nba",
                event=event, bet_type=BetType.MONEYLINE,
                home_team="B", away_team="A",
                home_value=-145, away_value=125,
                timestamp=datetime(2026, 2, 1),
            ),
        ]

    @patch("line_tracker.slate.recommend_best_bets")
    def test_tier1a_in_slate(self, mock_rbb):
        """Strong rec → tier1a in slate, and in combined tier1."""
        mock_rbb.return_value = [self._make_rec()]
        result = build_daily_slate({"e1": self._lines()})
        assert len(result["tier1a"]) == 1
        assert len(result["tier1"]) == 1

    @patch("line_tracker.slate.recommend_best_bets")
    def test_tier1b_in_slate(self, mock_rbb):
        """5-book rec → tier1b, not tier1a."""
        mock_rbb.return_value = [self._make_rec(books_used_count=5)]
        result = build_daily_slate({"e1": self._lines()})
        assert len(result["tier1a"]) == 0
        assert len(result["tier1b"]) == 1
        assert len(result["tier1"]) == 1  # combined

    @patch("line_tracker.slate.recommend_best_bets")
    def test_counts_tier1a_tier1b(self, mock_rbb):
        """Counts include tier1a, tier1b, and combined tier1."""
        mock_rbb.return_value = [self._make_rec()]
        result = build_daily_slate({"e1": self._lines()})
        assert result["counts"]["tier1a"] == 1
        assert result["counts"]["tier1b"] == 0
        assert result["counts"]["tier1"] == 1


# ---------------------------------------------------------------------------
# calibration_stats
# ---------------------------------------------------------------------------


class TestCalibrationStats:
    def _make_df(self, rows):
        df = pd.DataFrame(rows)
        df["clv_decimal"] = 0.05
        df["clv_prob"] = df.get("clv_prob", 0.01)
        return df

    def test_empty_df(self):
        assert calibration_stats(pd.DataFrame()) == {}

    def test_missing_columns(self):
        df = pd.DataFrame({"foo": [1, 2]})
        assert calibration_stats(df) == {}

    def test_tier1_proxy(self):
        """High confidence + Strong quality + edge >= 3 → Tier 1 proxy."""
        df = self._make_df([
            {"confidence_at_pick": "High", "quality_tier_at_pick": "Strong",
             "edge_pct_at_pick": 4.0, "clv_prob": 0.02},
            {"confidence_at_pick": "High", "quality_tier_at_pick": "Elite",
             "edge_pct_at_pick": 3.5, "clv_prob": 0.03},
        ])
        cal = calibration_stats(df)
        assert cal["Tier 1"]["legs"] == 2
        assert cal["Tier 1"]["beating_pct"] == 100.0

    def test_tier2_proxy(self):
        """Medium confidence + Moderate quality + edge >= 1.5 → Tier 2."""
        df = self._make_df([
            {"confidence_at_pick": "Medium", "quality_tier_at_pick": "Moderate",
             "edge_pct_at_pick": 2.0, "clv_prob": 0.01},
        ])
        cal = calibration_stats(df)
        assert cal["Tier 2"]["legs"] == 1

    def test_stay_away_proxy(self):
        """Low confidence + Thin quality → Stay Away."""
        df = self._make_df([
            {"confidence_at_pick": "Low", "quality_tier_at_pick": "Thin",
             "edge_pct_at_pick": 0.5, "clv_prob": -0.02},
        ])
        cal = calibration_stats(df)
        assert cal["Stay Away"]["legs"] == 1
        assert cal["Stay Away"]["beating_pct"] == 0.0

    def test_mixed_tiers(self):
        """All three tiers represented with correct grouping."""
        df = self._make_df([
            {"confidence_at_pick": "High", "quality_tier_at_pick": "Strong",
             "edge_pct_at_pick": 4.0, "clv_prob": 0.03},
            {"confidence_at_pick": "Medium", "quality_tier_at_pick": "Moderate",
             "edge_pct_at_pick": 2.0, "clv_prob": 0.01},
            {"confidence_at_pick": "Low", "quality_tier_at_pick": "Thin",
             "edge_pct_at_pick": 0.5, "clv_prob": -0.01},
        ])
        cal = calibration_stats(df)
        assert cal["Tier 1"]["legs"] == 1
        assert cal["Tier 2"]["legs"] == 1
        assert cal["Stay Away"]["legs"] == 1

    def test_tier1_avg_clv_computed(self):
        """avg_clv_prob is correctly computed for Tier 1 proxy."""
        df = self._make_df([
            {"confidence_at_pick": "High", "quality_tier_at_pick": "Strong",
             "edge_pct_at_pick": 4.0, "clv_prob": 0.02},
            {"confidence_at_pick": "High", "quality_tier_at_pick": "Elite",
             "edge_pct_at_pick": 3.5, "clv_prob": 0.04},
        ])
        cal = calibration_stats(df)
        assert cal["Tier 1"]["avg_clv_prob"] == 0.03  # mean of 0.02, 0.04

    def test_empty_tier_gets_zeros(self):
        """Tiers with no legs get zero-filled stats."""
        df = self._make_df([
            {"confidence_at_pick": "High", "quality_tier_at_pick": "Strong",
             "edge_pct_at_pick": 4.0, "clv_prob": 0.02},
        ])
        cal = calibration_stats(df)
        assert cal["Stay Away"]["legs"] == 0
        assert cal["Stay Away"]["avg_clv_prob"] == 0.0

    def test_edge_at_tier1_boundary(self):
        """edge_pct_at_pick exactly at 3.0 → Tier 1 proxy."""
        df = self._make_df([
            {"confidence_at_pick": "High", "quality_tier_at_pick": "Strong",
             "edge_pct_at_pick": 3.0, "clv_prob": 0.01},
        ])
        cal = calibration_stats(df)
        assert cal["Tier 1"]["legs"] == 1

    def test_edge_below_tier1_boundary(self):
        """edge_pct_at_pick=2.9, High/Strong → Tier 2 proxy (not Tier 1)."""
        df = self._make_df([
            {"confidence_at_pick": "High", "quality_tier_at_pick": "Strong",
             "edge_pct_at_pick": 2.9, "clv_prob": 0.01},
        ])
        cal = calibration_stats(df)
        assert cal["Tier 2"]["legs"] == 1
        assert cal["Tier 1"]["legs"] == 0

    def test_none_edge_treated_as_zero(self):
        """None edge_pct_at_pick is treated as 0 → Stay Away."""
        df = self._make_df([
            {"confidence_at_pick": "High", "quality_tier_at_pick": "Strong",
             "edge_pct_at_pick": None, "clv_prob": 0.01},
        ])
        cal = calibration_stats(df)
        assert cal["Stay Away"]["legs"] == 1
