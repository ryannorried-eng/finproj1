"""Tests for Pro Mode gating and calibration stats."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from line_tracker.performance import calibration_stats
from line_tracker.slate import (
    _PRO_BOOKS_MIN,
    _PRO_EDGE_MIN,
    _PRO_EDGE_Z_MIN,
    _PRO_HOLD_MAX,
    _TIER1_BASE_EDGE,
    _TIER1_SIGMA_MULT,
    build_daily_slate,
    classify_rec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PRO = {"pro_mode": True}
_STD = {"pro_mode": False}


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
    """Build an entry dict that passes standard AND Pro Mode Tier 1."""
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
# Pro Mode OFF → standard Tier 1 behaviour
# ---------------------------------------------------------------------------


class TestStandardModeUnchanged:
    def test_standard_tier1(self):
        """Standard mode: entry meeting classic criteria → tier1."""
        result = classify_rec(_entry(), settings=_STD)
        assert result["tier"] == "tier1"

    def test_standard_ignores_edge_z(self):
        """Standard mode: edge_z not required for Tier 1."""
        result = classify_rec(_entry(edge_z=0.0), settings=_STD)
        assert result["tier"] == "tier1"

    def test_standard_ignores_books(self):
        """Standard mode: books_used=4 (>= _MIN_BOOKS) still tier1."""
        result = classify_rec(_entry(books_used=4), settings=_STD)
        assert result["tier"] == "tier1"

    def test_standard_ignores_hold(self):
        """Standard mode: high hold does not block Tier 1."""
        result = classify_rec(_entry(market_hold_median=9.0), settings=_STD)
        assert result["tier"] == "tier1"

    def test_none_settings_is_standard(self):
        """settings=None behaves like standard mode."""
        result = classify_rec(_entry())
        assert result["tier"] == "tier1"


# ---------------------------------------------------------------------------
# Pro Mode ON → tighter Tier 1 gates
# ---------------------------------------------------------------------------


class TestProModeGating:
    def test_pro_tier1_passes_all_gates(self):
        """Entry meeting all Pro Mode gates → tier1."""
        result = classify_rec(_entry(), settings=_PRO)
        assert result["tier"] == "tier1"

    def test_pro_rejects_low_edge(self):
        """Edge below _PRO_EDGE_MIN → not tier1 (falls to tier2)."""
        result = classify_rec(
            _entry(edge_pct=3.2), settings=_PRO,
        )
        assert result["tier"] != "tier1"

    def test_pro_rejects_low_edge_z(self):
        """edge_z below _PRO_EDGE_Z_MIN → not tier1."""
        result = classify_rec(
            _entry(edge_z=2.5), settings=_PRO,
        )
        assert result["tier"] != "tier1"

    def test_pro_rejects_zero_edge_z(self):
        """edge_z == 0 (unavailable) → Pro Mode rejects for Tier 1."""
        result = classify_rec(
            _entry(edge_z=0.0), settings=_PRO,
        )
        assert result["tier"] != "tier1"

    def test_pro_rejects_few_books(self):
        """books_used below _PRO_BOOKS_MIN → not tier1."""
        result = classify_rec(
            _entry(books_used=5), settings=_PRO,
        )
        assert result["tier"] != "tier1"

    def test_pro_rejects_high_hold(self):
        """hold_median above _PRO_HOLD_MAX → not tier1."""
        result = classify_rec(
            _entry(market_hold_median=7.0), settings=_PRO,
        )
        assert result["tier"] != "tier1"

    def test_pro_hold_at_threshold_passes(self):
        """hold_median == _PRO_HOLD_MAX → tier1 (<=)."""
        result = classify_rec(
            _entry(market_hold_median=_PRO_HOLD_MAX), settings=_PRO,
        )
        assert result["tier"] == "tier1"

    def test_pro_edge_at_threshold_passes(self):
        """edge_pct == _PRO_EDGE_MIN → tier1 (>=)."""
        result = classify_rec(
            _entry(edge_pct=_PRO_EDGE_MIN), settings=_PRO,
        )
        assert result["tier"] == "tier1"

    def test_pro_edge_z_at_threshold_passes(self):
        """edge_z == _PRO_EDGE_Z_MIN → tier1 (>=)."""
        result = classify_rec(
            _entry(edge_z=_PRO_EDGE_Z_MIN), settings=_PRO,
        )
        assert result["tier"] == "tier1"

    def test_pro_books_at_threshold_passes(self):
        """books_used == _PRO_BOOKS_MIN → tier1 (>=)."""
        result = classify_rec(
            _entry(books_used=_PRO_BOOKS_MIN), settings=_PRO,
        )
        assert result["tier"] == "tier1"


# ---------------------------------------------------------------------------
# Pro Mode still respects dynamic floor
# ---------------------------------------------------------------------------


class TestProModeDynamicFloor:
    def test_pro_uses_dynamic_floor_when_higher(self):
        """When dyn_floor > _PRO_EDGE_MIN, Pro Mode uses dyn_floor."""
        sigma = 1.0  # floor = 3.0 + 1.2*1.0 = 4.2 > 3.5
        floor = _TIER1_BASE_EDGE + _TIER1_SIGMA_MULT * sigma
        # Edge between _PRO_EDGE_MIN and dyn_floor → not tier1
        result = classify_rec(
            _entry(edge_pct=4.0, market_volatility_sigma=sigma),
            settings=_PRO,
        )
        assert result["tier"] != "tier1"
        assert result["dynamic_edge_floor"] == floor

    def test_pro_uses_pro_edge_when_higher(self):
        """When _PRO_EDGE_MIN > dyn_floor, Pro Mode uses _PRO_EDGE_MIN."""
        # sigma=0 → dyn_floor = 3.0 < 3.5
        result = classify_rec(
            _entry(edge_pct=3.3, market_volatility_sigma=0.0),
            settings=_PRO,
        )
        # 3.3 >= dyn_floor(3.0) but < _PRO_EDGE_MIN(3.5) → not tier1
        assert result["tier"] != "tier1"


# ---------------------------------------------------------------------------
# Pro Mode does NOT affect Tier 2 or Stay Away classification
# ---------------------------------------------------------------------------


class TestProModeDoesNotAffectTier2:
    def test_tier2_unchanged_in_pro_mode(self):
        """Tier 2 criteria are identical regardless of Pro Mode."""
        e = _entry(
            edge_pct=2.0, confidence="Medium", quality_tier="Moderate",
            edge_z=1.5, books_used=5, market_hold_median=8.0,
        )
        std = classify_rec(e, settings=_STD)
        pro = classify_rec(e, settings=_PRO)
        assert std["tier"] == "tier2"
        assert pro["tier"] == "tier2"

    def test_avoid_unchanged_in_pro_mode(self):
        """Stay Away criteria are identical regardless of Pro Mode."""
        e = _entry(
            edge_pct=0.5, confidence="Low", quality_tier="Thin",
        )
        std = classify_rec(e, settings=_STD)
        pro = classify_rec(e, settings=_PRO)
        assert std["tier"] == "avoid"
        assert pro["tier"] == "avoid"
        assert std["reasons"] == pro["reasons"]


# ---------------------------------------------------------------------------
# Pro Mode demotes standard-Tier1 entries to Tier 2
# ---------------------------------------------------------------------------


class TestProModeDemotion:
    def test_standard_tier1_demoted_to_tier2_by_pro(self):
        """Entry that passes standard Tier 1 but fails Pro gates → tier2."""
        # edge=3.2 passes standard (>= 3.0) but fails Pro (< 3.5)
        e = _entry(edge_pct=3.2, edge_z=1.5, books_used=5)
        std = classify_rec(e, settings=_STD)
        pro = classify_rec(e, settings=_PRO)
        assert std["tier"] == "tier1"
        assert pro["tier"] == "tier2"

    def test_standard_tier1_demoted_by_edge_z(self):
        """edge_z=2.0 passes standard but fails Pro → tier2."""
        e = _entry(edge_z=2.0)
        std = classify_rec(e, settings=_STD)
        pro = classify_rec(e, settings=_PRO)
        assert std["tier"] == "tier1"
        assert pro["tier"] == "tier2"

    def test_standard_tier1_demoted_by_hold(self):
        """hold=7% passes standard but fails Pro → tier2."""
        e = _entry(market_hold_median=7.0)
        std = classify_rec(e, settings=_STD)
        pro = classify_rec(e, settings=_PRO)
        assert std["tier"] == "tier1"
        assert pro["tier"] == "tier2"


# ---------------------------------------------------------------------------
# build_daily_slate with Pro Mode
# ---------------------------------------------------------------------------


class TestBuildSlateProMode:
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
    def test_pro_mode_off_tier1(self, mock_rbb):
        """Standard mode: qualifying rec → tier1."""
        mock_rbb.return_value = [self._make_rec()]
        result = build_daily_slate(
            {"e1": self._lines()}, settings={"pro_mode": False},
        )
        assert len(result["tier1"]) == 1

    @patch("line_tracker.slate.recommend_best_bets")
    def test_pro_mode_on_fewer_tier1(self, mock_rbb):
        """Pro Mode: rec with edge_z=1.5 fails Pro gates → not tier1."""
        mock_rbb.return_value = [self._make_rec(edge_z=1.5)]
        result = build_daily_slate(
            {"e1": self._lines()}, settings={"pro_mode": True},
        )
        assert len(result["tier1"]) == 0
        # Should fall to tier2 instead
        assert len(result["tier2"]) == 1

    @patch("line_tracker.slate.recommend_best_bets")
    def test_pro_mode_on_still_tier1_when_strong(self, mock_rbb):
        """Pro Mode: rec meeting all Pro gates → still tier1."""
        mock_rbb.return_value = [self._make_rec(
            edge_pct=4.0, edge_z=3.0, books_used_count=7,
            market_hold_median=4.0,
        )]
        result = build_daily_slate(
            {"e1": self._lines()}, settings={"pro_mode": True},
        )
        assert len(result["tier1"]) == 1


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
