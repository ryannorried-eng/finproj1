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
    PRO_THRESHOLDS,
    STANDARD_THRESHOLDS,
    TierThresholds,
    _percentiles,
    build_daily_slate,
    classify_rec,
    compute_volume_tuning_stats,
    dyn_floor_1a,
    get_thresholds,
    print_slate_summary,
    suggest_thresholds,
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
        result = classify_rec(_entry(edge_pct=1.9))
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
        """Low conf fails 1B (edge/edge_z below override)."""
        result = classify_rec(_entry(confidence="Low", edge_pct=1.9, edge_z=1.9))
        assert result["tier"] not in ("tier1a", "tier1b")

    def test_tier1b_thin_quality_override(self):
        """Thin quality passes 1B when override conditions are met."""
        # Default: edge=4.0, hold=4.0, books=7 → all override gates met
        result = classify_rec(_entry(quality_tier="Thin"))
        assert result["tier"] == "tier1b"

    def test_tier1b_rejects_thin_quality_few_books(self):
        """Thin quality fails 1B override when books < 6."""
        result = classify_rec(_entry(quality_tier="Thin", books_used=5))
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
        """sigma=0.02 → floor=max(2.0,2.0)=2.0; 2.1>=2.0 → 1a."""
        result = classify_rec(_entry(
            edge_pct=2.1, market_volatility_sigma=0.02,
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
            edge_pct=4.0, ev_per_100=4.0, confidence="High",
            ev_100=4.0, ev_roi=0.04, p_be=0.52, edge_pp=0.03,
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


# ---------------------------------------------------------------------------
# TierThresholds parameter pack
# ---------------------------------------------------------------------------


class TestTierThresholds:
    def test_standard_defaults(self):
        """STANDARD_THRESHOLDS has expected default values."""
        th = STANDARD_THRESHOLDS
        assert th.mode == "Standard"
        assert th.tier1b_base_edge == 0.5
        assert th.tier1b_sigma_mult == 100.0
        assert th.tier1b_floor_min == 1.0
        assert th.tier2_edge_min == 0.0
        assert th.tier2_edge_z_min == 0.0

    def test_pro_thresholds(self):
        """PRO_THRESHOLDS has stricter values."""
        th = PRO_THRESHOLDS
        assert th.mode == "Pro"
        assert th.tier1b_base_edge == 1.0
        assert th.tier1b_floor_min == 1.5
        assert th.tier2_edge_min == 0.5
        assert th.tier2_edge_z_min == 1.0

    def test_get_thresholds_standard(self):
        assert get_thresholds("Standard") is STANDARD_THRESHOLDS

    def test_get_thresholds_pro(self):
        assert get_thresholds("Pro") is PRO_THRESHOLDS

    def test_frozen(self):
        """TierThresholds is immutable."""
        import dataclasses
        assert dataclasses.fields(TierThresholds)
        try:
            STANDARD_THRESHOLDS.mode = "Bad"  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")
        except dataclasses.FrozenInstanceError:
            pass

    def test_classify_with_pro_thresholds(self):
        """Pro thresholds disable Low-confidence override for Tier 1B."""
        # Low conf + high edge_z → tier1b under Standard, but avoid under Pro
        # (because edge=4.0 >= outlier threshold with Low conf → hard avoid)
        e = _entry(
            confidence="Low", edge_pct=3.5, edge_z=3.0,
            books_used=7, market_hold_median=4.0,
        )
        std = classify_rec(e, thresholds=STANDARD_THRESHOLDS)
        pro = classify_rec(e, thresholds=PRO_THRESHOLDS)
        assert std["tier"] == "tier1b"
        assert pro["tier"] == "tier3"  # Low conf, no override in Pro


# ---------------------------------------------------------------------------
# Low-confidence override for Tier 1B
# ---------------------------------------------------------------------------


class TestTier1BLowConfOverride:
    def test_low_conf_with_high_edge_z_passes(self):
        """Low conf + edge_z >= 2.0 + edge >= 2.5 → tier1b."""
        result = classify_rec(_entry(
            confidence="Low", edge_pct=2.5, edge_z=2.0,
            quality_tier="Strong", books_used=5,
        ))
        assert result["tier"] == "tier1b"

    def test_low_conf_insufficient_edge_z(self):
        """Low conf + edge_z < 2.0 → fails 1B override → falls to tier2
        (T2 low-conf override: edge_z=1.9 >= 1.5 and edge=3.0 >= 1.0)."""
        result = classify_rec(_entry(
            confidence="Low", edge_pct=3.0, edge_z=1.9,
            quality_tier="Strong", books_used=5,
        ))
        assert result["tier"] != "tier1b"
        assert result["tier"] == "tier2"

    def test_low_conf_insufficient_edge_pct(self):
        """Low conf + edge_pct < 2.0 → fails 1B override → falls to tier2
        (T2 low-conf override: edge_z=3.0 >= 1.5 and edge=1.9 >= 0.5)."""
        result = classify_rec(_entry(
            confidence="Low", edge_pct=1.9, edge_z=3.0,
            quality_tier="Strong", books_used=5,
        ))
        assert result["tier"] != "tier1b"
        assert result["tier"] == "tier2"

    def test_low_conf_edge_z_zero_means_unavailable(self):
        """Low conf + edge_z=0 (unavailable) → no override → tier3."""
        result = classify_rec(_entry(
            confidence="Low", edge_pct=3.0, edge_z=0.0,
            quality_tier="Strong", books_used=5,
        ))
        assert result["tier"] == "tier3"

    def test_low_conf_override_at_boundary(self):
        """Exactly at threshold values → passes."""
        result = classify_rec(_entry(
            confidence="Low", edge_pct=2.5, edge_z=2.0,
            quality_tier="Moderate", books_used=5,
        ))
        assert result["tier"] == "tier1b"


# ---------------------------------------------------------------------------
# Thin-quality override for Tier 1B
# ---------------------------------------------------------------------------


class TestTier1BThinOverride:
    def test_thin_quality_with_override_passes(self):
        """Thin + edge>=3.0 + hold<=6.5 + books>=6 → tier1b."""
        result = classify_rec(_entry(
            quality_tier="Thin", edge_pct=3.0, market_hold_median=6.5,
            books_used=6, confidence="High",
        ))
        assert result["tier"] == "tier1b"

    def test_thin_quality_edge_too_low(self):
        """Thin + edge<3.0 → fails override."""
        result = classify_rec(_entry(
            quality_tier="Thin", edge_pct=2.9, market_hold_median=4.0,
            books_used=7, confidence="High",
        ))
        assert result["tier"] == "tier3"

    def test_thin_quality_hold_too_high(self):
        """Thin + hold>6.5 → fails override."""
        result = classify_rec(_entry(
            quality_tier="Thin", edge_pct=3.5, market_hold_median=6.6,
            books_used=7, confidence="High",
        ))
        assert result["tier"] == "tier3"

    def test_thin_quality_books_too_few(self):
        """Thin + books<6 → fails override."""
        result = classify_rec(_entry(
            quality_tier="Thin", edge_pct=3.5, market_hold_median=4.0,
            books_used=5, confidence="High",
        ))
        assert result["tier"] == "tier3"


# ---------------------------------------------------------------------------
# Tier 2 Low-confidence override
# ---------------------------------------------------------------------------


class TestTier2LowConfOverride:
    def test_low_conf_with_edge_z_override(self):
        """Low conf + edge_z >= 1.5 + edge >= 1.0 → tier2."""
        result = classify_rec(_entry(
            confidence="Low", edge_pct=1.0, edge_z=1.5,
            quality_tier="Strong", books_used=5,
        ))
        assert result["tier"] == "tier2"

    def test_low_conf_insufficient_edge_z_for_tier2(self):
        """Low conf + edge_z < 1.5 → no override → tier3."""
        result = classify_rec(_entry(
            confidence="Low", edge_pct=1.0, edge_z=1.4,
            quality_tier="Strong", books_used=5,
        ))
        assert result["tier"] == "tier3"

    def test_low_conf_insufficient_edge_for_tier2(self):
        """Low conf + edge < 0.5 → no override → tier3."""
        result = classify_rec(_entry(
            confidence="Low", edge_pct=0.4, edge_z=2.0,
            quality_tier="Strong", books_used=5,
        ))
        assert result["tier"] == "tier3"

    def test_low_conf_tier2_at_boundary(self):
        """Low conf at exact threshold boundaries → tier2."""
        result = classify_rec(_entry(
            confidence="Low", edge_pct=1.0, edge_z=1.5,
            quality_tier="Moderate", books_used=4,
        ))
        assert result["tier"] == "tier2"


# ---------------------------------------------------------------------------
# Stay Away: only explicit hard conditions
# ---------------------------------------------------------------------------


class TestStayAwayExplicitOnly:
    def test_stay_away_edge_zero(self):
        """edge=0 → stay away."""
        result = classify_rec(_entry(edge_pct=0.0))
        assert result["tier"] == "avoid"

    def test_stay_away_edge_negative(self):
        """Negative edge → stay away."""
        result = classify_rec(_entry(edge_pct=-1.0))
        assert result["tier"] == "avoid"

    def test_stay_away_books_too_few(self):
        """books < 4 → stay away."""
        result = classify_rec(_entry(books_used=3))
        assert result["tier"] == "avoid"

    def test_stay_away_hold_too_high(self):
        """hold >= 8 → stay away."""
        result = classify_rec(_entry(market_hold_median=8.0))
        assert result["tier"] == "avoid"

    def test_stay_away_unstable(self):
        """market_unstable → stay away."""
        result = classify_rec(_entry(market_unstable=True))
        assert result["tier"] == "avoid"

    def test_stay_away_stale(self):
        """oldest_update_age_min > 120 → stay away."""
        result = classify_rec(_entry(oldest_update_age_min=121.0))
        assert result["tier"] == "avoid"

    def test_stay_away_edge_outlier_low_conf(self):
        """edge >= 8.0 + Low conf → stay away."""
        result = classify_rec(_entry(edge_pct=8.0, confidence="Low"))
        assert result["tier"] == "avoid"

    def test_positive_edge_never_stay_away(self):
        """Any positive-edge entry without hard flags is NOT stay away."""
        # Low/Thin/low books - still NOT stay away if edge > 0 and books >= 4
        result = classify_rec(_entry(
            edge_pct=0.1, confidence="Low", quality_tier="Thin",
            books_used=4, market_hold_median=7.9,
        ))
        assert result["tier"] != "avoid"


# ---------------------------------------------------------------------------
# suggest_thresholds
# ---------------------------------------------------------------------------


class TestSuggestThresholds:
    def test_no_relaxation_when_tier1a_populated(self):
        """suggest_thresholds returns base when tier1a has entries."""
        stats = {"counts_by_tier": {"tier1a": 1, "tier1b": 3}}
        result = suggest_thresholds(stats)
        assert result is STANDARD_THRESHOLDS

    def test_relaxes_when_tier1a_empty(self):
        """suggest_thresholds relaxes all 4 Tier 1A params when empty."""
        stats = {"counts_by_tier": {"tier1b": 3, "tier2": 5}}
        result = suggest_thresholds(stats)
        assert result.tier1a_books_min == 5  # 6 - 1
        assert result.tier1a_hold_max == 6.5  # 6.0 + 0.5
        assert result.tier1a_base_edge == 1.75  # 2.0 - 0.25
        assert result.tier1a_sigma_mult == 90.0  # 100.0 - 10.0

    def test_floors_respected(self):
        """Relaxation floors prevent excessive loosening."""
        # Start with already-relaxed thresholds at their floors
        base = TierThresholds(
            tier1a_books_min=5,
            tier1a_hold_max=7.0,
            tier1a_base_edge=1.5,
            tier1a_sigma_mult=50.0,
        )
        stats = {"counts_by_tier": {"tier1b": 1}}
        result = suggest_thresholds(stats, base=base)
        assert result.tier1a_books_min == 5  # floor 5
        assert result.tier1a_hold_max == 7.0  # cap 7.0
        assert result.tier1a_base_edge == 1.5  # floor 1.5
        assert result.tier1a_sigma_mult == 50.0  # floor 50.0

    def test_tier1b_unchanged(self):
        """suggest_thresholds only relaxes Tier 1A, not 1B."""
        stats = {"counts_by_tier": {}}
        result = suggest_thresholds(stats)
        assert result.tier1b_books_min == STANDARD_THRESHOLDS.tier1b_books_min
        assert result.tier1b_hold_max == STANDARD_THRESHOLDS.tier1b_hold_max


# ---------------------------------------------------------------------------
# compute_volume_tuning_stats
# ---------------------------------------------------------------------------


class TestVolumetuningStats:
    def test_empty_entries(self):
        result = compute_volume_tuning_stats([])
        assert result == {"total": 0}

    def test_basic_stats(self):
        entries = [
            {**_entry(edge_pct=2.0, books_used=5), "tier": "tier1b"},
            {**_entry(edge_pct=4.0, books_used=7), "tier": "tier1a"},
        ]
        result = compute_volume_tuning_stats(entries)
        assert result["total"] == 2
        assert result["counts_by_tier"]["tier1b"] == 1
        assert result["counts_by_tier"]["tier1a"] == 1
        assert result["edge_pct"]["p50"] is not None

    def test_books_histogram(self):
        entries = [
            {**_entry(books_used=2), "tier": "avoid"},
            {**_entry(books_used=5), "tier": "tier1b"},
            {**_entry(books_used=7), "tier": "tier1a"},
            {**_entry(books_used=10), "tier": "tier1a"},
        ]
        result = compute_volume_tuning_stats(entries)
        assert result["books_histogram"]["1-3"] == 1
        assert result["books_histogram"]["4-5"] == 1
        assert result["books_histogram"]["6-7"] == 1
        assert result["books_histogram"]["8+"] == 1

    def test_percentiles(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = _percentiles(vals)
        assert result["p10"] == 2.0
        assert result["p50"] == 6.0
        assert result["p90"] == 10.0

    def test_percentiles_empty(self):
        result = _percentiles([])
        assert result["p10"] is None
        assert result["p50"] is None
        assert result["p90"] is None


# ---------------------------------------------------------------------------
# print_slate_summary
# ---------------------------------------------------------------------------


class TestPrintSlateSummary:
    def test_basic_output(self):
        slate = {
            "counts": {
                "total_recs": 10, "tier1a": 2, "tier1b": 3,
                "tier1": 5, "tier2": 3, "tier3": 1, "stay_away": 1,
            },
            "thresholds": STANDARD_THRESHOLDS,
        }
        output = print_slate_summary(slate)
        assert "Mode: Standard" in output
        assert "Total recs: 10" in output
        assert "Tier 1A: 2" in output

    def test_with_volume_tuning(self):
        slate = {
            "counts": {"total_recs": 1, "tier1a": 0, "tier1b": 1,
                        "tier1": 1, "tier2": 0, "tier3": 0, "stay_away": 0},
            "thresholds": STANDARD_THRESHOLDS,
            "volume_tuning": {
                "total": 1,
                "edge_pct": {"p10": 3.0, "p50": 3.0, "p90": 3.0},
                "edge_z": {"p10": None, "p50": None, "p90": None},
                "sigma": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
                "hold_median": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
                "books_used": {"p10": 5.0, "p50": 5.0, "p90": 5.0},
                "books_histogram": {"1-3": 0, "4-5": 1, "6-7": 0, "8+": 0},
            },
        }
        output = print_slate_summary(slate)
        assert "Volume Tuning:" in output
        assert "edge_pct:" in output
