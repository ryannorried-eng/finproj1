"""Tests for the pluggable tiering system (line_tracker.tiering).

Covers:
  - Percentile math correctness
  - Deterministic output
  - No NaNs in any method
  - Stable behavior with small slate sizes (1, 2, 3 recs)
  - All four tiering methods
  - Composite score normalization
  - assign_tiers API and error handling
  - Edge cases (all-identical recs, negative edges, empty slates)
"""

from __future__ import annotations

import math

import pytest

from line_tracker.best_bets import BetRecommendation
from line_tracker.tiering import (
    STAY_AWAY,
    TIER_1,
    TIER_2,
    TIER_3,
    _min_max_normalize,
    _percentile,
    assign_tiers,
    compute_composite_score,
    slate_distribution_stats,
    tier_frequency_report,
)

# ── Helpers ───────────────────────────────────────────────────────────

VALID_TIERS = {TIER_1, TIER_2, TIER_3, STAY_AWAY}


def _make_rec(
    edge_pct: float = 2.0,
    edge_z: float = 1.8,
    quality_score: int = 70,
    quality_tier: str = "Moderate",
    confidence: str = "Medium",
    **kwargs,
) -> BetRecommendation:
    """Create a minimal BetRecommendation for testing."""
    defaults = dict(
        market="moneyline",
        selection="Team_A",
        side="home",
        line=None,
        consensus_prob=0.55,
        best_sportsbook="DraftKings",
        best_odds=-110,
        breakeven_prob=0.524,
        ev=0.05,
        ev_per_100=2.2,
    )
    defaults.update(kwargs)
    return BetRecommendation(
        edge_pct=edge_pct,
        edge_z=edge_z,
        quality_score=quality_score,
        quality_tier=quality_tier,
        confidence=confidence,
        **defaults,
    )


def _make_slate(specs: list[dict]) -> list[BetRecommendation]:
    """Create a slate from a list of {edge_pct, edge_z, quality_score, ...} dicts."""
    return [_make_rec(**s) for s in specs]


# =====================================================================
# Percentile helper tests
# =====================================================================


class TestPercentileHelper:
    def test_empty_list(self):
        assert _percentile([], 50) == 0.0

    def test_single_element(self):
        assert _percentile([5.0], 0) == 5.0
        assert _percentile([5.0], 50) == 5.0
        assert _percentile([5.0], 100) == 5.0

    def test_two_elements(self):
        assert _percentile([10.0, 20.0], 0) == 10.0
        assert _percentile([10.0, 20.0], 50) == 15.0
        assert _percentile([10.0, 20.0], 100) == 20.0

    def test_known_percentiles(self):
        vals = list(range(1, 101))  # 1..100
        assert _percentile(vals, 50) == pytest.approx(50.5, abs=0.01)
        assert _percentile(vals, 0) == 1.0
        assert _percentile(vals, 100) == 100.0

    def test_unsorted_input(self):
        """Percentile should work even if input is not pre-sorted."""
        vals = [5.0, 1.0, 3.0, 4.0, 2.0]
        assert _percentile(vals, 50) == 3.0

    def test_returns_float(self):
        result = _percentile([1, 2, 3], 50)
        assert isinstance(result, (int, float))
        assert not math.isnan(result)


class TestMinMaxNormalize:
    def test_zero_range(self):
        assert _min_max_normalize(5.0, 5.0, 5.0) == 0.0

    def test_normal_range(self):
        assert _min_max_normalize(5.0, 0.0, 10.0) == pytest.approx(0.5)
        assert _min_max_normalize(0.0, 0.0, 10.0) == pytest.approx(0.0)
        assert _min_max_normalize(10.0, 0.0, 10.0) == pytest.approx(1.0)

    def test_clamped(self):
        assert _min_max_normalize(-5.0, 0.0, 10.0) == 0.0
        assert _min_max_normalize(15.0, 0.0, 10.0) == 1.0


# =====================================================================
# Absolute tiering
# =====================================================================


class TestAbsoluteTiering:
    def test_tier1_high_quality(self):
        rec = _make_rec(
            edge_pct=3.0, edge_z=2.5, quality_score=85,
            quality_tier="Strong", confidence="High",
        )
        assign_tiers([rec], method="absolute")
        assert rec.bet_tier == TIER_1

    def test_tier2_moderate(self):
        rec = _make_rec(
            edge_pct=1.5, edge_z=1.0, quality_score=70,
            quality_tier="Moderate", confidence="Low",
        )
        assign_tiers([rec], method="absolute")
        assert rec.bet_tier == TIER_2

    def test_tier3_low_quality(self):
        rec = _make_rec(
            edge_pct=0.5, edge_z=0.3, quality_score=45,
            quality_tier="Thin", confidence="Low",
        )
        assign_tiers([rec], method="absolute")
        assert rec.bet_tier == TIER_3

    def test_stay_away_negative_edge(self):
        rec = _make_rec(
            edge_pct=-1.0, edge_z=-0.5, quality_score=30,
            quality_tier="Thin", confidence="Low",
        )
        assign_tiers([rec], method="absolute")
        assert rec.bet_tier == STAY_AWAY

    def test_stay_away_zero_edge_low_quality(self):
        rec = _make_rec(
            edge_pct=0.0, edge_z=0.0, quality_score=20,
            quality_tier="Thin", confidence="Low",
        )
        assign_tiers([rec], method="absolute")
        assert rec.bet_tier == STAY_AWAY

    def test_edge_floor(self):
        """Tier 1 requires edge >= dynamic floor."""
        rec = _make_rec(
            edge_pct=1.5, edge_z=2.0, quality_score=80,
            quality_tier="Strong", confidence="High",
        )
        assign_tiers([rec], method="absolute", edge_floor=2.0)
        # edge_pct=1.5 < floor=2.0, so not Tier 1
        assert rec.bet_tier == TIER_2


# =====================================================================
# Percentile tiering (Option A)
# =====================================================================


class TestPercentileTiering:
    def test_basic_ranking(self):
        """Best-Z rec should be Tier 1, worst should be Stay Away."""
        slate = _make_slate([
            {"edge_pct": 5.0, "edge_z": 4.0, "quality_score": 80},
            {"edge_pct": 3.0, "edge_z": 2.5, "quality_score": 70},
            {"edge_pct": 1.5, "edge_z": 1.0, "quality_score": 60},
            {"edge_pct": 0.5, "edge_z": 0.3, "quality_score": 55},
            {"edge_pct": -0.5, "edge_z": -0.3, "quality_score": 40},
        ])
        assign_tiers(slate, method="percentile")
        assert slate[0].bet_tier == TIER_1
        assert slate[-1].bet_tier == STAY_AWAY

    def test_guarantee_one_tier1(self):
        """At least one Tier 1 when any rec passes floors."""
        slate = _make_slate([
            {"edge_pct": 0.5, "edge_z": 0.3, "quality_score": 55},
            {"edge_pct": 0.4, "edge_z": 0.2, "quality_score": 52},
            {"edge_pct": 0.3, "edge_z": 0.1, "quality_score": 51},
        ])
        assign_tiers(slate, method="percentile")
        t1_count = sum(1 for r in slate if r.bet_tier == TIER_1)
        assert t1_count >= 1

    def test_quality_floor_enforced(self):
        """Recs with quality < 50 get Stay Away even with high Z."""
        rec = _make_rec(edge_pct=3.0, edge_z=3.0, quality_score=40)
        assign_tiers([rec], method="percentile")
        assert rec.bet_tier == STAY_AWAY

    def test_negative_edge_stays_away(self):
        rec = _make_rec(edge_pct=-1.0, edge_z=2.0, quality_score=80)
        assign_tiers([rec], method="percentile")
        assert rec.bet_tier == STAY_AWAY

    def test_all_identical_z(self):
        """When all Z-values are identical, best-of-slate guarantee kicks in."""
        slate = _make_slate([
            {"edge_pct": 2.0, "edge_z": 1.5, "quality_score": 70},
            {"edge_pct": 2.0, "edge_z": 1.5, "quality_score": 70},
            {"edge_pct": 2.0, "edge_z": 1.5, "quality_score": 70},
        ])
        assign_tiers(slate, method="percentile")
        t1_count = sum(1 for r in slate if r.bet_tier == TIER_1)
        assert t1_count >= 1


# =====================================================================
# Hybrid tiering (Option B)
# =====================================================================


class TestHybridTiering:
    def test_strong_slate_absolute_applies(self):
        """On a strong slate, absolute Z floor (1.5) gates Tier 1."""
        slate = _make_slate([
            {"edge_pct": 5.0, "edge_z": 3.0},  # above both
            {"edge_pct": 3.0, "edge_z": 1.8},   # above absolute but may miss p90
            {"edge_pct": 1.0, "edge_z": 0.5},   # below absolute
        ])
        assign_tiers(slate, method="hybrid")
        assert slate[0].bet_tier == TIER_1
        # Low-Z rec should be Tier 3 or Stay Away
        assert slate[2].bet_tier in {TIER_3, STAY_AWAY}

    def test_weak_slate_no_forced_tier1(self):
        """On a weak slate, Tier 1 is empty if no rec meets Z >= 1.5."""
        slate = _make_slate([
            {"edge_pct": 0.5, "edge_z": 0.8},
            {"edge_pct": 0.3, "edge_z": 0.4},
            {"edge_pct": 0.1, "edge_z": 0.1},
        ])
        assign_tiers(slate, method="hybrid")
        t1_count = sum(1 for r in slate if r.bet_tier == TIER_1)
        assert t1_count == 0  # No forced plays

    def test_negative_edge_stays_away(self):
        rec = _make_rec(edge_pct=-1.0, edge_z=3.0)
        assign_tiers([rec], method="hybrid")
        assert rec.bet_tier == STAY_AWAY

    def test_tier_ordering(self):
        """Tiers should be monotonically ordered by Z-score."""
        slate = _make_slate([
            {"edge_pct": 5.0, "edge_z": 3.5},
            {"edge_pct": 3.0, "edge_z": 2.0},
            {"edge_pct": 1.5, "edge_z": 1.2},
            {"edge_pct": 0.5, "edge_z": 0.3},
        ])
        assign_tiers(slate, method="hybrid")
        tier_order = {TIER_1: 0, TIER_2: 1, TIER_3: 2, STAY_AWAY: 3}
        tiers = [tier_order[r.bet_tier] for r in slate]
        # Should be non-decreasing (higher Z → lower tier number)
        assert tiers == sorted(tiers)


# =====================================================================
# Composite tiering (Option C)
# =====================================================================


class TestCompositeTiering:
    def test_best_rec_gets_tier1(self):
        """Rec with highest composite should be Tier 1."""
        slate = _make_slate([
            {"edge_pct": 5.0, "edge_z": 4.0, "quality_score": 90},
            {"edge_pct": 2.0, "edge_z": 1.5, "quality_score": 65},
            {"edge_pct": 0.5, "edge_z": 0.2, "quality_score": 40},
            {"edge_pct": -0.5, "edge_z": -0.3, "quality_score": 30},
        ])
        assign_tiers(slate, method="composite")
        assert slate[0].bet_tier == TIER_1

    def test_composite_score_range(self):
        """Composite score should be in [0, 1]."""
        score = compute_composite_score(
            edge_pct=3.0,
            edge_z=2.0,
            quality_score=75.0,
            slate_edge_pct_range=(0.0, 5.0),
            slate_edge_z_range=(0.0, 4.0),
            slate_quality_range=(30.0, 90.0),
        )
        assert 0.0 <= score <= 1.0

    def test_composite_monotonic_in_inputs(self):
        """Higher edge_pct / edge_z / quality should yield higher score."""
        ranges = dict(
            slate_edge_pct_range=(0.0, 10.0),
            slate_edge_z_range=(0.0, 5.0),
            slate_quality_range=(0.0, 100.0),
        )
        s_low = compute_composite_score(1.0, 0.5, 30.0, **ranges)
        s_high = compute_composite_score(8.0, 4.0, 90.0, **ranges)
        assert s_high > s_low

    def test_zero_range_normalization(self):
        """When all values are identical, normalization returns 0 (no crash)."""
        slate = _make_slate([
            {"edge_pct": 2.0, "edge_z": 1.5, "quality_score": 70},
            {"edge_pct": 2.0, "edge_z": 1.5, "quality_score": 70},
        ])
        assign_tiers(slate, method="composite")
        for r in slate:
            assert r.bet_tier in VALID_TIERS


# =====================================================================
# assign_tiers API tests
# =====================================================================


class TestAssignTiersAPI:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown tiering method"):
            assign_tiers([], method="bogus")

    def test_empty_slate(self):
        result = assign_tiers([], method="hybrid")
        assert result == []

    def test_returns_same_list(self):
        slate = [_make_rec()]
        result = assign_tiers(slate, method="hybrid")
        assert result is slate

    def test_all_methods_produce_valid_tiers(self):
        """Every method assigns one of the four valid tier labels."""
        slate = _make_slate([
            {"edge_pct": 4.0, "edge_z": 3.0, "quality_score": 85,
             "quality_tier": "Strong", "confidence": "High"},
            {"edge_pct": 1.5, "edge_z": 1.0, "quality_score": 60,
             "quality_tier": "Moderate", "confidence": "Low"},
            {"edge_pct": -0.5, "edge_z": -0.2, "quality_score": 30,
             "quality_tier": "Thin", "confidence": "Low"},
        ])
        for method in ("absolute", "percentile", "hybrid", "composite"):
            for rec in slate:
                rec.bet_tier = ""
            assign_tiers(slate, method=method)
            for rec in slate:
                assert rec.bet_tier in VALID_TIERS, (
                    f"method={method} produced invalid tier: {rec.bet_tier!r}"
                )

    def test_propagates_to_best_bet_result(self):
        """bet_tier is propagated to attached BestBetResult."""
        from line_tracker.models import BestBetResult

        rec = _make_rec(edge_pct=4.0, edge_z=3.0, quality_score=85,
                        quality_tier="Strong", confidence="High")
        bbr = BestBetResult(
            edge_pct=4.0, consensus_prob=0.55,
            best_odds_american=-110, best_odds_decimal=1.909,
            books_used=["DK"], volatility_sigma=0.02,
            recency_weight=0.9, outliers_removed=0,
        )
        rec.best_bet_result = bbr
        assign_tiers([rec], method="hybrid")
        assert bbr.bet_tier == rec.bet_tier


# =====================================================================
# Determinism
# =====================================================================


class TestDeterminism:
    def test_same_input_same_output(self):
        """Running assign_tiers twice on identical inputs produces same result."""
        for method in ("absolute", "percentile", "hybrid", "composite"):
            slate1 = _make_slate([
                {"edge_pct": 3.0, "edge_z": 2.5, "quality_score": 80},
                {"edge_pct": 1.0, "edge_z": 0.8, "quality_score": 55},
                {"edge_pct": -0.5, "edge_z": -0.2, "quality_score": 30},
            ])
            slate2 = _make_slate([
                {"edge_pct": 3.0, "edge_z": 2.5, "quality_score": 80},
                {"edge_pct": 1.0, "edge_z": 0.8, "quality_score": 55},
                {"edge_pct": -0.5, "edge_z": -0.2, "quality_score": 30},
            ])
            assign_tiers(slate1, method=method)
            assign_tiers(slate2, method=method)
            for r1, r2 in zip(slate1, slate2):
                assert r1.bet_tier == r2.bet_tier, (
                    f"Non-deterministic: method={method}"
                )


# =====================================================================
# No NaNs
# =====================================================================


class TestNoNaNs:
    def test_no_nan_tiers(self):
        """No method should produce NaN or empty-string bet_tier."""
        slates = [
            # Normal
            _make_slate([
                {"edge_pct": 2.0, "edge_z": 1.5, "quality_score": 70},
            ]),
            # All zeros
            _make_slate([
                {"edge_pct": 0.0, "edge_z": 0.0, "quality_score": 50},
            ]),
            # Extreme values
            _make_slate([
                {"edge_pct": 15.0, "edge_z": 8.0, "quality_score": 100},
                {"edge_pct": -5.0, "edge_z": -3.0, "quality_score": 10},
            ]),
        ]
        for method in ("absolute", "percentile", "hybrid", "composite"):
            for slate in slates:
                for rec in slate:
                    rec.bet_tier = ""
                assign_tiers(slate, method=method)
                for rec in slate:
                    assert rec.bet_tier != "", f"Empty tier: method={method}"
                    assert rec.bet_tier in VALID_TIERS


# =====================================================================
# Small slate stability
# =====================================================================


_ALL_METHODS = ["absolute", "percentile", "hybrid", "composite"]


class TestSmallSlateStability:
    @pytest.mark.parametrize("method", _ALL_METHODS)
    def test_single_rec(self, method):
        slate = [_make_rec(
            edge_pct=2.0, edge_z=2.0, quality_score=75,
            quality_tier="Moderate", confidence="Medium",
        )]
        assign_tiers(slate, method=method)
        assert slate[0].bet_tier in VALID_TIERS

    @pytest.mark.parametrize("method", _ALL_METHODS)
    def test_two_recs(self, method):
        slate = _make_slate([
            {"edge_pct": 3.0, "edge_z": 2.5, "quality_score": 80},
            {"edge_pct": 0.5, "edge_z": 0.2, "quality_score": 50},
        ])
        assign_tiers(slate, method=method)
        for r in slate:
            assert r.bet_tier in VALID_TIERS

    @pytest.mark.parametrize("method", _ALL_METHODS)
    def test_three_recs(self, method):
        slate = _make_slate([
            {"edge_pct": 4.0, "edge_z": 3.0, "quality_score": 85},
            {"edge_pct": 2.0, "edge_z": 1.5, "quality_score": 65},
            {"edge_pct": 0.5, "edge_z": 0.3, "quality_score": 45},
        ])
        assign_tiers(slate, method=method)
        for r in slate:
            assert r.bet_tier in VALID_TIERS


# =====================================================================
# Distribution stats helper
# =====================================================================


class TestSlateDistributionStats:
    def test_empty(self):
        stats = slate_distribution_stats([])
        assert stats["n"] == 0

    def test_basic_stats(self):
        slate = _make_slate([
            {"edge_pct": 1.0, "edge_z": 0.5, "quality_score": 60},
            {"edge_pct": 3.0, "edge_z": 2.0, "quality_score": 80},
            {"edge_pct": 5.0, "edge_z": 3.5, "quality_score": 90},
        ])
        stats = slate_distribution_stats(slate)
        assert stats["n"] == 3
        assert stats["edge_pct_min"] == 1.0
        assert stats["edge_pct_max"] == 5.0
        assert "edge_z_p90" in stats
        assert "quality_score_p50" in stats


class TestTierFrequencyReport:
    def test_all_tiers_counted(self):
        slate = _make_slate([
            {"edge_pct": 5.0, "edge_z": 4.0, "quality_score": 90,
             "quality_tier": "Elite", "confidence": "High"},
            {"edge_pct": 2.0, "edge_z": 1.5, "quality_score": 65,
             "quality_tier": "Moderate", "confidence": "Low"},
            {"edge_pct": -1.0, "edge_z": -0.5, "quality_score": 30,
             "quality_tier": "Thin", "confidence": "Low"},
        ])
        assign_tiers(slate, method="absolute")
        report = tier_frequency_report(slate)
        assert report["total"] == 3
        total_counted = sum(
            report[f"{t}_count"] for t in [TIER_1, TIER_2, TIER_3, STAY_AWAY]
        )
        assert total_counted == 3
