"""Tests for the confidence gating layer.

Covers:
  - confidence_label assignment (High, Medium, Low)
  - Tier 1 probability floor (confidence gating in assign_tiers)
  - Hit-mode primary card excludes Low confidence when alternatives exist
  - Calibration report includes confidence breakdown
"""

from __future__ import annotations

import pandas as pd
import pytest

from line_tracker.best_bets import BetRecommendation
from line_tracker.performance import confidence_label_report
from line_tracker.scoring import (
    compute_confidence_label,
    enrich_entry,
    filter_candidates,
    rank_candidates,
)
from line_tracker.tiering import (
    TIER_1,
    TIER_2,
    assign_tiers,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_entry(**overrides) -> dict:
    """Create a minimal entry dict for scoring tests."""
    base = {
        "consensus_prob": 0.55,
        "quality_score": 75,
        "market_hold_median": 5.0,  # 5% hold → 0.05 decimal
        "alpha_label": "Neutral",
        "alpha_score": 50,
        "edge_z": 1.5,
        "edge_pct": 2.0,
        "edge_ev_shrunk": 0.02,
        "edge_shrunk_pct": 2.0,
        "ev_100": 2.0,
        "market": "moneyline",
        "selection": "Team_A",
        "kelly_suggested": 0.03,
        "agreement_score": 70.0,
        "market_volatility_sigma": 0.006,
    }
    base.update(overrides)
    return base


def _make_rec(
    edge_pct: float = 2.0,
    edge_z: float = 1.8,
    quality_score: int = 70,
    quality_tier: str = "Moderate",
    confidence: str = "Medium",
    **kwargs,
) -> BetRecommendation:
    """Create a minimal BetRecommendation for testing."""
    # Separate out fields that are not BetRecommendation constructor args
    extra_attrs = {}
    for key in ("alpha_label", "alpha_score"):
        if key in kwargs:
            extra_attrs[key] = kwargs.pop(key)

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
    rec = BetRecommendation(
        edge_pct=edge_pct,
        edge_z=edge_z,
        quality_score=quality_score,
        quality_tier=quality_tier,
        confidence=confidence,
        **defaults,
    )
    # Set non-dataclass attrs (used by confidence gating)
    for attr, val in extra_attrs.items():
        setattr(rec, attr, val)
    return rec


# =====================================================================
# Task A: confidence_label assignment
# =====================================================================


class TestConfidenceLabelAssignment:
    """Verify compute_confidence_label returns correct labels."""

    def test_high_confidence_case(self):
        entry = _make_entry(
            quality_score=75,
            consensus_prob=0.50,
            market_hold_median=4.0,  # 0.04 decimal
            alpha_label="Strong",
            edge_z=1.0,
        )
        assert compute_confidence_label(entry) == "High"

    def test_high_confidence_via_edge_z(self):
        """High via edge_z >= 0.8 (alpha not Strong)."""
        entry = _make_entry(
            quality_score=75,
            consensus_prob=0.50,
            market_hold_median=4.0,
            alpha_label="Neutral",
            edge_z=0.9,
        )
        assert compute_confidence_label(entry) == "High"

    def test_medium_confidence_case(self):
        entry = _make_entry(
            quality_score=65,
            consensus_prob=0.40,
            market_hold_median=5.0,  # 0.05 decimal
            alpha_label="Neutral",
            edge_z=0.5,  # below 0.8, alpha not Strong → not High
        )
        assert compute_confidence_label(entry) == "Medium"

    def test_low_confidence_case(self):
        entry = _make_entry(
            quality_score=50,
            consensus_prob=0.25,
            market_hold_median=12.0,  # 0.12 decimal, exceeds 0.10
            alpha_label="Weak",
            edge_z=0.3,
        )
        assert compute_confidence_label(entry) == "Low"

    def test_low_when_quality_too_low_for_medium(self):
        """Quality < 60 → cannot be Medium."""
        entry = _make_entry(
            quality_score=55,
            consensus_prob=0.40,
            market_hold_median=5.0,
            alpha_label="Neutral",
            edge_z=0.5,
        )
        assert compute_confidence_label(entry) == "Low"

    def test_low_when_hold_too_high_for_high(self):
        """hold > 0.09 prevents High even with other good signals."""
        entry = _make_entry(
            quality_score=80,
            consensus_prob=0.55,
            market_hold_median=10.0,  # 0.10 > 0.09
            alpha_label="Strong",
            edge_z=2.0,
        )
        # Passes High quality/prob/alpha but fails hold → Medium
        assert compute_confidence_label(entry) == "Medium"

    def test_enrich_entry_adds_confidence_label(self):
        entry = _make_entry(
            quality_score=75,
            consensus_prob=0.50,
            market_hold_median=4.0,
            edge_z=1.0,
            # alpha_label will be computed by enrich_entry
        )
        enriched = enrich_entry(entry)
        assert "confidence_label" in enriched
        assert enriched["confidence_label"] in ("High", "Medium", "Low")


# =====================================================================
# Task B: Tier1 probability floor (confidence gating)
# =====================================================================


class TestTier1ConfidenceGating:
    """Verify that Tier 1 assignment is gated by probability floor."""

    def test_longshot_cannot_become_tier1_unless_alpha_strong(self):
        """prob=0.16, alpha Neutral → demoted from Tier 1."""
        rec = _make_rec(
            edge_pct=5.0,
            edge_z=3.0,
            quality_score=85,
            quality_tier="Elite",
            confidence="High",
            consensus_prob=0.16,
            alpha_label="Neutral",
            edge_ev_shrunk=0.05,
        )
        # Use absolute method which would normally give Tier 1
        assign_tiers([rec], method="absolute")
        # Confidence gate demotes to Tier 2
        assert rec.bet_tier == TIER_2

    def test_alpha_strong_overrides_probability_floor(self):
        """prob=0.16, alpha Strong → stays Tier 1."""
        rec = _make_rec(
            edge_pct=5.0,
            edge_z=3.0,
            quality_score=85,
            quality_tier="Elite",
            confidence="High",
            consensus_prob=0.16,
            alpha_label="Strong",
            edge_ev_shrunk=0.05,
        )
        assign_tiers([rec], method="absolute")
        assert rec.bet_tier == TIER_1

    def test_normal_prob_stays_tier1(self):
        """prob=0.55 (above 0.40) → stays Tier 1."""
        rec = _make_rec(
            edge_pct=3.0,
            edge_z=2.0,
            quality_score=80,
            quality_tier="Strong",
            confidence="High",
            consensus_prob=0.55,
            alpha_label="Neutral",
            edge_ev_shrunk=0.03,
        )
        assign_tiers([rec], method="absolute")
        assert rec.bet_tier == TIER_1

    def test_borderline_prob_at_threshold(self):
        """prob=0.40 exactly → stays Tier 1."""
        rec = _make_rec(
            edge_pct=3.0,
            edge_z=2.0,
            quality_score=80,
            quality_tier="Strong",
            confidence="High",
            consensus_prob=0.40,
            alpha_label="Neutral",
            edge_ev_shrunk=0.03,
        )
        assign_tiers([rec], method="absolute")
        assert rec.bet_tier == TIER_1

    def test_gating_does_not_remove_from_ranking(self):
        """Demoted rec stays in list (not removed)."""
        recs = [
            _make_rec(
                edge_pct=5.0, edge_z=3.0, quality_score=85,
                quality_tier="Elite", confidence="High",
                consensus_prob=0.16, alpha_label="Neutral",
                edge_ev_shrunk=0.05,
            ),
            _make_rec(
                edge_pct=2.0, edge_z=1.5, quality_score=70,
                quality_tier="Moderate", confidence="Medium",
                consensus_prob=0.55, alpha_label="Neutral",
                edge_ev_shrunk=0.02,
            ),
        ]
        assign_tiers(recs, method="absolute")
        assert len(recs) == 2  # Both still present


# =====================================================================
# Task C: Hit-mode primary card excludes Low confidence
# =====================================================================


class TestHitModePrimaryGating:
    """Verify that hit-mode primary card prefers non-Low confidence."""

    def test_hit_mode_primary_excludes_low_when_alternatives_exist(self):
        """The first qualified entry should be non-Low if possible."""
        # Create entries: first has highest prob but Low confidence_label
        low_conf = _make_entry(
            consensus_prob=0.70,
            quality_score=80,
            confidence_label="Low",
            alpha_score=40,
            edge_z=0.5,
        )
        medium_conf = _make_entry(
            consensus_prob=0.60,
            quality_score=75,
            confidence_label="Medium",
            alpha_score=60,
            edge_z=1.5,
            selection="Team_B",
        )

        entries = [enrich_entry(low_conf), enrich_entry(medium_conf)]
        ranked = rank_candidates(entries, mode="hit")
        qualified = filter_candidates(ranked, "hit", 0.0, 50)

        # Apply confidence gating (same logic as dashboard)
        if len(qualified) > 1:
            non_low = [
                e for e in qualified
                if e.get("confidence_label") != "Low"
            ]
            if non_low:
                primary = non_low[0]
                others_q = [e for e in qualified if e is not primary]
                qualified = [primary] + others_q

        # Primary should be the Medium confidence entry
        assert qualified[0].get("confidence_label") != "Low"

    def test_hit_mode_falls_back_when_all_low(self):
        """If all are Low, falls back to ranked[0]."""
        # Both entries will compute to Low confidence_label:
        # quality < 60, prob < 0.35, high hold → Low
        low1 = _make_entry(
            consensus_prob=0.25,
            quality_score=50,
            market_hold_median=12.0,
            alpha_score=40,
            edge_z=0.3,
        )
        low2 = _make_entry(
            consensus_prob=0.20,
            quality_score=45,
            market_hold_median=12.0,
            alpha_score=30,
            edge_z=0.2,
            selection="Team_B",
        )

        entries = [enrich_entry(low1), enrich_entry(low2)]
        # Verify both are indeed Low
        assert entries[0]["confidence_label"] == "Low"
        assert entries[1]["confidence_label"] == "Low"

        ranked = rank_candidates(entries, mode="hit")
        qualified = filter_candidates(ranked, "hit", 0.0, 40)

        # Gating with all-Low → no reorder
        non_low = [
            e for e in qualified if e.get("confidence_label") != "Low"
        ]
        if non_low:
            primary = non_low[0]
            others_q = [e for e in qualified if e is not primary]
            qualified = [primary] + others_q

        # Still returns the first by ranking
        assert len(qualified) == 2
        assert qualified[0].get("confidence_label") == "Low"

    def test_value_mode_no_gating(self):
        """Value mode does not reorder by confidence_label."""
        low_conf = _make_entry(
            consensus_prob=0.55,
            quality_score=80,
            confidence_label="Low",
            alpha_score=40,
            edge_z=0.5,
            edge_ev_shrunk=0.05,
            edge_shrunk_pct=5.0,
        )
        medium_conf = _make_entry(
            consensus_prob=0.60,
            quality_score=75,
            confidence_label="Medium",
            alpha_score=60,
            edge_z=1.5,
            edge_ev_shrunk=0.02,
            edge_shrunk_pct=2.0,
            selection="Team_B",
        )

        entries = [enrich_entry(low_conf), enrich_entry(medium_conf)]
        ranked = rank_candidates(entries, mode="value")
        qualified = filter_candidates(ranked, "value", 0.0, 50)

        # Value mode: no confidence gating → first by EV
        # low_conf has higher edge_ev_shrunk so it stays first
        assert qualified[0].get("edge_ev_shrunk") >= qualified[1].get(
            "edge_ev_shrunk"
        )


# =====================================================================
# Task E: Calibration report includes confidence breakdown
# =====================================================================


class TestConfidenceLabelReport:
    """Verify confidence_label_report produces correct breakdowns."""

    @staticmethod
    def _make_clv_df() -> pd.DataFrame:
        """Build a small test CLV DataFrame."""
        return pd.DataFrame({
            "confidence_label_at_pick": [
                "High", "High", "Medium", "Medium", "Low",
            ],
            "consensus_prob_at_pick": [0.55, 0.50, 0.38, 0.42, 0.20],
            "clv_decimal": [0.03, 0.01, -0.01, 0.02, -0.02],
            "clv_prob": [0.02, 0.01, -0.005, 0.01, -0.01],
        })

    def test_by_label_keys(self):
        df = self._make_clv_df()
        report = confidence_label_report(df)
        assert "by_label" in report
        for label in ("High", "Medium", "Low"):
            assert label in report["by_label"]

    def test_by_label_counts(self):
        df = self._make_clv_df()
        report = confidence_label_report(df)
        assert report["by_label"]["High"]["count"] == 2
        assert report["by_label"]["Medium"]["count"] == 2
        assert report["by_label"]["Low"]["count"] == 1

    def test_by_label_beating_pct(self):
        df = self._make_clv_df()
        report = confidence_label_report(df)
        # High: both positive clv_prob → 100%
        assert report["by_label"]["High"]["beating_pct"] == 100.0
        # Low: negative clv_prob → 0%
        assert report["by_label"]["Low"]["beating_pct"] == 0.0

    def test_by_prob_bucket_keys(self):
        df = self._make_clv_df()
        report = confidence_label_report(df)
        assert "by_prob_bucket" in report
        for bucket in ("< 0.30", "0.30–0.45", "> 0.45"):
            assert bucket in report["by_prob_bucket"]

    def test_by_prob_bucket_counts(self):
        df = self._make_clv_df()
        report = confidence_label_report(df)
        # prob < 0.30: only 0.20
        assert report["by_prob_bucket"]["< 0.30"]["count"] == 1
        # prob 0.30–0.45: 0.38, 0.42
        assert report["by_prob_bucket"]["0.30–0.45"]["count"] == 2
        # prob > 0.45: 0.55, 0.50
        assert report["by_prob_bucket"]["> 0.45"]["count"] == 2

    def test_empty_df(self):
        report = confidence_label_report(pd.DataFrame())
        assert report == {"by_label": {}, "by_prob_bucket": {}}

    def test_missing_confidence_label_column(self):
        """When column not present, by_label is empty but prob buckets work."""
        df = pd.DataFrame({
            "consensus_prob_at_pick": [0.55, 0.30, 0.20],
            "clv_decimal": [0.01, 0.02, -0.01],
            "clv_prob": [0.01, 0.02, -0.01],
        })
        report = confidence_label_report(df)
        assert report["by_label"] == {}
        assert report["by_prob_bucket"]["> 0.45"]["count"] == 1
