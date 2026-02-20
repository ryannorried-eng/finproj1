"""Tests for market weighting layer and effective Kelly sizing.

Covers:
  1. market_weight mapping by market type
  2. Tier quantile scores use weighted tier_score
  3. Hybrid score weighting applied only in hybrid mode
  4. Effective Kelly multiplier by confidence_label
  5. Effective Kelly does not change pick selection
"""

from __future__ import annotations

import pytest

from line_tracker.best_bets import BetRecommendation
from line_tracker.scoring import (
    classify_market,
    compute_hybrid_fields,
    compute_kelly_effective,
    enrich_entry,
    market_weight,
    rank_candidates,
)
from line_tracker.tiering import (
    STAY_AWAY,
    TIER_1,
    TIER_2,
    assign_tiers,
    assign_tiers_quantile,
    market_tier_report,
)

# ── Helpers ───────────────────────────────────────────────────────────


def _entry(
    *,
    market: str = "moneyline",
    best_odds: float = -110,
    consensus_prob: float = 0.55,
    alpha_score: int = 70,
    kelly_suggested: float = 0.02,
    kelly_base: float = 0.04,
    quality_score: int = 75,
    quality_tier: str = "Strong",
    edge_z: float = 2.0,
    edge_ev_shrunk: float = 0.02,
    ev_100: float = 2.0,
    agreement_score: float = 80.0,
    market_volatility_sigma: float = 0.005,
    market_hold_median: float = 5.0,
    books_used: int = 6,
    **extra,
) -> dict:
    """Build a minimal entry dict for scoring tests."""
    d: dict = {
        "market": market,
        "best_odds": best_odds,
        "consensus_prob": consensus_prob,
        "alpha_score": alpha_score,
        "kelly_suggested": kelly_suggested,
        "kelly_base": kelly_base,
        "quality_score": quality_score,
        "quality_tier": quality_tier,
        "edge_z": edge_z,
        "edge_ev_shrunk": edge_ev_shrunk,
        "ev_100": ev_100,
        "agreement_score": agreement_score,
        "market_volatility_sigma": market_volatility_sigma,
        "market_hold_median": market_hold_median,
        "books_used": books_used,
    }
    d.update(extra)
    return d


def _make_rec(
    edge_pct: float = 2.0,
    edge_z: float = 1.8,
    quality_score: int = 70,
    market: str = "moneyline",
    consensus_prob: float = 0.55,
    best_odds: float = -110,
    **kwargs,
) -> BetRecommendation:
    """Create a minimal BetRecommendation for testing."""
    defaults = dict(
        selection="Team_A",
        side="home",
        line=None,
        best_sportsbook="DraftKings",
        breakeven_prob=0.524,
        ev=0.05,
        ev_per_100=2.2,
        quality_tier="Moderate",
        confidence="Medium",
    )
    defaults.update(kwargs)
    return BetRecommendation(
        market=market,
        edge_pct=edge_pct,
        edge_z=edge_z,
        quality_score=quality_score,
        consensus_prob=consensus_prob,
        best_odds=best_odds,
        **defaults,
    )


# =====================================================================
# 1. test_market_weight_mapping
# =====================================================================


class TestMarketWeightMapping:
    """Verify market_weight returns correct weights for each category."""

    def test_spreads_returns_1_00(self):
        e = _entry(market="spreads")
        assert market_weight(e) == pytest.approx(1.00)

    def test_spread_singular_returns_1_00(self):
        e = _entry(market="spread")
        assert market_weight(e) == pytest.approx(1.00)

    def test_totals_returns_0_95(self):
        e = _entry(market="totals")
        assert market_weight(e) == pytest.approx(0.95)

    def test_total_singular_returns_0_95(self):
        e = _entry(market="total")
        assert market_weight(e) == pytest.approx(0.95)

    def test_ml_favorite_returns_0_90(self):
        """Negative American odds → ml_favorite → 0.90."""
        e = _entry(market="moneyline", best_odds=-150, consensus_prob=0.60)
        assert market_weight(e) == pytest.approx(0.90)

    def test_ml_favorite_via_prob(self):
        """Positive odds but consensus_prob >= 0.55 → ml_favorite."""
        e = _entry(market="moneyline", best_odds=110, consensus_prob=0.56)
        assert market_weight(e) == pytest.approx(0.90)

    def test_ml_underdog_returns_0_75(self):
        """Positive odds + prob < 0.55 → ml_underdog → 0.75."""
        e = _entry(market="moneyline", best_odds=200, consensus_prob=0.35)
        assert market_weight(e) == pytest.approx(0.75)

    def test_ml_longshot_multiplicative_penalty(self):
        """ML underdog + prob < 0.20 → 0.75 * 0.90 = 0.675."""
        e = _entry(market="moneyline", best_odds=500, consensus_prob=0.15)
        expected = 0.75 * 0.90  # ml_underdog * longshot
        assert market_weight(e) == pytest.approx(expected, abs=0.001)

    def test_ml_favorite_longshot_penalty(self):
        """ML favorite + prob < 0.20 → 0.90 * 0.90 = 0.81."""
        # Edge case: negative odds but very low prob (unusual but possible)
        e = _entry(market="moneyline", best_odds=-110, consensus_prob=0.18)
        expected = 0.90 * 0.90
        assert market_weight(e) == pytest.approx(expected, abs=0.001)

    def test_h2h_treated_as_moneyline(self):
        """'h2h' market type is equivalent to 'moneyline'."""
        e = _entry(market="h2h", best_odds=-150, consensus_prob=0.60)
        assert market_weight(e) == pytest.approx(0.90)

    def test_unknown_market_returns_1_00(self):
        """Unknown market type → neutral weight 1.0."""
        e = _entry(market="player_props")
        assert market_weight(e) == pytest.approx(1.00)

    def test_missing_market_returns_1_00(self):
        """Missing market field → neutral weight 1.0."""
        assert market_weight({}) == pytest.approx(1.00)

    def test_works_with_bet_recommendation(self):
        """market_weight works with BetRecommendation objects."""
        rec = _make_rec(market="spreads")
        assert market_weight(rec) == pytest.approx(1.00)

        rec_dog = _make_rec(market="moneyline", best_odds=200, consensus_prob=0.35)
        assert market_weight(rec_dog) == pytest.approx(0.75)


class TestClassifyMarket:
    def test_classify_spread(self):
        assert classify_market({"market": "spread"}) == "spreads"
        assert classify_market({"market": "spreads"}) == "spreads"

    def test_classify_total(self):
        assert classify_market({"market": "total"}) == "totals"
        assert classify_market({"market": "totals"}) == "totals"

    def test_classify_ml_favorite(self):
        assert classify_market(
            {"market": "moneyline", "best_odds": -150, "consensus_prob": 0.60}
        ) == "ml_favorite"

    def test_classify_ml_underdog(self):
        assert classify_market(
            {"market": "moneyline", "best_odds": 200, "consensus_prob": 0.35}
        ) == "ml_underdog"

    def test_classify_unknown(self):
        assert classify_market({"market": "props"}) == "unknown"
        assert classify_market({}) == "unknown"


# =====================================================================
# 2. test_tier_quantiles_use_weighted_score
# =====================================================================


class TestTierQuantilesUseWeightedScore:
    """Verify that quantile tiering uses market-weighted scores."""

    def test_spread_beats_ml_underdog_with_equal_base_score(self):
        """Given equal base tier_score, a spread rec should rank higher
        in weighted_score and be more likely to land in Tier 1."""
        # Create 15 padding recs so quantile bucketing behaves normally
        padding = [
            _make_rec(
                edge_pct=1.0 + i * 0.3,
                edge_z=1.0 + i * 0.15,
                quality_score=55 + i * 2,
                edge_ev_shrunk=0.01 + i * 0.002,
                agreement_score=50.0 + i * 2,
                books_used_count=5,
                market_hold_median=5.0,
                market="spreads",
                consensus_prob=0.55,
                best_odds=-110,
                selection=f"Pad_{i}",
            )
            for i in range(15)
        ]

        # Two recs with identical signals except market type
        spread_rec = _make_rec(
            edge_pct=4.0,
            edge_z=2.5,
            quality_score=80,
            edge_ev_shrunk=0.04,
            agreement_score=85.0,
            books_used_count=7,
            market_hold_median=4.0,
            market="spreads",
            consensus_prob=0.55,
            best_odds=-110,
            selection="Spread_Team",
        )
        dog_rec = _make_rec(
            edge_pct=4.0,
            edge_z=2.5,
            quality_score=80,
            edge_ev_shrunk=0.04,
            agreement_score=85.0,
            books_used_count=7,
            market_hold_median=4.0,
            market="moneyline",
            consensus_prob=0.35,  # underdog
            best_odds=200,
            selection="Dog_Team",
        )

        slate = [spread_rec, dog_rec] + padding
        info = assign_tiers_quantile(slate)

        # Check weighted scores: spread should be higher
        scores_by_idx = dict(info["scores"])

        spread_weighted = scores_by_idx[0]
        dog_weighted = scores_by_idx[1]

        assert spread_weighted > dog_weighted, (
            f"Spread weighted ({spread_weighted}) should be > "
            f"Dog weighted ({dog_weighted})"
        )

        # Spread should be Tier 1 or at least higher-tiered than dog
        tier_order = {TIER_1: 0, TIER_2: 1, STAY_AWAY: 2}
        spread_rank = tier_order.get(spread_rec.bet_tier, 3)
        dog_rank = tier_order.get(dog_rec.bet_tier, 3)
        assert spread_rank <= dog_rank

    def test_base_scores_preserved_in_metadata(self):
        """assign_tiers_quantile returns base_scores for debugging."""
        slate = [
            _make_rec(
                edge_pct=3.0, edge_z=2.0, quality_score=75,
                edge_ev_shrunk=0.03, market="spreads",
            ),
        ]
        info = assign_tiers_quantile(slate, min_candidates=1)
        assert "base_scores" in info
        assert "market_weights" in info
        assert len(info["base_scores"]) == 1
        assert len(info["market_weights"]) == 1

    def test_market_weights_in_metadata(self):
        """assign_tiers_quantile returns per-rec market_weights."""
        slate = [
            _make_rec(market="spreads", edge_ev_shrunk=0.03, edge_pct=3.0),
            _make_rec(
                market="moneyline", best_odds=200, consensus_prob=0.35,
                edge_ev_shrunk=0.03, edge_pct=3.0, selection="Team_B",
            ),
        ]
        info = assign_tiers_quantile(slate, min_candidates=1)
        mw_dict = dict(info["market_weights"])
        assert mw_dict[0] == pytest.approx(1.00)  # spreads
        assert mw_dict[1] == pytest.approx(0.75)  # ml_underdog


# =====================================================================
# 3. test_hybrid_score_weighting_applied_only_in_hybrid_mode
# =====================================================================


class TestHybridScoreWeightingByMode:
    """Verify that market weighting affects hybrid mode but not hit/value."""

    def test_hybrid_mode_ordering_changes_due_to_weights(self):
        """In hybrid mode, market weight should change ordering when
        two entries have similar raw scores but different markets."""
        # ML underdog: good raw score but penalized by market_weight (0.75)
        dog = _entry(
            market="moneyline",
            best_odds=200,
            consensus_prob=0.35,
            alpha_score=80,
            kelly_suggested=0.03,
        )
        # Spread: slightly lower raw score but full weight (1.0)
        spread = _entry(
            market="spreads",
            best_odds=-110,
            consensus_prob=0.35,
            alpha_score=72,  # lower alpha
            kelly_suggested=0.025,
        )
        for e in [dog, spread]:
            enrich_entry(e)

        # Check raw scores: dog has higher raw
        assert dog["hybrid_score_raw"] > spread["hybrid_score_raw"]
        # Check weighted scores: spread should be higher due to 1.0 vs 0.75
        assert spread["hybrid_score"] > dog["hybrid_score"]

        # Hybrid mode: spread should rank first
        candidates = [dog, spread]
        rank_candidates(candidates, mode="hybrid")
        assert candidates[0] is spread

    def test_hit_mode_ignores_market_weight(self):
        """Hit mode ordering is probability-first; market weight doesn't
        override probability ordering."""
        # High prob but penalized market
        high_prob = _entry(
            market="moneyline",
            best_odds=200,
            consensus_prob=0.70,
            alpha_score=60,
        )
        # Lower prob but full market weight
        low_prob = _entry(
            market="spreads",
            consensus_prob=0.50,
            alpha_score=70,
        )
        for e in [high_prob, low_prob]:
            enrich_entry(e)

        candidates = [low_prob, high_prob]
        rank_candidates(candidates, mode="hit")
        # Hit mode: higher prob wins regardless of market weight
        assert candidates[0] is high_prob

    def test_value_mode_ignores_market_weight(self):
        """Value mode ordering is EV-first; market weight doesn't affect it."""
        high_ev = _entry(
            market="moneyline",
            best_odds=200,
            consensus_prob=0.35,
            edge_ev_shrunk=0.05,
            ev_100=5.0,
        )
        low_ev = _entry(
            market="spreads",
            edge_ev_shrunk=0.02,
            ev_100=2.0,
        )
        for e in [high_ev, low_ev]:
            enrich_entry(e)

        candidates = [low_ev, high_ev]
        rank_candidates(candidates, mode="value")
        # Value mode: higher EV wins regardless of market weight
        assert candidates[0] is high_ev

    def test_hybrid_score_raw_preserved(self):
        """compute_hybrid_fields returns both hybrid_score_raw and hybrid_score."""
        e = _entry(market="moneyline", best_odds=-150, consensus_prob=0.60)
        fields = compute_hybrid_fields(e)
        assert "hybrid_score_raw" in fields
        assert "hybrid_score" in fields
        assert "market_weight" in fields
        # With a moneyline favorite, weight is 0.90
        assert fields["market_weight"] == pytest.approx(0.90)
        assert fields["hybrid_score"] == pytest.approx(
            fields["hybrid_score_raw"] * 0.90, abs=0.001
        )

    def test_no_market_field_gives_weight_1(self):
        """Entry without market field gets weight 1.0 (backward compat)."""
        fields = compute_hybrid_fields(
            {"alpha_score": 80, "kelly_suggested": 0.025, "consensus_prob": 0.55}
        )
        assert fields["market_weight"] == pytest.approx(1.0)
        assert fields["hybrid_score"] == fields["hybrid_score_raw"]


# =====================================================================
# 4. test_effective_kelly_multiplier_by_confidence
# =====================================================================


class TestEffectiveKellyMultiplier:
    """Verify kelly_effective = kelly_raw * multiplier."""

    def test_high_confidence_multiplier(self):
        e = _entry(kelly_base=0.10, confidence_label="High")
        result = compute_kelly_effective(e)
        assert result["kelly_multiplier"] == pytest.approx(1.00)
        assert result["kelly_effective"] == pytest.approx(0.10)

    def test_medium_confidence_multiplier(self):
        e = _entry(kelly_base=0.10, confidence_label="Medium")
        result = compute_kelly_effective(e)
        assert result["kelly_multiplier"] == pytest.approx(0.70)
        assert result["kelly_effective"] == pytest.approx(0.07)

    def test_low_confidence_multiplier(self):
        e = _entry(kelly_base=0.10, confidence_label="Low")
        result = compute_kelly_effective(e)
        assert result["kelly_multiplier"] == pytest.approx(0.40)
        assert result["kelly_effective"] == pytest.approx(0.04)

    def test_kelly_raw_equals_kelly_base(self):
        """kelly_raw should match the existing kelly_base field."""
        e = _entry(kelly_base=0.05, confidence_label="High")
        result = compute_kelly_effective(e)
        assert result["kelly_raw"] == pytest.approx(0.05)

    def test_kelly_effective_clamped_at_cap(self):
        """kelly_effective cannot exceed the 0.25 cap."""
        e = _entry(kelly_base=0.25, confidence_label="High")
        result = compute_kelly_effective(e)
        assert result["kelly_effective"] <= 0.25

    def test_kelly_effective_non_negative(self):
        """kelly_effective is always >= 0."""
        e = _entry(kelly_base=0.0, confidence_label="High")
        result = compute_kelly_effective(e)
        assert result["kelly_effective"] >= 0.0

    def test_unknown_confidence_uses_low(self):
        """Unknown confidence_label falls back to Low multiplier."""
        e = _entry(kelly_base=0.10, confidence_label="Unknown")
        result = compute_kelly_effective(e)
        assert result["kelly_multiplier"] == pytest.approx(0.40)

    def test_enrich_entry_adds_kelly_fields(self):
        """enrich_entry populates kelly_raw, kelly_multiplier, kelly_effective."""
        e = _entry()
        enrich_entry(e)
        assert "kelly_raw" in e
        assert "kelly_multiplier" in e
        assert "kelly_effective" in e
        assert e["kelly_effective"] == pytest.approx(
            e["kelly_raw"] * e["kelly_multiplier"], abs=1e-6
        )

    def test_kelly_effective_formula(self):
        """kelly_effective = kelly_base * multiplier for each label."""
        for label, mult in [("High", 1.0), ("Medium", 0.7), ("Low", 0.4)]:
            e = _entry(kelly_base=0.08, confidence_label=label)
            result = compute_kelly_effective(e)
            assert result["kelly_effective"] == pytest.approx(
                0.08 * mult, abs=1e-6
            ), f"Failed for {label}"


# =====================================================================
# 5. test_effective_kelly_does_not_change_pick_selection
# =====================================================================


class TestEffectiveKellyDoesNotChangePickSelection:
    """Verify that kelly_effective only affects sizing, not ranking."""

    def test_hybrid_ranking_unchanged_by_kelly_multiplier(self):
        """Top pick under hybrid mode should not change when only
        kelly_multiplier changes (kelly_effective is for sizing only)."""
        # Create two entries: one clearly better by hybrid criteria
        better = _entry(
            market="spreads",
            alpha_score=85,
            kelly_suggested=0.04,
            consensus_prob=0.60,
            quality_score=85,
            kelly_base=0.08,
            selection="Better",
        )
        worse = _entry(
            market="spreads",
            alpha_score=50,
            kelly_suggested=0.01,
            consensus_prob=0.40,
            quality_score=60,
            kelly_base=0.15,  # Higher kelly_base but worse everything else
            selection="Worse",
        )
        for e in [better, worse]:
            enrich_entry(e)

        # Verify top pick: better entry should always rank first in hybrid
        candidates = [worse, better]
        rank_candidates(candidates, mode="hybrid")
        assert candidates[0] is better
        assert candidates[0].get("selection") == "Better"

        # Even though "Worse" has higher kelly_effective, it doesn't affect rank
        assert worse["kelly_effective"] > better["kelly_effective"]

    def test_hit_mode_unchanged_by_kelly_multiplier(self):
        """Hit mode ranks by probability, not kelly_effective."""
        high_prob = _entry(
            consensus_prob=0.70, kelly_base=0.02,
            alpha_score=60, market="spreads",
        )
        low_prob = _entry(
            consensus_prob=0.40, kelly_base=0.15,
            alpha_score=70, market="spreads",
        )
        for e in [high_prob, low_prob]:
            enrich_entry(e)

        candidates = [low_prob, high_prob]
        rank_candidates(candidates, mode="hit")
        assert candidates[0] is high_prob

    def test_value_mode_unchanged_by_kelly_multiplier(self):
        """Value mode ranks by edge_ev_shrunk, not kelly_effective."""
        high_ev = _entry(
            edge_ev_shrunk=0.05, ev_100=5.0, kelly_base=0.02,
            market="spreads",
        )
        low_ev = _entry(
            edge_ev_shrunk=0.01, ev_100=1.0, kelly_base=0.15,
            market="spreads",
        )
        for e in [high_ev, low_ev]:
            enrich_entry(e)

        candidates = [low_ev, high_ev]
        rank_candidates(candidates, mode="value")
        assert candidates[0] is high_ev


# =====================================================================
# Supplementary: market_tier_report
# =====================================================================


class TestMarketTierReport:
    """Verify the market breakdown report."""

    def test_basic_report(self):
        recs = [
            _make_rec(market="spreads", edge_pct=3.0, edge_ev_shrunk=0.03,
                      quality_score=75, selection="S1"),
            _make_rec(market="moneyline", best_odds=-150, consensus_prob=0.60,
                      edge_pct=3.0, edge_ev_shrunk=0.03,
                      quality_score=75, selection="ML1"),
            _make_rec(market="moneyline", best_odds=200, consensus_prob=0.35,
                      edge_pct=3.0, edge_ev_shrunk=0.03,
                      quality_score=75, selection="ML2"),
        ]
        assign_tiers(recs, method="quantile", min_candidates=1)
        report = market_tier_report(recs)

        assert "by_market" in report
        assert "by_tier" in report
        assert "spreads" in report["by_market"]
        assert report["by_market"]["spreads"]["count"] == 1
        mw = report["by_market"]["spreads"]["avg_market_weight"]
        assert mw == pytest.approx(1.00)

    def test_avg_market_weight_per_tier(self):
        """by_tier should have avg_market_weight."""
        recs = [
            _make_rec(market="spreads", edge_pct=3.0, edge_ev_shrunk=0.03,
                      quality_score=75),
        ]
        assign_tiers(recs, method="quantile", min_candidates=1)
        report = market_tier_report(recs)
        for tier in (TIER_1, TIER_2, STAY_AWAY):
            assert "avg_market_weight" in report["by_tier"][tier]


# =====================================================================
# Supplementary: determinism
# =====================================================================


class TestDeterminism:
    """Verify all new features are deterministic."""

    def test_market_weight_deterministic(self):
        e = _entry(market="moneyline", best_odds=-150, consensus_prob=0.60)
        w1 = market_weight(e)
        w2 = market_weight(e)
        assert w1 == w2

    def test_hybrid_score_deterministic(self):
        e = _entry(market="spreads")
        f1 = compute_hybrid_fields(e)
        f2 = compute_hybrid_fields(e)
        assert f1["hybrid_score"] == f2["hybrid_score"]
        assert f1["hybrid_score_raw"] == f2["hybrid_score_raw"]

    def test_kelly_effective_deterministic(self):
        e = _entry(kelly_base=0.10, confidence_label="Medium")
        r1 = compute_kelly_effective(e)
        r2 = compute_kelly_effective(e)
        assert r1["kelly_effective"] == r2["kelly_effective"]
