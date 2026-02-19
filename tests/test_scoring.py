"""Tests for the shared scoring and ranking module."""

import pytest

from line_tracker.scoring import (
    RANKING_MODES,
    compute_alpha_fields,
    compute_hybrid_fields,
    enrich_entry,
    rank_candidates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    *,
    books_used: int = 7,
    market_hold_median: float = 5.0,
    edge_z: float = 2.0,
    edge_ev_shrunk: float = 0.02,
    agreement_score: float = 85.0,
    market_volatility_sigma: float = 0.005,
    consensus_prob: float = 0.55,
    kelly_suggested: float = 0.02,
    quality_score: int = 80,
    quality_tier: str = "Strong",
    ev_100: float = 3.0,
    alpha_score: int | None = None,
    best_odds: float | None = None,
    market: str | None = None,
) -> dict:
    """Build a minimal entry dict for scoring tests."""
    d: dict = {
        "books_used": books_used,
        "market_hold_median": market_hold_median,
        "edge_z": edge_z,
        "edge_ev_shrunk": edge_ev_shrunk,
        "agreement_score": agreement_score,
        "market_volatility_sigma": market_volatility_sigma,
        "consensus_prob": consensus_prob,
        "kelly_suggested": kelly_suggested,
        "quality_score": quality_score,
        "quality_tier": quality_tier,
        "ev_100": ev_100,
    }
    if alpha_score is not None:
        d["alpha_score"] = alpha_score
    if best_odds is not None:
        d["best_odds"] = best_odds
    if market is not None:
        d["market"] = market
    return d


# ── compute_alpha_fields ──────────────────────────────────────────────


class TestComputeAlphaFields:
    def test_returns_required_keys(self):
        result = compute_alpha_fields(_entry())
        assert "alpha_score" in result
        assert "alpha_label" in result
        assert "alpha_components" in result

    def test_score_in_range(self):
        result = compute_alpha_fields(_entry())
        assert 0 <= result["alpha_score"] <= 100

    def test_label_matches_score(self):
        # Strong alpha entry
        result = compute_alpha_fields(
            _entry(
                books_used=10,
                edge_z=2.5,
                edge_ev_shrunk=0.03,
                agreement_score=95,
                market_volatility_sigma=0.003,
                consensus_prob=0.60,
            )
        )
        assert result["alpha_label"] in ("Strong", "Neutral", "Weak")

    def test_does_not_mutate_entry(self):
        e = _entry()
        original_keys = set(e.keys())
        compute_alpha_fields(e)
        assert set(e.keys()) == original_keys

    def test_weak_for_thin_market(self):
        result = compute_alpha_fields(
            _entry(
                books_used=2,
                edge_z=0.5,
                edge_ev_shrunk=0.001,
                agreement_score=30,
                consensus_prob=0.15,
            )
        )
        assert result["alpha_label"] == "Weak"


# ── compute_hybrid_fields ─────────────────────────────────────────────


class TestComputeHybridFields:
    def test_returns_required_keys(self):
        result = compute_hybrid_fields(_entry(alpha_score=70))
        assert "hybrid_score" in result
        assert "hybrid_components" in result

    def test_score_in_unit_range(self):
        result = compute_hybrid_fields(_entry(alpha_score=70))
        assert 0.0 <= result["hybrid_score"] <= 1.0

    def test_zero_when_all_missing(self):
        result = compute_hybrid_fields({})
        assert result["hybrid_score"] == 0.0

    def test_weights_sum_to_one(self):
        result = compute_hybrid_fields(_entry(alpha_score=100))
        w = result["hybrid_components"]["weights"]
        assert abs(w["alpha"] + w["kelly"] + w["prob"] - 1.0) < 1e-9

    def test_does_not_mutate_entry(self):
        e = _entry(alpha_score=70)
        original_keys = set(e.keys())
        compute_hybrid_fields(e)
        assert set(e.keys()) == original_keys

    def test_known_value(self):
        """Exact calculation: alpha=80 -> 0.32, kelly=0.025/0.05=0.5 -> 0.20,
        prob=0.55 -> 0.11. Total = 0.63."""
        result = compute_hybrid_fields(
            {"alpha_score": 80, "kelly_suggested": 0.025, "consensus_prob": 0.55}
        )
        assert result["hybrid_score"] == pytest.approx(0.63, abs=0.001)

    def test_kelly_capped_at_one(self):
        """Kelly above cap should normalise to 1.0, not exceed."""
        result = compute_hybrid_fields(
            {"alpha_score": 100, "kelly_suggested": 0.10, "consensus_prob": 1.0}
        )
        # 0.4*1.0 + 0.4*1.0 + 0.2*1.0 = 1.0
        assert result["hybrid_score"] == pytest.approx(1.0, abs=0.001)


# ── rank_candidates mode="hybrid" ────────────────────────────────────


class TestRankHybridMode:
    def test_sorts_by_hybrid_score_desc(self):
        a = _entry(alpha_score=90, consensus_prob=0.60)
        b = _entry(alpha_score=40, consensus_prob=0.30)
        for e in [a, b]:
            enrich_entry(e)
        candidates = [b, a]
        rank_candidates(candidates, mode="hybrid")
        # a should rank higher (better alpha and prob)
        assert candidates[0] is a

    def test_tiebreaker_edge_ev_shrunk(self):
        """Same hybrid_score should tiebreak on edge_ev_shrunk."""
        a = _entry(alpha_score=70, edge_ev_shrunk=0.05)
        b = _entry(alpha_score=70, edge_ev_shrunk=0.02)
        for e in [a, b]:
            enrich_entry(e)
        # Force same hybrid_score
        a["hybrid_score"] = b["hybrid_score"] = 0.50
        candidates = [b, a]
        rank_candidates(candidates, mode="hybrid")
        assert candidates[0] is a  # higher edge_ev_shrunk first


# ── rank_candidates mode="hit" ───────────────────────────────────────


class TestRankHitMode:
    def test_favorite_over_longshot(self):
        """Hit mode should prefer the favorite (higher consensus_prob)
        over a longshot dog, even if the dog has slightly higher edge."""
        favorite = _entry(
            consensus_prob=0.65,
            alpha_score=70,
            edge_ev_shrunk=0.01,
            agreement_score=90,
            market_volatility_sigma=0.004,
        )
        longshot = _entry(
            consensus_prob=0.25,
            alpha_score=45,
            edge_ev_shrunk=0.03,
            agreement_score=70,
            market_volatility_sigma=0.012,
        )
        for e in [favorite, longshot]:
            enrich_entry(e)
        candidates = [longshot, favorite]
        rank_candidates(candidates, mode="hit")
        assert candidates[0] is favorite

    def test_demotes_negative_edge(self):
        """Negative edge_ev_shrunk should sort to the bottom."""
        good = _entry(consensus_prob=0.55, edge_ev_shrunk=0.01)
        bad = _entry(consensus_prob=0.80, edge_ev_shrunk=-0.01)
        for e in [good, bad]:
            enrich_entry(e)
        candidates = [bad, good]
        rank_candidates(candidates, mode="hit")
        assert candidates[0] is good

    def test_higher_agreement_preferred(self):
        """Among similar prob entries, higher agreement should rank first."""
        a = _entry(consensus_prob=0.55, agreement_score=95, edge_ev_shrunk=0.01)
        b = _entry(consensus_prob=0.55, agreement_score=60, edge_ev_shrunk=0.01)
        for e in [a, b]:
            enrich_entry(e)
        candidates = [b, a]
        rank_candidates(candidates, mode="hit")
        assert candidates[0] is a

    def test_lower_sigma_preferred(self):
        """Among identical prob/agreement/alpha entries, lower sigma wins."""
        a = _entry(
            consensus_prob=0.55,
            agreement_score=85,
            market_volatility_sigma=0.003,
            edge_ev_shrunk=0.01,
        )
        b = _entry(
            consensus_prob=0.55,
            agreement_score=85,
            market_volatility_sigma=0.015,
            edge_ev_shrunk=0.01,
        )
        for e in [a, b]:
            enrich_entry(e)
        # Force same alpha_score so the sigma tiebreaker activates
        a["alpha_score"] = b["alpha_score"] = 60
        candidates = [b, a]
        rank_candidates(candidates, mode="hit")
        assert candidates[0] is a


# ── rank_candidates mode="value" ──────────────────────────────────────


class TestRankValueMode:
    def test_highest_ev_first(self):
        a = _entry(edge_ev_shrunk=0.05, ev_100=5.0, quality_score=80)
        b = _entry(edge_ev_shrunk=0.02, ev_100=2.0, quality_score=90)
        for e in [a, b]:
            enrich_entry(e)
        candidates = [b, a]
        rank_candidates(candidates, mode="value")
        assert candidates[0] is a

    def test_tiebreaker_ev_100(self):
        a = _entry(edge_ev_shrunk=0.02, ev_100=3.0)
        b = _entry(edge_ev_shrunk=0.02, ev_100=1.0)
        for e in [a, b]:
            enrich_entry(e)
        candidates = [b, a]
        rank_candidates(candidates, mode="value")
        assert candidates[0] is a


# ── rank_candidates error handling ────────────────────────────────────


class TestRankCandidatesErrors:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown ranking mode"):
            rank_candidates([], mode="invalid")

    def test_empty_list_ok(self):
        result = rank_candidates([], mode="hybrid")
        assert result == []


# ── enrich_entry ──────────────────────────────────────────────────────


class TestEnrichEntry:
    def test_adds_all_fields(self):
        e = _entry()
        enrich_entry(e)
        assert "alpha_score" in e
        assert "alpha_label" in e
        assert "alpha_components" in e
        assert "hybrid_score" in e
        assert "hybrid_components" in e

    def test_returns_same_dict(self):
        e = _entry()
        result = enrich_entry(e)
        assert result is e


# ── RANKING_MODES constant ────────────────────────────────────────────


class TestRankingModes:
    def test_contains_all(self):
        assert "hybrid" in RANKING_MODES
        assert "hit" in RANKING_MODES
        assert "value" in RANKING_MODES


# ── Integration: hybrid score matches slate.compute_hybrid_score ──────


class TestHybridMatchesSlate:
    """Ensure the shared module produces identical results to the
    original ``slate.compute_hybrid_score``."""

    def test_identical_output(self):
        from line_tracker.slate import compute_hybrid_score

        e = _entry(alpha_score=75, kelly_suggested=0.03, consensus_prob=0.58)
        expected = compute_hybrid_score(dict(e))
        result = compute_hybrid_fields(e)
        assert result["hybrid_score"] == pytest.approx(expected, abs=1e-6)


# ── Integration: best bet entries get hybrid_score after enrichment ────


class TestBestBetEnrichment:
    """Test that BetRecommendation-derived entries gain hybrid_score."""

    def test_enriched_entry_has_hybrid_score(self):
        """Simulate the dashboard flow: rec → entry dict → enrich."""
        entry = _entry(
            books_used=6,
            edge_z=1.8,
            edge_ev_shrunk=0.015,
            agreement_score=80,
            market_volatility_sigma=0.006,
            consensus_prob=0.52,
            kelly_suggested=0.015,
            quality_score=75,
        )
        enrich_entry(entry)
        assert "hybrid_score" in entry
        assert entry["hybrid_score"] > 0

    def test_ranking_mode_changes_order(self):
        """Value mode and hit mode should produce different rankings
        when one candidate has high EV but low prob and another the reverse."""
        high_ev = _entry(
            edge_ev_shrunk=0.04,
            ev_100=4.0,
            consensus_prob=0.25,
            agreement_score=70,
            market_volatility_sigma=0.012,
        )
        high_prob = _entry(
            edge_ev_shrunk=0.01,
            ev_100=1.0,
            consensus_prob=0.70,
            agreement_score=90,
            market_volatility_sigma=0.003,
        )
        for e in [high_ev, high_prob]:
            enrich_entry(e)

        # Value mode: high_ev first
        value_order = rank_candidates([dict(high_prob), dict(high_ev)], mode="value")
        assert value_order[0]["edge_ev_shrunk"] == 0.04

        # Hit mode: high_prob first
        hit_order = rank_candidates([dict(high_ev), dict(high_prob)], mode="hit")
        assert hit_order[0]["consensus_prob"] == 0.70
