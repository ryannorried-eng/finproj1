"""Tests for EV-based edge math: weighted consensus, shrinkage, edge_z."""

from __future__ import annotations

from line_tracker.best_bets import (
    _MIN_EV_SIGMA,
    _SHRINKAGE_K,
    compute_edge_z,
    compute_ev_edge,
    compute_shrinkage,
    weighted_robust_consensus,
)


class TestEvEdgeZeroAtBreakeven:
    def test_ev_edge_is_zero_when_p_equals_breakeven(self):
        """edge_ev == 0 when p_cons = 1/d (breakeven)."""
        d = 2.0  # decimal odds
        p_cons = 1.0 / d  # 0.5 — breakeven
        assert compute_ev_edge(p_cons, d) == 0.0

    def test_ev_edge_is_zero_at_various_odds(self):
        for d in [1.5, 2.0, 3.0, 5.0, 10.0]:
            p = 1.0 / d
            ev = compute_ev_edge(p, d)
            assert abs(ev) < 1e-12, f"d={d}: expected 0, got {ev}"


class TestEvEdgeSignFavoriteUnderdog:
    def test_positive_ev_when_p_above_breakeven(self):
        """p_cons > 1/d ⇒ positive EV."""
        d = 2.0
        p = 0.55  # above 0.5 breakeven
        ev = compute_ev_edge(p, d)
        assert ev > 0

    def test_negative_ev_when_p_below_breakeven(self):
        """p_cons < 1/d ⇒ negative EV."""
        d = 2.0
        p = 0.45  # below 0.5 breakeven
        ev = compute_ev_edge(p, d)
        assert ev < 0

    def test_ev_scales_with_odds(self):
        """Higher decimal odds amplify the EV for the same edge in prob."""
        p = 0.55
        ev_low = compute_ev_edge(p, 1.5)  # short odds
        ev_high = compute_ev_edge(p, 3.0)  # long odds
        assert ev_high > ev_low

    def test_underdog_positive_ev(self):
        """Underdog at +200 (d=3.0) with p=0.40 has positive EV."""
        d = 3.0
        p = 0.40  # breakeven is 1/3 ≈ 0.333
        ev = compute_ev_edge(p, d)
        assert ev > 0
        assert abs(ev - (0.40 * 3.0 - 1.0)) < 1e-12


class TestWeightedConsensusRobustToOutlier:
    def test_cluster_with_outlier(self):
        """Outlier with high hold is down-weighted; consensus near cluster."""
        probs = [0.55, 0.55, 0.56, 0.54, 0.70]  # last is outlier
        holds = [2.0, 2.0, 2.0, 2.0, 8.0]  # high hold on outlier
        p_cons = weighted_robust_consensus(probs, holds)
        # Should be much closer to 0.55 cluster than to mean(probs)≈0.58
        assert abs(p_cons - 0.55) < 0.02
        assert p_cons < 0.58  # less than simple mean

    def test_all_same_probs(self):
        """When all probs are identical, consensus equals that value."""
        probs = [0.60, 0.60, 0.60]
        holds = [2.0, 3.0, 5.0]
        p_cons = weighted_robust_consensus(probs, holds)
        assert abs(p_cons - 0.60) < 1e-10

    def test_single_book(self):
        """Single book returns its probability."""
        assert weighted_robust_consensus([0.55], [3.0]) == 0.55

    def test_empty_list(self):
        """Empty input returns 0."""
        assert weighted_robust_consensus([], []) == 0.0

    def test_low_hold_books_weighted_higher(self):
        """Books with lower hold get more weight."""
        # Two books: sharp (low hold) says 0.55, retail (high hold) says 0.50
        probs = [0.55, 0.50]
        holds = [1.0, 8.0]  # first is sharp
        p_cons = weighted_robust_consensus(probs, holds)
        # Should be closer to 0.55 (sharp book)
        assert p_cons > 0.52


class TestShrinkageDecreaseEdgeWhenFewBooks:
    def test_shrinkage_reduces_with_fewer_books(self):
        """Same edge_ev but lower n_eff → more shrinkage."""
        ev = 0.05
        shrunk_many = compute_shrinkage(ev, n_eff=10.0)
        shrunk_few = compute_shrinkage(ev, n_eff=2.0)
        assert shrunk_many > shrunk_few

    def test_shrinkage_zero_n_eff(self):
        """n_eff=0 → edge_ev_shrunk = 0."""
        assert compute_shrinkage(0.05, n_eff=0.0) == 0.0

    def test_shrinkage_preserves_sign(self):
        """Negative edge stays negative after shrinkage."""
        result = compute_shrinkage(-0.03, n_eff=5.0)
        assert result < 0

    def test_shrinkage_approaches_identity_at_large_n(self):
        """With many books, shrinkage is minimal."""
        ev = 0.05
        shrunk = compute_shrinkage(ev, n_eff=100.0)
        # 100 / (100+5) = 0.952...
        assert abs(shrunk - ev * (100.0 / 105.0)) < 1e-10

    def test_shrinkage_formula(self):
        """Verify exact formula: ev * n_eff / (n_eff + k)."""
        ev, n = 0.04, 8.0
        expected = ev * (n / (n + _SHRINKAGE_K))
        assert abs(compute_shrinkage(ev, n) - expected) < 1e-12


class TestEdgeZUsesEvSigmaFloor:
    def test_tiny_sigma_uses_floor(self):
        """When ev_sigma is tiny, denominator is min_ev_sigma → finite z."""
        edge_ev_shrunk = 0.02
        ev_sigma = 0.0001  # well below floor
        z = compute_edge_z(edge_ev_shrunk, ev_sigma, min_ev_sigma=_MIN_EV_SIGMA)
        expected = edge_ev_shrunk / _MIN_EV_SIGMA
        assert abs(z - expected) < 1e-10

    def test_zero_sigma_uses_floor(self):
        """ev_sigma=0 → denominator = min_ev_sigma."""
        z = compute_edge_z(0.01, 0.0, min_ev_sigma=_MIN_EV_SIGMA)
        assert z == 0.01 / _MIN_EV_SIGMA

    def test_large_sigma_not_floored(self):
        """When ev_sigma > floor, use actual sigma."""
        z = compute_edge_z(0.02, 0.01, min_ev_sigma=_MIN_EV_SIGMA)
        assert abs(z - 2.0) < 1e-10

    def test_negative_edge_gives_negative_z(self):
        """Negative edge_ev_shrunk → negative edge_z."""
        z = compute_edge_z(-0.01, 0.005, min_ev_sigma=_MIN_EV_SIGMA)
        assert z < 0

    def test_custom_floor(self):
        """Custom min_ev_sigma parameter is respected."""
        z = compute_edge_z(0.01, 0.0, min_ev_sigma=0.005)
        assert abs(z - 2.0) < 1e-10
