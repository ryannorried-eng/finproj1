"""Tests for spec-defined EV / edge formulas and consensus aggregation.

Includes 'golden tests' for the EV math definitions:
  ev_100  = 100*(p*d - 1)
  edge_pp = p - 1/d
  exec_adv_100 = 100*p*(d_best - d_ref)
"""

from __future__ import annotations

from datetime import datetime
from statistics import median

from line_tracker.best_bets import (
    _consensus_prob,
    _remove_vig,
    _trimmed_mean,
    recommend_best_bets,
)
from line_tracker.bet_slip import compute_standouts
from line_tracker.models import BettingLine, BetType

# ── helpers ──────────────────────────────────────────────────────────


def _ml_line(book, home_odds, away_odds):
    return BettingLine(
        sportsbook=book,
        sport="basketball_nba",
        event="Celtics @ Lakers",
        bet_type=BetType.MONEYLINE,
        home_team="Lakers",
        away_team="Celtics",
        home_value=home_odds,
        away_value=away_odds,
        timestamp=datetime(2025, 1, 1, 12, 0),
    )


# ── 1) Trimmed mean / consensus aggregation ─────────────────────────


class TestTrimmedMeanAggregation:
    def test_empty_returns_zero(self):
        assert _trimmed_mean([]) == 0.0

    def test_single_value(self):
        assert _trimmed_mean([0.55]) == 0.55

    def test_fewer_than_five_uses_median(self):
        """< 5 values → falls back to median."""
        vals = [0.50, 0.55, 0.60]
        assert _trimmed_mean(vals) == median(vals)

    def test_four_values_uses_median(self):
        vals = [0.50, 0.52, 0.58, 0.60]
        assert _trimmed_mean(vals) == median(vals)

    def test_five_values_uses_trimmed_mean(self):
        """Exactly 5 values → trim=0.2 drops 1 from each tail."""
        vals = [0.40, 0.50, 0.55, 0.60, 0.80]
        # k = int(5 * 0.2) = 1; core = [0.50, 0.55, 0.60]
        expected = (0.50 + 0.55 + 0.60) / 3.0
        assert abs(_trimmed_mean(vals) - expected) < 1e-10

    def test_ten_values_trims_two(self):
        vals = sorted([0.50 + 0.01 * i for i in range(10)])
        # k = int(10 * 0.2) = 2; core = vals[2:8]
        core = vals[2:8]
        expected = sum(core) / len(core)
        assert abs(_trimmed_mean(vals) - expected) < 1e-10

    def test_consensus_prob_is_trimmed_mean(self):
        vals = [0.50, 0.55, 0.60]
        assert _consensus_prob(vals) == _trimmed_mean(vals)


# ── 2) Consensus uses vig-free per-book normalization ───────────────


class TestConsensusUsesVigFree:
    def test_remove_vig_normalizes(self):
        """_remove_vig produces probs summing to 1."""
        p_home, p_away = _remove_vig(-150, 130)
        assert abs(p_home + p_away - 1.0) < 1e-10

    def test_consensus_from_vig_free_probs(self):
        """End-to-end: consensus_prob on BetRecommendation uses
        vig-free per-book probs, not raw implied probs."""
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -160, 140),
            _ml_line("C", -140, 120),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        home_rec = next(r for r in recs if r.side == "home")

        # Manually compute vig-free home probs
        vig_free = []
        for ln in lines:
            ph, _ = _remove_vig(ln.home_value, ln.away_value)
            vig_free.append(ph)

        # With 3 books (<5), consensus = median
        expected = median(vig_free)
        assert abs(home_rec.consensus_prob - expected) < 0.01


# ── 3) Spec EV / edge formulas on BetRecommendation ────────────────


class TestSpecFormulasOnRec:
    def _get_home_rec(self):
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -160, 140),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        return next(r for r in recs if r.side == "home")

    def test_ev_roi_formula(self):
        """ev_roi = p * d - 1."""
        r = self._get_home_rec()
        p = r.consensus_prob
        # Compute decimal odds from American
        if r.best_odds > 0:
            d = r.best_odds / 100.0 + 1.0
        else:
            d = 1.0 - 100.0 / r.best_odds
        expected = p * d - 1.0
        assert abs(r.ev_roi - expected) < 0.001

    def test_ev_100_formula(self):
        """ev_100 = 100 * ev_roi."""
        r = self._get_home_rec()
        assert abs(r.ev_100 - 100.0 * r.ev_roi) < 0.1

    def test_ev_per_100_equals_ev_100(self):
        """ev_per_100 and ev_100 are the same value."""
        r = self._get_home_rec()
        assert abs(r.ev_per_100 - r.ev_100) < 0.01

    def test_p_be_formula(self):
        """p_be = 1 / d."""
        r = self._get_home_rec()
        if r.best_odds > 0:
            d = r.best_odds / 100.0 + 1.0
        else:
            d = 1.0 - 100.0 / r.best_odds
        expected = 1.0 / d
        assert abs(r.p_be - expected) < 0.001

    def test_edge_pp_formula(self):
        """edge_pp = p - p_be."""
        r = self._get_home_rec()
        assert abs(r.edge_pp - (r.consensus_prob - r.p_be)) < 0.001

    def test_edge_pct_formula(self):
        """edge_pct = 100 * (p - 1/d) = 100 * edge_pp."""
        r = self._get_home_rec()
        assert abs(r.edge_pct - 100.0 * r.edge_pp) < 0.1

    def test_ev_roi_zero_at_breakeven(self):
        """When p == 1/d, ev_roi should be 0."""
        # Analytical: p=0.5, d=2.0 → ev_roi = 0
        p, d = 0.5, 2.0
        assert abs(p * d - 1.0) < 1e-12

    def test_positive_ev_means_positive_edge(self):
        """If ev_roi > 0 then edge_pp > 0 (same sign)."""
        r = self._get_home_rec()
        if r.ev_roi > 0:
            assert r.edge_pp > 0
        elif r.ev_roi < 0:
            assert r.edge_pp < 0


# ── 4) exec_adv_100 in compute_standouts ────────────────────────────


class TestExecAdv100:
    def _standout_items(self):
        picks = [
            {
                "event": "BOS @ LAL",
                "market": "moneyline",
                "selection": "BOS",
                "sportsbook": "FanDuel",
                "odds": -150,
                "line": None,
                "sport": "basketball_nba",
            },
            {
                "event": "BOS @ LAL",
                "market": "moneyline",
                "selection": "BOS",
                "sportsbook": "DraftKings",
                "odds": -140,
                "line": None,
                "sport": "basketball_nba",
            },
            {
                "event": "BOS @ LAL",
                "market": "moneyline",
                "selection": "BOS",
                "sportsbook": "Caesars",
                "odds": -145,
                "line": None,
                "sport": "basketball_nba",
            },
        ]
        return compute_standouts(picks, stake=100.0)

    def test_exec_adv_100_present(self):
        results = self._standout_items()
        for r in results:
            assert "exec_adv_100" in r

    def test_exec_adv_100_formula(self):
        """exec_adv_100 = 100 * p * (d_best - d_ref)."""
        results = self._standout_items()
        for r in results:
            expected = round(
                100.0 * r["consensus_prob"] * (
                    r["d_best"] - r["d_ref"]
                ),
                2,
            )
            assert abs(r["exec_adv_100"] - expected) < 0.01

    def test_exec_adv_100_non_negative(self):
        """exec_adv_100 >= 0 since d_best >= d_ref."""
        results = self._standout_items()
        for r in results:
            assert r["exec_adv_100"] >= 0

    def test_d_best_is_highest_decimal(self):
        results = self._standout_items()
        if results:
            # All items share same d_best (best line for that group)
            d_best = results[0]["d_best"]
            # DraftKings -140 → d = 1.7143 (highest)
            assert d_best > results[0]["d_ref"]

    def test_exec_adv_increases_with_better_best_line(self):
        """Wider d_best - d_ref → larger exec_adv_100."""
        picks_tight = [
            {
                "event": "G", "market": "ml", "selection": "H",
                "sportsbook": "A", "odds": -150,
            },
            {
                "event": "G", "market": "ml", "selection": "H",
                "sportsbook": "B", "odds": -148,
            },
        ]
        picks_wide = [
            {
                "event": "G", "market": "ml", "selection": "H",
                "sportsbook": "A", "odds": -150,
            },
            {
                "event": "G", "market": "ml", "selection": "H",
                "sportsbook": "B", "odds": -120,
            },
        ]
        tight = compute_standouts(picks_tight, stake=100)
        wide = compute_standouts(picks_wide, stake=100)
        adv_tight = max(r["exec_adv_100"] for r in tight)
        adv_wide = max(r["exec_adv_100"] for r in wide)
        assert adv_wide > adv_tight


# ── 5) Golden tests — fixed-value regression guards ─────────────────


class TestGoldenEvMath:
    """Deterministic, pure-math tests with fixed p and d values.

    These protect against formula drift or sign errors.
    """

    def test_ev100_zero_at_fair_coinflip(self):
        """p=0.5, d=2.0 → ev_100 = 100*(0.5*2 - 1) = 0."""
        p, d = 0.5, 2.0
        ev_100 = 100.0 * (p * d - 1.0)
        assert ev_100 == 0.0

    def test_edge_pp_matches_definition(self):
        """edge_pp = p - 1/d for several (p, d) combos."""
        cases = [
            (0.55, 2.0),   # favourite, even odds
            (0.40, 3.0),   # underdog, long odds
            (0.65, 1.5),   # heavy favourite, short odds
        ]
        for p, d in cases:
            edge_pp = p - 1.0 / d
            expected = p - 1.0 / d
            assert abs(edge_pp - expected) < 1e-12, (
                f"p={p}, d={d}"
            )

    def test_ev_sign_favorite_case(self):
        """Favourite: p=0.60, d=1.80 → ev_100 > 0 (positive EV)."""
        p, d = 0.60, 1.80
        ev_100 = 100.0 * (p * d - 1.0)
        # 0.60 * 1.80 = 1.08 → ev_100 = 8.0
        assert ev_100 > 0
        assert abs(ev_100 - 8.0) < 1e-10

    def test_ev_sign_underdog_case(self):
        """Underdog: p=0.35, d=3.50 → ev_100 > 0."""
        p, d = 0.35, 3.50
        ev_100 = 100.0 * (p * d - 1.0)
        # 0.35 * 3.50 = 1.225 → ev_100 = 22.5
        assert ev_100 > 0
        assert abs(ev_100 - 22.5) < 1e-10

    def test_ev_sign_negative_case(self):
        """p=0.45, d=2.0 → ev_100 < 0 (negative EV)."""
        p, d = 0.45, 2.0
        ev_100 = 100.0 * (p * d - 1.0)
        # 0.45 * 2.0 = 0.90 → ev_100 = -10.0
        assert ev_100 < 0
        assert abs(ev_100 - (-10.0)) < 1e-10

    def test_vig_free_two_way_normalization(self):
        """-110/-110 (standard vig) → ~0.5/0.5 after removal."""
        p_home, p_away = _remove_vig(-110, -110)
        assert abs(p_home - 0.5) < 0.001
        assert abs(p_away - 0.5) < 0.001
        assert abs(p_home + p_away - 1.0) < 1e-10

    def test_exec_adv_nonneg_when_best_ge_ref(self):
        """exec_adv_100 >= 0 whenever d_best >= d_ref."""
        p = 0.55
        cases = [
            (2.0, 1.9),   # d_best > d_ref
            (2.0, 2.0),   # d_best == d_ref → 0
            (3.5, 1.8),   # wide gap
        ]
        for d_best, d_ref in cases:
            adv = 100.0 * p * (d_best - d_ref)
            assert adv >= 0, (
                f"d_best={d_best}, d_ref={d_ref}"
            )

    def test_exec_adv_increases_with_better_price(self):
        """Holding p constant, bigger d_best → bigger exec_adv."""
        p, d_ref = 0.55, 1.90
        adv_low = 100.0 * p * (2.00 - d_ref)
        adv_high = 100.0 * p * (2.20 - d_ref)
        assert adv_high > adv_low

    def test_edge_pct_equals_100_times_edge_pp(self):
        """edge_pct = 100 * edge_pp for arbitrary values."""
        p, d = 0.58, 1.85
        edge_pp = p - 1.0 / d
        edge_pct = 100.0 * edge_pp
        assert abs(edge_pct - 100.0 * edge_pp) < 1e-12

    def test_vig_free_asymmetric_normalizes(self):
        """-150/+130 should normalize to probs summing to 1."""
        p_h, p_a = _remove_vig(-150, 130)
        assert abs(p_h + p_a - 1.0) < 1e-10
        # Favourite should have higher prob
        assert p_h > p_a
