"""Model-validation tests for betting math, CLV, and ranking consistency.

Covers:
  1) Invariants — odds conversions, EV/edge/Kelly, CLV sign conventions
  2) Golden cases — hand-verified deterministic scenarios + full odds x prob grid
  3) Ranking/explanation consistency — 6-book deterministic slate
  4) Fuzz/random — lightweight randomized invariant checks

All expected values are computed independently (no production helpers in
expected-value computation) to catch formula drift.
"""

from __future__ import annotations

import math
import random

import pytest

from line_tracker.core.clv import compute_clv_metrics
from line_tracker.core.math import (
    american_to_decimal,
    decimal_to_american,
    ev_per_dollar,
    implied_probability,
    kelly_fraction,
    kelly_suggested,
)

# =====================================================================
# Independent helpers — intentionally NOT calling production code.
# =====================================================================


def _ind_dec(odds):
    return (odds / 100.0 + 1.0) if odds >= 0 else (100.0 / abs(odds) + 1.0)


def _ind_ip(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return 0.5


def _ind_ev(prob, odds):
    d = _ind_dec(odds)
    return prob * (d - 1.0) - (1.0 - prob)


def _ind_ev100(prob, odds):
    return 100.0 * _ind_ev(prob, odds)


def _ind_edge_pct(prob, odds):
    d = _ind_dec(odds)
    return 100.0 * (prob - 1.0 / d)


def _ind_kelly(p, dec, cap=0.25):
    if dec <= 1.0 or p <= 0.0:
        return 0.0
    raw = (p * dec - 1.0) / (dec - 1.0)
    return min(max(raw, 0.0), cap)


# =====================================================================
# 1) INVARIANT TESTS
# =====================================================================


class TestOddsConversionInvariants:
    """Odds conversion roundtrip and domain invariants."""

    ODDS = [
        -850, -400, -300, -200, -150, -110,
        +100, +110, +150, +200, +250, +400, +600, +1200,
    ]

    @pytest.mark.parametrize("american", ODDS)
    def test_american_decimal_roundtrip(self, american):
        dec = american_to_decimal(american)
        back = decimal_to_american(dec)
        assert abs(back - american) < 1.0

    @pytest.mark.parametrize("american", ODDS)
    def test_decimal_greater_than_one(self, american):
        assert american_to_decimal(american) > 1.0

    @pytest.mark.parametrize("american", ODDS)
    def test_implied_prob_equals_one_over_decimal(self, american):
        ip = implied_probability(american)
        dec = american_to_decimal(american)
        assert abs(ip - 1.0 / dec) < 1e-9

    @pytest.mark.parametrize("american", ODDS)
    def test_implied_prob_in_zero_one(self, american):
        ip = implied_probability(american)
        assert 0.0 < ip < 1.0

    @pytest.mark.parametrize("american", ODDS)
    def test_breakeven_matches_implied(self, american):
        dec = american_to_decimal(american)
        assert abs(1.0 / dec - implied_probability(american)) < 1e-9


class TestEvEdgeInvariants:
    """EV and edge metric invariants."""

    ODDS = [-850, -400, -200, -150, -110, +100, +150, +200, +300, +600]

    @pytest.mark.parametrize("american", ODDS)
    def test_ev_zero_at_breakeven(self, american):
        be = 1.0 / american_to_decimal(american)
        assert abs(ev_per_dollar(be, american)) < 1e-9

    @pytest.mark.parametrize("american", ODDS)
    def test_ev100_zero_at_breakeven(self, american):
        be = 1.0 / american_to_decimal(american)
        assert abs(100.0 * ev_per_dollar(be, american)) < 1e-7

    @pytest.mark.parametrize("american", ODDS)
    def test_edge_pct_zero_at_breakeven(self, american):
        dec = american_to_decimal(american)
        be = 1.0 / dec
        assert abs(100.0 * (be - 1.0 / dec)) < 1e-9

    def test_monotonicity_better_price_higher_ev(self):
        p = 0.55
        assert (
            ev_per_dollar(p, +150)
            > ev_per_dollar(p, +100)
            > ev_per_dollar(p, -110)
        )

    def test_monotonicity_higher_prob_higher_ev(self):
        assert (
            ev_per_dollar(0.60, -110)
            > ev_per_dollar(0.50, -110)
            > ev_per_dollar(0.40, -110)
        )

    @pytest.mark.parametrize("prob", [0.01, 0.1, 0.5, 0.9, 0.99])
    @pytest.mark.parametrize("american", [-300, -110, +100, +200, +600])
    def test_ev_no_nan(self, prob, american):
        ev = ev_per_dollar(prob, american)
        assert not math.isnan(ev)
        assert math.isfinite(ev)

    @pytest.mark.parametrize("prob", [0.01, 0.1, 0.5, 0.9, 0.99])
    @pytest.mark.parametrize("american", [-300, -110, +100, +200, +600])
    def test_edge_no_nan(self, prob, american):
        dec = american_to_decimal(american)
        edge = prob - 1.0 / dec
        assert not math.isnan(edge)
        assert math.isfinite(edge)


class TestKellyInvariants:
    """Kelly criterion invariants."""

    ODDS = [-200, -150, -110, +100, +150, +200, +300, +600]

    @pytest.mark.parametrize("american", ODDS)
    def test_kelly_zero_when_ev_nonpositive(self, american):
        """Kelly should be 0 when prob <= breakeven (EV <= 0)."""
        dec = american_to_decimal(american)
        be = 1.0 / dec
        # At breakeven
        assert kelly_fraction(be, dec) == 0.0
        # Below breakeven
        if be > 0.02:
            assert kelly_fraction(be - 0.01, dec) == 0.0

    @pytest.mark.parametrize("american", ODDS)
    def test_kelly_bounded_zero_to_cap(self, american):
        """Kelly always in [0, 0.25] for any valid prob."""
        dec = american_to_decimal(american)
        for p in [0.01, 0.3, 0.5, 0.7, 0.99]:
            kf = kelly_fraction(p, dec)
            assert 0.0 <= kf <= 0.25

    @pytest.mark.parametrize("american", ODDS)
    def test_kelly_no_nan(self, american):
        dec = american_to_decimal(american)
        for p in [0.0, 0.01, 0.5, 0.99, 1.0]:
            kf = kelly_fraction(p, dec)
            assert not math.isnan(kf)

    def test_kelly_suggested_scales_by_confidence(self):
        dec = 2.0  # +100
        p = 0.6
        kh = kelly_suggested(p, dec, "High")
        km = kelly_suggested(p, dec, "Medium")
        kl = kelly_suggested(p, dec, "Low")
        assert kh > km > kl > 0

    def test_kelly_increases_with_edge(self):
        dec = 2.0  # +100
        k1 = kelly_fraction(0.55, dec)
        k2 = kelly_fraction(0.60, dec)
        k3 = kelly_fraction(0.70, dec)
        assert k3 >= k2 >= k1 > 0


class TestCLVInvariants:
    """CLV sign convention and edge cases."""

    def test_beating_close_positive_clv(self):
        m = compute_clv_metrics(
            pick_dec=2.00, close_dec=1.80,
            pick_prob=0.50, close_prob=0.55,
        )
        assert m["clv_decimal"] > 0
        assert m["clv_prob"] > 0

    def test_losing_close_negative_clv(self):
        m = compute_clv_metrics(
            pick_dec=1.80, close_dec=2.00,
            pick_prob=0.55, close_prob=0.50,
        )
        assert m["clv_decimal"] < 0
        assert m["clv_prob"] < 0

    def test_zero_clv_when_identical(self):
        m = compute_clv_metrics(
            pick_dec=1.91, close_dec=1.91,
            pick_prob=0.52, close_prob=0.52,
        )
        assert m["clv_decimal"] == 0.0
        assert m["clv_prob"] == 0.0

    def test_close_prob_zero_is_valid_not_missing(self):
        m = compute_clv_metrics(
            pick_dec=1.91, close_dec=1.85,
            pick_prob=0.52, close_prob=0.0,
        )
        assert m["clv_prob"] == pytest.approx(-0.52, abs=0.001)

    def test_none_close_prob_falls_back_to_pick(self):
        m = compute_clv_metrics(
            pick_dec=1.91, close_dec=1.85,
            pick_prob=0.52, close_prob=None,
        )
        assert m["clv_prob"] == 0.0

    def test_clv_decimal_formula(self):
        m = compute_clv_metrics(
            pick_dec=2.10, close_dec=1.95,
            pick_prob=0.48, close_prob=0.51,
        )
        assert m["clv_decimal"] == pytest.approx(0.15, abs=0.001)

    def test_clv_prob_formula(self):
        m = compute_clv_metrics(
            pick_dec=2.10, close_dec=1.95,
            pick_prob=0.48, close_prob=0.51,
        )
        assert m["clv_prob"] == pytest.approx(0.03, abs=0.001)


# =====================================================================
# 2) GOLDEN CASES — hand-verified + parametrized grid
# =====================================================================

# (label, american, prob, exp_dec, exp_be, exp_ev, exp_ev100, exp_edge_pct, exp_kelly)
GOLDEN = [
    ("std-110_p50", -110, 0.50,
     1.909090909, 0.523809524, -0.045454545, -4.5454545, -2.3809524, 0.0),
    ("even+100_p50", +100, 0.50,
     2.0, 0.5, 0.0, 0.0, 0.0, 0.0),
    ("even+100_p55", +100, 0.55,
     2.0, 0.5, 0.10, 10.0, 5.0, 0.10),
    ("-110_p55", -110, 0.55,
     1.909090909, 0.523809524, 0.05, 5.0, 2.619048, 0.055),
    ("+150_p40_be", +150, 0.40,
     2.5, 0.4, 0.0, 0.0, 0.0, 0.0),
    ("+150_p50", +150, 0.50,
     2.5, 0.4, 0.25, 25.0, 10.0, 0.16667),
    ("-200_p60", -200, 0.60,
     1.5, 0.66667, -0.10, -10.0, -6.6667, 0.0),
    ("-200_p70", -200, 0.70,
     1.5, 0.66667, 0.05, 5.0, 3.3333, 0.10),
    ("-850_p90", -850, 0.90,
     1.117647059, 0.894736842, 0.005882353, 0.5882353, 0.526316, 0.05),
    ("+600_p20", +600, 0.20,
     7.0, 0.142857143, 0.40, 40.0, 5.714286, 0.066667),
    ("+600_p14", +600, 0.14,
     7.0, 0.142857143, -0.02, -2.0, -0.285714, 0.0),
    ("-110_p60", -110, 0.60,
     1.909090909, 0.523809524, 0.145454545, 14.5454545, 7.619048, 0.16),
    ("+200_p35", +200, 0.35,
     3.0, 0.333333, 0.05, 5.0, 1.6667, 0.025),
    ("+200_p50", +200, 0.50,
     3.0, 0.333333, 0.50, 50.0, 16.6667, 0.25),
    ("-300_p80", -300, 0.80,
     1.333333, 0.75, 0.066667, 6.6667, 5.0, 0.20),
    ("+100_p60", +100, 0.60,
     2.0, 0.5, 0.20, 20.0, 10.0, 0.20),
    ("-150_p65", -150, 0.65,
     1.666667, 0.60, 0.083333, 8.3333, 5.0, 0.125),
    ("+250_p30", +250, 0.30,
     3.5, 0.285714, 0.05, 5.0, 1.4286, 0.02),
    ("-400_p82", -400, 0.82,
     1.25, 0.80, 0.025, 2.5, 2.0, 0.10),
    ("+100_p45_neg", +100, 0.45,
     2.0, 0.5, -0.10, -10.0, -5.0, 0.0),
]


class TestGoldenCases:
    @pytest.mark.parametrize(
        "label,american,prob,exp_dec,exp_be,exp_ev,exp_ev100,exp_edge,exp_kelly",
        GOLDEN, ids=[c[0] for c in GOLDEN],
    )
    def test_decimal(self, label, american, prob, exp_dec, exp_be,
                     exp_ev, exp_ev100, exp_edge, exp_kelly):
        prod = american_to_decimal(american)
        assert abs(prod - exp_dec) < 1e-4
        assert abs(prod - _ind_dec(american)) < 1e-9

    @pytest.mark.parametrize(
        "label,american,prob,exp_dec,exp_be,exp_ev,exp_ev100,exp_edge,exp_kelly",
        GOLDEN, ids=[c[0] for c in GOLDEN],
    )
    def test_breakeven(self, label, american, prob, exp_dec, exp_be,
                       exp_ev, exp_ev100, exp_edge, exp_kelly):
        prod = implied_probability(american)
        assert abs(prod - exp_be) < 1e-4
        assert abs(prod - _ind_ip(american)) < 1e-9

    @pytest.mark.parametrize(
        "label,american,prob,exp_dec,exp_be,exp_ev,exp_ev100,exp_edge,exp_kelly",
        GOLDEN, ids=[c[0] for c in GOLDEN],
    )
    def test_ev(self, label, american, prob, exp_dec, exp_be,
                exp_ev, exp_ev100, exp_edge, exp_kelly):
        prod = ev_per_dollar(prob, american)
        assert abs(prod - exp_ev) < 1e-4
        assert abs(prod - _ind_ev(prob, american)) < 1e-9

    @pytest.mark.parametrize(
        "label,american,prob,exp_dec,exp_be,exp_ev,exp_ev100,exp_edge,exp_kelly",
        GOLDEN, ids=[c[0] for c in GOLDEN],
    )
    def test_ev100(self, label, american, prob, exp_dec, exp_be,
                   exp_ev, exp_ev100, exp_edge, exp_kelly):
        assert abs(100.0 * ev_per_dollar(prob, american) - exp_ev100) < 0.01

    @pytest.mark.parametrize(
        "label,american,prob,exp_dec,exp_be,exp_ev,exp_ev100,exp_edge,exp_kelly",
        GOLDEN, ids=[c[0] for c in GOLDEN],
    )
    def test_edge_pct(self, label, american, prob, exp_dec, exp_be,
                      exp_ev, exp_ev100, exp_edge, exp_kelly):
        dec = american_to_decimal(american)
        assert abs(100.0 * (prob - 1.0 / dec) - exp_edge) < 0.01

    @pytest.mark.parametrize(
        "label,american,prob,exp_dec,exp_be,exp_ev,exp_ev100,exp_edge,exp_kelly",
        GOLDEN, ids=[c[0] for c in GOLDEN],
    )
    def test_kelly(self, label, american, prob, exp_dec, exp_be,
                   exp_ev, exp_ev100, exp_edge, exp_kelly):
        dec = american_to_decimal(american)
        prod = kelly_fraction(prob, dec)
        assert abs(prod - exp_kelly) < 0.005
        assert abs(prod - _ind_kelly(prob, dec)) < 1e-9


# --- Full parametrized grid: production == independent for every combo ---

GRID_ODDS = [-850, -400, -300, -200, -150, -110, +100, +150, +200, +250, +600, +1200]
GRID_PROBS = [0.14, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]


class TestGoldenGrid:
    """Full odds x prob grid: production output matches independent computation."""

    @pytest.mark.parametrize("odds", GRID_ODDS)
    def test_decimal_conversion(self, odds):
        assert abs(american_to_decimal(odds) - _ind_dec(odds)) < 1e-9

    @pytest.mark.parametrize("odds", GRID_ODDS)
    def test_breakeven_prob(self, odds):
        assert abs(implied_probability(odds) - _ind_ip(odds)) < 1e-9

    @pytest.mark.parametrize("odds", GRID_ODDS)
    @pytest.mark.parametrize("prob", GRID_PROBS)
    def test_ev_roi(self, odds, prob):
        assert abs(ev_per_dollar(prob, odds) - _ind_ev(prob, odds)) < 1e-9

    @pytest.mark.parametrize("odds", GRID_ODDS)
    @pytest.mark.parametrize("prob", GRID_PROBS)
    def test_ev_100(self, odds, prob):
        assert abs(
            100.0 * ev_per_dollar(prob, odds) - _ind_ev100(prob, odds)
        ) < 1e-7

    @pytest.mark.parametrize("odds", GRID_ODDS)
    @pytest.mark.parametrize("prob", GRID_PROBS)
    def test_edge_pct(self, odds, prob):
        assert abs(
            100.0 * (prob - 1.0 / american_to_decimal(odds))
            - _ind_edge_pct(prob, odds)
        ) < 1e-9

    @pytest.mark.parametrize("odds", GRID_ODDS)
    @pytest.mark.parametrize("prob", GRID_PROBS)
    def test_kelly_fraction(self, odds, prob):
        dec = american_to_decimal(odds)
        assert abs(kelly_fraction(prob, dec) - _ind_kelly(prob, dec)) < 1e-9


# =====================================================================
# 3) RANKING / EXPLANATION CONSISTENCY
# =====================================================================


class TestRankingConsistency:
    """Explanation edge_breakdown fields match BetRecommendation fields."""

    @pytest.fixture()
    def recs(self):
        from datetime import datetime

        from line_tracker.best_bets import recommend_best_bets
        from line_tracker.models import BettingLine, BetType

        # 6 books with varied odds for realistic consensus / outlier detection
        books_odds = [
            ("DraftKings", -155, 135),
            ("FanDuel", -150, 130),
            ("BetMGM", -145, 125),
            ("Caesars", -160, 140),
            ("PointsBet", -150, 130),
            ("BetRivers", -148, 128),
        ]
        lines = [
            BettingLine(
                sportsbook=b,
                sport="basketball_nba",
                event="TeamA @ TeamB",
                bet_type=BetType.MONEYLINE,
                home_team="TeamB",
                away_team="TeamA",
                home_value=float(home_odds),
                away_value=float(away_odds),
                timestamp=datetime(2025, 1, 1, 12, 0),
            )
            for b, home_odds, away_odds in books_odds
        ]
        return recommend_best_bets(lines, top_n=10)

    def test_every_rec_has_best_bet_result(self, recs):
        for r in recs:
            assert hasattr(r, "best_bet_result"), (
                f"Missing best_bet_result on {r.market}/{r.side}"
            )
            assert r.best_bet_result is not None

    def test_explanation_has_required_top_level_keys(self, recs):
        required_keys = {
            "consensus_method", "edge_breakdown", "ev_edge",
            "confidence_reasoning", "quality_factors", "outlier_info",
            "market_context", "recency", "kelly",
        }
        for r in recs:
            expl = r.best_bet_result.explanation
            missing = required_keys - expl.keys()
            assert not missing, f"Missing keys: {missing}"

    def test_edge_breakdown_has_required_keys(self, recs):
        required = {
            "consensus_prob", "breakeven_prob", "edge_pp",
            "edge_pct", "ev_roi", "ev_100",
        }
        for r in recs:
            eb = r.best_bet_result.explanation["edge_breakdown"]
            assert required.issubset(eb.keys())

    def test_edge_breakdown_matches_rec(self, recs):
        for r in recs:
            eb = r.best_bet_result.explanation["edge_breakdown"]
            assert abs(eb["consensus_prob"] - r.consensus_prob) < 0.001
            assert abs(eb["breakeven_prob"] - r.breakeven_prob) < 0.001
            assert abs(eb["edge_pp"] - r.edge_pp) < 0.001
            assert abs(eb["edge_pct"] - r.edge_pct) < 0.1
            assert abs(eb["ev_roi"] - r.ev_roi) < 0.001
            assert abs(eb["ev_100"] - r.ev_100) < 0.1

    def test_outliers_removed_consistent(self, recs):
        for r in recs:
            expected = max(0, r.total_books_count - r.books_used_count)
            oi = r.best_bet_result.explanation["outlier_info"]
            assert oi["outliers_removed"] == expected
            assert r.best_bet_result.outliers_removed == expected

    def test_kelly_explanation_matches_core(self, recs):
        for r in recs:
            ke = r.best_bet_result.explanation["kelly"]
            assert abs(ke["base"] - r.kelly_base) < 1e-6
            assert abs(ke["suggested"] - r.kelly_suggested) < 1e-6
            dec = american_to_decimal(r.best_odds)
            # consensus_prob_weighted is stored rounded to 4dp, so Kelly
            # recomputed from it may differ slightly from the original
            # (computed from full-precision p_cons_w).  Use 1e-3 tolerance.
            core_base = kelly_fraction(r.consensus_prob_weighted, dec)
            core_sugg = kelly_suggested(
                r.consensus_prob_weighted, dec, r.confidence,
            )
            assert abs(r.kelly_base - core_base) < 1e-3
            assert abs(r.kelly_suggested - core_sugg) < 1e-3

    def test_quality_factors_in_explanation(self, recs):
        for r in recs:
            qf = r.best_bet_result.explanation["quality_factors"]
            assert abs(qf["quality_score"] - r.quality_score) <= 1
            assert qf["quality_tier"] == r.quality_tier


# =====================================================================
# 4) FUZZ / RANDOM TESTS
# =====================================================================


def _valid_american(rng):
    """Generate a valid American odds value in [-1000,-101] U [101,2000]."""
    if rng.random() < 0.5:
        return rng.randint(-1000, -101)
    return rng.randint(101, 2000)


class TestFuzzOdds:
    N = 200

    def test_roundtrip_and_domain(self):
        rng = random.Random(42)
        for _ in range(self.N):
            am = _valid_american(rng)
            dec = american_to_decimal(am)
            assert dec > 1.0
            assert not math.isnan(dec)
            ip = implied_probability(am)
            assert 0 < ip < 1
            assert abs(ip - 1.0 / dec) < 1e-9
            back = decimal_to_american(dec)
            assert abs(back - am) < 1.5


class TestFuzzEv:
    N = 200

    def test_ev_invariants(self):
        rng = random.Random(123)
        for _ in range(self.N):
            am = _valid_american(rng)
            prob = rng.uniform(0.01, 0.99)
            ev = ev_per_dollar(prob, am)
            assert not math.isnan(ev)
            assert math.isfinite(ev)
            # Production matches independent
            assert abs(ev - _ind_ev(prob, am)) < 1e-12
            # Breakeven -> EV=0
            be = 1.0 / american_to_decimal(am)
            assert abs(ev_per_dollar(be, am)) < 1e-9
            # Kelly in bounds and matches independent
            dec = american_to_decimal(am)
            kf = kelly_fraction(prob, dec)
            assert 0.0 <= kf <= 0.25
            assert not math.isnan(kf)
            assert abs(kf - _ind_kelly(prob, dec)) < 1e-9


class TestFuzzCLV:
    N = 100

    def test_clv_no_nan_formula_correct(self):
        rng = random.Random(456)
        for _ in range(self.N):
            pd, cd = rng.uniform(1.05, 10.0), rng.uniform(1.05, 10.0)
            pp, cp = rng.uniform(0.01, 0.99), rng.uniform(0.01, 0.99)
            m = compute_clv_metrics(
                pick_dec=pd, close_dec=cd, pick_prob=pp, close_prob=cp,
            )
            assert not math.isnan(m["clv_decimal"])
            assert not math.isnan(m["clv_prob"])
            assert abs(m["clv_decimal"] - round(pd - cd, 4)) < 1e-4
            assert abs(m["clv_prob"] - round(cp - pp, 4)) < 1e-4

    def test_clv_sign_logic(self):
        """Verify CLV sign: better pick -> positive, worse -> negative."""
        rng = random.Random(999)
        for _ in range(self.N):
            pp = rng.uniform(0.1, 0.9)
            # Case 1: pick is better (higher dec = better payout for bettor)
            pd = rng.uniform(2.0, 5.0)
            cd = pd - rng.uniform(0.01, 0.5)
            cp = pp + rng.uniform(0.01, min(0.1, 1.0 - pp - 0.01))
            m = compute_clv_metrics(
                pick_dec=pd, close_dec=cd, pick_prob=pp, close_prob=cp,
            )
            assert m["clv_decimal"] > 0
            assert m["clv_prob"] > 0

            # Case 2: pick is worse
            pd2 = rng.uniform(1.5, 3.0)
            cd2 = pd2 + rng.uniform(0.01, 0.5)
            cp2 = pp - rng.uniform(0.01, min(0.1, pp - 0.01))
            m2 = compute_clv_metrics(
                pick_dec=pd2, close_dec=cd2, pick_prob=pp, close_prob=cp2,
            )
            assert m2["clv_decimal"] < 0
            assert m2["clv_prob"] < 0

    def test_clv_none_close_prob_no_crash(self):
        rng = random.Random(789)
        for _ in range(50):
            pd, cd = rng.uniform(1.05, 10.0), rng.uniform(1.05, 10.0)
            pp = rng.uniform(0.01, 0.99)
            m = compute_clv_metrics(
                pick_dec=pd, close_dec=cd, pick_prob=pp, close_prob=None,
            )
            assert m["clv_prob"] == 0.0
            assert not math.isnan(m["clv_decimal"])
