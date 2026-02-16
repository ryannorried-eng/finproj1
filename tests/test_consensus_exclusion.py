"""Golden tests for consensus-excluding-book logic.

Verifies that consensus probability excludes the evaluated sportsbook,
uses median for n_excl < 6 and trimmed mean for n_excl >= 6, and that
edge/EV formulas are computed from the exclusion consensus.
"""

from __future__ import annotations

from datetime import datetime
from statistics import median

from line_tracker.best_bets import (
    _remove_vig,
    _trimmed_mean,
    consensus_prob_excluding_book,
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


# ── 1) consensus_prob_excluding_book unit tests ──────────────────────


class TestConsensusProbExcludingBook:
    def test_excludes_specified_book(self):
        """The excluded book's prob must not appear in the consensus input."""
        probs = [("A", 0.55), ("B", 0.50), ("C", 0.60)]
        p_excl, n_excl, _ = consensus_prob_excluding_book(probs, "B")
        # B (0.50) excluded → remaining = [0.55, 0.60]
        assert n_excl == 2
        assert abs(p_excl - median([0.55, 0.60])) < 1e-10

    def test_n_excl_lt_6_uses_median(self):
        """When n_excl < 6, should use median."""
        probs = [("A", 0.50), ("B", 0.55), ("C", 0.60), ("D", 0.52)]
        _, n_excl, method = consensus_prob_excluding_book(probs, "A")
        assert n_excl == 3
        assert method == "median"

    def test_n_excl_ge_6_uses_trimmed_mean(self):
        """When n_excl >= 6, should use trimmed mean."""
        probs = [
            ("A", 0.50), ("B", 0.51), ("C", 0.52),
            ("D", 0.53), ("E", 0.54), ("F", 0.55), ("G", 0.56),
        ]
        p_excl, n_excl, method = consensus_prob_excluding_book(probs, "A")
        assert n_excl == 6
        assert method == "trimmed_mean"
        # Remaining after excluding A: [0.51, 0.52, 0.53, 0.54, 0.55, 0.56]
        # Trimmed mean drops lowest (0.51) and highest (0.56)
        # Core = [0.52, 0.53, 0.54, 0.55]
        expected = (0.52 + 0.53 + 0.54 + 0.55) / 4.0
        assert abs(p_excl - expected) < 1e-10

    def test_all_books_equal_excl_equals_incl(self):
        """When all books post the same prob, p_excl == p_incl."""
        probs = [("A", 0.55), ("B", 0.55), ("C", 0.55)]
        p_excl, _, _ = consensus_prob_excluding_book(probs, "A")
        p_incl = median([0.55, 0.55, 0.55])
        assert abs(p_excl - p_incl) < 1e-10

    def test_outlier_excluded_changes_consensus(self):
        """When the outlier book is excluded, consensus shifts away from it."""
        probs = [
            ("A", 0.50), ("B", 0.55), ("Outlier", 0.90),
        ]
        p_incl = median([0.50, 0.55, 0.90])  # = 0.55
        p_excl, _, _ = consensus_prob_excluding_book(probs, "Outlier")
        # Without the outlier, consensus = median([0.50, 0.55]) = 0.525
        assert abs(p_excl - 0.525) < 1e-10
        # The outlier was pulling consensus up; excluding it lowers it
        assert p_excl < p_incl

    def test_single_book_fallback(self):
        """If only 1 book total and it's excluded, fallback to all-book median."""
        probs = [("A", 0.60)]
        p_excl, n_excl, method = consensus_prob_excluding_book(probs, "A")
        # Fallback: use all probs
        assert abs(p_excl - 0.60) < 1e-10
        assert n_excl == 1
        assert method == "median"

    def test_book_not_present_uses_all(self):
        """If exclude_book isn't in the list, all books are used."""
        probs = [("A", 0.50), ("B", 0.55), ("C", 0.60)]
        p_excl, n_excl, _ = consensus_prob_excluding_book(probs, "Z")
        assert n_excl == 3
        assert abs(p_excl - median([0.50, 0.55, 0.60])) < 1e-10


# ── 2) _trimmed_mean unit tests ──────────────────────────────────────


class TestTrimmedMean:
    def test_empty_returns_zero(self):
        assert _trimmed_mean([]) == 0.0

    def test_single_value(self):
        assert _trimmed_mean([0.55]) == 0.55

    def test_two_values_uses_median(self):
        """< 3 values → falls back to median."""
        vals = [0.50, 0.60]
        assert _trimmed_mean(vals) == median(vals)

    def test_three_values_trims_one_each_side(self):
        """Exactly 3 → drop lowest and highest, average the core (= middle)."""
        vals = [0.40, 0.55, 0.80]
        # core = [0.55]
        assert abs(_trimmed_mean(vals) - 0.55) < 1e-10

    def test_six_values_trims_one_each_side(self):
        vals = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55]
        # core = [0.51, 0.52, 0.53, 0.54]
        expected = (0.51 + 0.52 + 0.53 + 0.54) / 4.0
        assert abs(_trimmed_mean(vals) - expected) < 1e-10


# ── 3) End-to-end: recommend_best_bets uses exclusion ───────────────


class TestRecommendExclusion:
    def test_consensus_excludes_best_sportsbook_moneyline(self):
        """consensus_prob on the rec should exclude the best_sportsbook."""
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -140, 120),
            _ml_line("C", -145, 125),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        home_rec = next(r for r in recs if r.side == "home")

        # Best home line = B (-140 is worst for home, actually -150 is most
        # negative = biggest favourite)
        # max(lines, key=ln.home_value) picks -140 (highest value = B)
        # B is best_sportsbook for home
        assert home_rec.best_sportsbook == "B"

        # Consensus should exclude B
        vig_free = {}
        for ln in lines:
            ph, _ = _remove_vig(ln.home_value, ln.away_value)
            vig_free[ln.sportsbook] = ph

        remaining = [vig_free["A"], vig_free["C"]]
        expected = median(remaining)
        assert abs(home_rec.consensus_prob - expected) < 0.01

    def test_p_incl_is_inclusive_consensus(self):
        """p_incl should be the weighted-median of ALL books (inclusive)."""
        lines = [
            _ml_line("DraftKings", -150, 130),
            _ml_line("FanDuel", -140, 120),
            _ml_line("Caesars", -145, 125),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            # p_incl should be close to the unweighted (all equal weight)
            assert abs(r.p_incl - r.unweighted_consensus_prob) < 0.001

    def test_books_used_excl_field(self):
        """books_used_excl should be one less than books_used_count (minus best)."""
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -140, 120),
            _ml_line("C", -145, 125),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            assert r.books_used_excl == r.books_used_count - 1

    def test_consensus_method_field(self):
        """consensus_method should be 'median' when < 6 books."""
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -140, 120),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            assert r.consensus_method == "median"

    def test_all_equal_exclusion_no_change(self):
        """All books post identical odds — exclusion doesn't change p."""
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -150, 130),
            _ml_line("C", -150, 130),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            assert abs(r.consensus_prob - r.p_incl) < 0.001


# ── 4) End-to-end: compute_standouts uses per-book exclusion ────────


class TestStandoutsExclusion:
    def _entries(self):
        return [
            {"event": "G", "market": "ml", "selection": "H",
             "sportsbook": "A", "odds": -150, "line": None, "sport": "nba"},
            {"event": "G", "market": "ml", "selection": "H",
             "sportsbook": "B", "odds": -120, "line": None, "sport": "nba"},
            {"event": "G", "market": "ml", "selection": "H",
             "sportsbook": "C", "odds": -140, "line": None, "sport": "nba"},
        ]

    def test_each_standout_excludes_own_book(self):
        """Each standout's consensus_prob should exclude its own sportsbook."""
        results = compute_standouts(self._entries(), stake=100)
        for r in results:
            # books_used_excl should be total_books - 1
            assert r["books_used_excl"] == 2

    def test_consensus_differs_per_book(self):
        """Different books should get different exclusion consensus values."""
        results = compute_standouts(self._entries(), stake=100)
        consensuses = {r["sportsbook"]: r["consensus_prob"] for r in results}
        # A, B, C each get a different consensus (since they're excluded)
        unique_values = set(consensuses.values())
        assert len(unique_values) == 3

    def test_p_incl_same_for_all_books_in_group(self):
        """p_incl (inclusive median) should be the same for all books in a group."""
        results = compute_standouts(self._entries(), stake=100)
        p_incls = {r["p_incl"] for r in results}
        assert len(p_incls) == 1

    def test_outlier_excluded_increases_edge_for_others(self):
        """Excluding an outlier book from consensus should increase edge
        for non-outlier books (since the outlier was pulling consensus
        toward the wrong direction)."""
        entries = [
            {"event": "G", "market": "ml", "selection": "H",
             "sportsbook": "Normal1", "odds": -150, "line": None},
            {"event": "G", "market": "ml", "selection": "H",
             "sportsbook": "Normal2", "odds": -150, "line": None},
            {"event": "G", "market": "ml", "selection": "H",
             "sportsbook": "Outlier", "odds": +200, "line": None},
        ]
        results = compute_standouts(entries, stake=100)
        outlier_result = next(r for r in results if r["sportsbook"] == "Outlier")
        normal_result = next(r for r in results if r["sportsbook"] == "Normal1")

        # Outlier's consensus excludes itself → consensus = median of Normal1, Normal2
        # = implied_prob(-150) ≈ 0.6
        # Outlier's book_prob = implied_prob(+200) ≈ 0.333
        # edge = 0.6 - 0.333 = ~0.267 (very positive = outlier gets a great "edge"
        # because consensus from other books says the event is likely)
        assert outlier_result["edge"] > normal_result["edge"]

    def test_debug_fields_present(self):
        """All debug fields should be present in standout results."""
        results = compute_standouts(self._entries(), stake=100)
        for r in results:
            assert "p_incl" in r
            assert "books_used_excl" in r
            assert "consensus_method" in r

    def test_all_equal_odds_no_change(self):
        """When all books have the same odds, exclusion doesn't change consensus."""
        entries = [
            {"event": "G", "market": "ml", "selection": "H",
             "sportsbook": "A", "odds": -150, "line": None},
            {"event": "G", "market": "ml", "selection": "H",
             "sportsbook": "B", "odds": -150, "line": None},
            {"event": "G", "market": "ml", "selection": "H",
             "sportsbook": "C", "odds": -150, "line": None},
        ]
        results = compute_standouts(entries, stake=100)
        for r in results:
            assert abs(r["consensus_prob"] - r["p_incl"]) < 0.001
            assert abs(r["edge"]) < 0.001
