"""Tests for Kelly criterion sizing helpers."""

from datetime import datetime

import pytest

from line_tracker.best_bets import (
    _KELLY_CAP,
    _sizing_note,
    kelly_fraction,
    kelly_suggested,
    recommend_best_bets,
)
from line_tracker.models import BettingLine, BetType

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _ml_line(sportsbook: str, home_odds: float, away_odds: float) -> BettingLine:
    return BettingLine(
        sportsbook=sportsbook,
        sport="basketball_nba",
        event="Lakers @ Celtics",
        bet_type=BetType.MONEYLINE,
        home_team="Celtics",
        away_team="Lakers",
        home_value=home_odds,
        away_value=away_odds,
        timestamp=datetime(2025, 1, 1, 12, 0),
    )


# -------------------------------------------------------------------
# kelly_fraction
# -------------------------------------------------------------------

class TestKellyFraction:
    def test_known_values(self):
        """60% win prob at 2.0 decimal → (0.6*2 - 1) / (2 - 1) = 0.20."""
        f = kelly_fraction(0.60, 2.0)
        assert f == pytest.approx(0.20, abs=0.001)

    def test_fair_odds_returns_zero(self):
        """50% at 2.0 decimal → (1.0 - 1) / 1.0 = 0.0 → no edge."""
        f = kelly_fraction(0.50, 2.0)
        assert f == pytest.approx(0.0, abs=0.001)

    def test_negative_ev_returns_zero(self):
        """40% at 2.0 decimal → (0.8 - 1) / 1 = -0.2 → clipped to 0."""
        f = kelly_fraction(0.40, 2.0)
        assert f == 0.0

    def test_clipping_at_cap(self):
        """Very strong edge should be clipped at 25%."""
        # 90% win prob at 2.0 → (1.8 - 1) / 1 = 0.80, clipped to 0.25
        f = kelly_fraction(0.90, 2.0)
        assert f == _KELLY_CAP

    def test_custom_cap(self):
        """Custom cap should be respected."""
        f = kelly_fraction(0.60, 2.0, cap=0.10)
        assert f == pytest.approx(0.10, abs=0.001)

    def test_decimal_odds_at_one_returns_zero(self):
        """Odds of exactly 1.0 → zero-profit; denominator=0 → 0."""
        f = kelly_fraction(0.60, 1.0)
        assert f == 0.0

    def test_decimal_odds_below_one_returns_zero(self):
        """Invalid odds below 1.0 → 0."""
        f = kelly_fraction(0.60, 0.5)
        assert f == 0.0

    def test_zero_prob_returns_zero(self):
        f = kelly_fraction(0.0, 2.0)
        assert f == 0.0

    def test_slight_edge(self):
        """52% at 1.909 (-110 decimal) → small fraction."""
        f = kelly_fraction(0.52, 1.909)
        # (0.52 * 1.909 - 1) / (1.909 - 1) ≈ -0.0049 / 0.909 ≈ -0.005
        # Actually 0.52 * 1.909 = 0.99268 → neg → 0
        assert f == 0.0

    def test_moderate_edge(self):
        """55% at 1.909 → (1.04995 - 1) / 0.909 ≈ 0.055."""
        f = kelly_fraction(0.55, 1.909)
        assert 0.04 < f < 0.07


# -------------------------------------------------------------------
# kelly_suggested (with confidence multiplier)
# -------------------------------------------------------------------

class TestKellySuggested:
    def test_high_confidence_full_kelly(self):
        """High confidence → multiplier 1.0, so same as base."""
        base = kelly_fraction(0.60, 2.0)
        sugg = kelly_suggested(0.60, 2.0, "High")
        assert sugg == pytest.approx(base, abs=0.0001)

    def test_medium_confidence_half_kelly(self):
        """Medium confidence → multiplier 0.5."""
        base = kelly_fraction(0.60, 2.0)
        sugg = kelly_suggested(0.60, 2.0, "Medium")
        assert sugg == pytest.approx(base * 0.5, abs=0.0001)

    def test_low_confidence_quarter_kelly(self):
        """Low confidence → multiplier 0.25."""
        base = kelly_fraction(0.60, 2.0)
        sugg = kelly_suggested(0.60, 2.0, "Low")
        assert sugg == pytest.approx(base * 0.25, abs=0.0001)

    def test_unknown_confidence_uses_low(self):
        """Unknown confidence label falls back to 0.25."""
        base = kelly_fraction(0.60, 2.0)
        sugg = kelly_suggested(0.60, 2.0, "Unknown")
        assert sugg == pytest.approx(base * 0.25, abs=0.0001)

    def test_negative_ev_stays_zero(self):
        """No edge → base is 0, multiplier doesn't matter."""
        sugg = kelly_suggested(0.40, 2.0, "High")
        assert sugg == 0.0


# -------------------------------------------------------------------
# _sizing_note
# -------------------------------------------------------------------

class TestSizingNote:
    def test_zero_fraction(self):
        assert "skip" in _sizing_note(0.0).lower()

    def test_tiny_edge(self):
        note = _sizing_note(0.005)
        assert "tiny" in note.lower()

    def test_small_edge(self):
        note = _sizing_note(0.02)
        assert "small" in note.lower()

    def test_moderate_edge(self):
        note = _sizing_note(0.05)
        assert "moderate" in note.lower()

    def test_strong_edge(self):
        note = _sizing_note(0.10)
        assert "strong" in note.lower()

    def test_contains_percentage(self):
        note = _sizing_note(0.05)
        assert "5.0%" in note


# -------------------------------------------------------------------
# Integration: Kelly fields on BetRecommendation
# -------------------------------------------------------------------

class TestKellyOnRecommendation:
    def test_recommendation_has_kelly_fields(self):
        """recommend_best_bets populates kelly_base, kelly_suggested, sizing_note."""
        lines = [
            _ml_line("DraftKings", -150, 130),
            _ml_line("FanDuel", -140, 120),
            _ml_line("BetMGM", -160, 140),
        ]
        recs = recommend_best_bets(lines, top_n=6)
        assert len(recs) > 0
        for r in recs:
            assert hasattr(r, "kelly_base")
            assert hasattr(r, "kelly_suggested")
            assert hasattr(r, "sizing_note")
            assert isinstance(r.kelly_base, float)
            assert isinstance(r.kelly_suggested, float)
            assert isinstance(r.sizing_note, str)
            # Kelly base should be >= 0 and <= cap
            assert 0.0 <= r.kelly_base <= _KELLY_CAP
            assert 0.0 <= r.kelly_suggested <= _KELLY_CAP

    def test_positive_ev_has_nonzero_kelly(self):
        """Recs with positive EV should have kelly_base > 0."""
        lines = [
            _ml_line("DraftKings", -150, 130),
            _ml_line("FanDuel", -140, 120),
            _ml_line("BetMGM", -160, 140),
        ]
        recs = recommend_best_bets(lines, top_n=6)
        positive_ev = [r for r in recs if r.ev > 0]
        for r in positive_ev:
            assert r.kelly_base > 0, (
                f"{r.selection} has +EV but kelly_base=0"
            )
