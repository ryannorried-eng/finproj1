"""Tests for market structure analytics."""

from datetime import datetime

import pytest

from line_tracker.market_structure import (
    SHARP_BOOKS,
    analyze_market,
    hold_spread,
    line_dispersion,
    market_tag,
    sharp_retail_divergence,
)
from line_tracker.models import BettingLine, BetType

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

_NOW = datetime(2026, 2, 1, 12, 0)


def _ml_line(
    sportsbook: str,
    home_odds: float,
    away_odds: float,
    event: str = "Bills @ Chiefs",
) -> BettingLine:
    return BettingLine(
        sportsbook=sportsbook,
        sport="americanfootball_nfl",
        event=event,
        bet_type=BetType.MONEYLINE,
        home_team="Chiefs",
        away_team="Bills",
        home_value=home_odds,
        away_value=away_odds,
        timestamp=_NOW,
    )


def _spread_line(
    sportsbook: str,
    spread: float,
    home_price: float,
    away_price: float,
) -> BettingLine:
    return BettingLine(
        sportsbook=sportsbook,
        sport="americanfootball_nfl",
        event="Bills @ Chiefs",
        bet_type=BetType.SPREAD,
        home_team="Chiefs",
        away_team="Bills",
        home_value=spread,
        away_value=-spread,
        timestamp=_NOW,
        home_price=home_price,
        away_price=away_price,
    )


def _total_line(
    sportsbook: str,
    total: float,
    over_price: float,
    under_price: float,
) -> BettingLine:
    return BettingLine(
        sportsbook=sportsbook,
        sport="americanfootball_nfl",
        event="Bills @ Chiefs",
        bet_type=BetType.TOTAL,
        home_team="Chiefs",
        away_team="Bills",
        home_value=total,
        away_value=total,
        timestamp=_NOW,
        home_price=over_price,
        away_price=under_price,
    )


# -------------------------------------------------------------------
# line_dispersion
# -------------------------------------------------------------------

class TestLineDispersion:
    def test_empty(self):
        assert line_dispersion([]) == {}

    def test_uniform_spread(self):
        lines = [
            _spread_line("DK", -3.5, -110, -110),
            _spread_line("FD", -3.5, -105, -115),
            _spread_line("BetMGM", -3.5, -108, -112),
        ]
        disp = line_dispersion(lines)
        assert disp == {-3.5: 3}

    def test_mixed_spreads(self):
        lines = [
            _spread_line("DK", -3.5, -110, -110),
            _spread_line("FD", -3.0, -105, -115),
            _spread_line("BetMGM", -3.5, -108, -112),
            _spread_line("Caesars", -3.0, -110, -110),
        ]
        disp = line_dispersion(lines)
        assert disp == {-3.5: 2, -3.0: 2}

    def test_totals(self):
        lines = [
            _total_line("DK", 48.5, -110, -110),
            _total_line("FD", 49.0, -105, -115),
            _total_line("BetMGM", 48.5, -108, -112),
        ]
        disp = line_dispersion(lines)
        assert disp == {48.5: 2, 49.0: 1}

    def test_sorted_output(self):
        lines = [
            _spread_line("DK", -4.0, -110, -110),
            _spread_line("FD", -3.0, -110, -110),
            _spread_line("BetMGM", -3.5, -110, -110),
        ]
        disp = line_dispersion(lines)
        assert list(disp.keys()) == [-4.0, -3.5, -3.0]


# -------------------------------------------------------------------
# sharp_retail_divergence
# -------------------------------------------------------------------

class TestSharpRetailDivergence:
    def test_divergence_with_enough_books(self):
        """With 2+ sharp and 2+ retail, divergence is computed."""
        lines = [
            # Sharp books
            _ml_line("Pinnacle", -150, 130),
            _ml_line("Circa", -145, 125),
            # Retail books
            _ml_line("DraftKings", -160, 140),
            _ml_line("FanDuel", -155, 135),
        ]
        result = sharp_retail_divergence(lines)
        assert result["sharp_consensus"] is not None
        assert result["retail_consensus"] is not None
        assert result["divergence"] is not None
        assert result["divergence"] >= 0
        assert len(result["sharp_books"]) == 2
        assert len(result["retail_books"]) == 2

    def test_insufficient_sharp_books(self):
        """With < 2 sharp books, divergence is None."""
        lines = [
            _ml_line("Pinnacle", -150, 130),
            _ml_line("DraftKings", -160, 140),
            _ml_line("FanDuel", -155, 135),
        ]
        result = sharp_retail_divergence(lines)
        assert result["divergence"] is None

    def test_insufficient_retail_books(self):
        """With < 2 retail books, divergence is None."""
        lines = [
            _ml_line("Pinnacle", -150, 130),
            _ml_line("Circa", -145, 125),
            _ml_line("FanDuel", -155, 135),
        ]
        result = sharp_retail_divergence(lines)
        assert result["divergence"] is None

    def test_identical_lines_zero_divergence(self):
        """When all books have the same odds, divergence is ~0."""
        lines = [
            _ml_line("Pinnacle", -150, 130),
            _ml_line("Circa", -150, 130),
            _ml_line("DraftKings", -150, 130),
            _ml_line("FanDuel", -150, 130),
        ]
        result = sharp_retail_divergence(lines)
        assert result["divergence"] == pytest.approx(0.0, abs=0.001)

    def test_divergent_lines(self):
        """Large difference between sharp and retail odds yields divergence."""
        lines = [
            # Sharp: home ~-120 (slight fav)
            _ml_line("Pinnacle", -120, 100),
            _ml_line("Circa", -125, 105),
            # Retail: home ~-180 (heavy fav)
            _ml_line("DraftKings", -180, 160),
            _ml_line("FanDuel", -175, 155),
        ]
        result = sharp_retail_divergence(lines)
        assert result["divergence"] > 0.02

    def test_sharp_books_classification(self):
        """Verify known sharp books are in the SHARP_BOOKS set."""
        for book in ["Pinnacle", "Circa", "Bookmaker", "BetOnline", "SuperBook"]:
            assert book in SHARP_BOOKS

    def test_spread_market_divergence(self):
        """Divergence works for spread markets (uses prices)."""
        lines = [
            _spread_line("Pinnacle", -3.5, -105, -115),
            _spread_line("Circa", -3.5, -107, -113),
            _spread_line("DraftKings", -3.5, -110, -110),
            _spread_line("FanDuel", -3.5, -112, -108),
        ]
        result = sharp_retail_divergence(lines)
        assert result["divergence"] is not None
        assert result["divergence"] >= 0


# -------------------------------------------------------------------
# hold_spread
# -------------------------------------------------------------------

class TestHoldSpread:
    def test_enough_books(self):
        holds = {"DK": 4.5, "FD": 4.8, "BetMGM": 5.2, "Caesars": 4.3}
        hs = hold_spread(holds)
        assert hs is not None
        assert hs >= 0

    def test_fewer_than_4_returns_none(self):
        holds = {"DK": 4.5, "FD": 4.8, "BetMGM": 5.2}
        assert hold_spread(holds) is None

    def test_uniform_holds(self):
        holds = {"A": 4.5, "B": 4.5, "C": 4.5, "D": 4.5}
        hs = hold_spread(holds)
        assert hs == 0.0


# -------------------------------------------------------------------
# market_tag
# -------------------------------------------------------------------

class TestMarketTag:
    def test_efficient(self):
        tag = market_tag(
            market_hold_median=3.5,
            volatility_sigma=0.02,
            divergence=0.01,
        )
        assert tag == "Efficient"

    def test_noisy_high_sigma(self):
        tag = market_tag(
            market_hold_median=3.5,
            volatility_sigma=0.08,
            divergence=0.01,
        )
        assert tag == "Noisy"

    def test_noisy_high_divergence(self):
        tag = market_tag(
            market_hold_median=3.5,
            volatility_sigma=0.02,
            divergence=0.06,
        )
        assert tag == "Noisy"

    def test_normal_moderate_values(self):
        tag = market_tag(
            market_hold_median=6.0,
            volatility_sigma=0.04,
            divergence=0.03,
        )
        assert tag == "Normal"

    def test_none_divergence_treated_as_zero(self):
        tag = market_tag(
            market_hold_median=3.5,
            volatility_sigma=0.02,
            divergence=None,
        )
        assert tag == "Efficient"


# -------------------------------------------------------------------
# analyze_market — integration
# -------------------------------------------------------------------

class TestAnalyzeMarket:
    def test_moneyline_analysis(self):
        lines = [
            _ml_line("Pinnacle", -150, 130),
            _ml_line("Circa", -145, 125),
            _ml_line("DraftKings", -160, 140),
            _ml_line("FanDuel", -155, 135),
            _ml_line("BetMGM", -152, 132),
        ]
        result = analyze_market(lines)
        assert result is not None
        assert result["bet_type"] == "moneyline"
        assert result["books_total"] == 5
        assert result["market_hold_median"] > 0
        assert result["volatility_sigma"] >= 0
        assert result["tag"] in ("Efficient", "Normal", "Noisy")
        assert result["divergence"] is not None
        assert result["line_dispersion"] is None  # ML has no dispersion

    def test_spread_analysis_has_dispersion(self):
        lines = [
            _spread_line("Pinnacle", -3.5, -105, -115),
            _spread_line("Circa", -3.5, -107, -113),
            _spread_line("DraftKings", -3.0, -110, -110),
            _spread_line("FanDuel", -3.5, -112, -108),
        ]
        result = analyze_market(lines)
        assert result is not None
        assert result["bet_type"] == "spread"
        assert result["line_dispersion"] is not None
        assert -3.5 in result["line_dispersion"]

    def test_insufficient_lines(self):
        lines = [_ml_line("DK", -150, 130)]
        assert analyze_market(lines) is None

    def test_empty_lines(self):
        assert analyze_market([]) is None

    def test_total_analysis(self):
        lines = [
            _total_line("Pinnacle", 48.5, -105, -115),
            _total_line("Circa", 48.5, -107, -113),
            _total_line("DraftKings", 49.0, -110, -110),
            _total_line("FanDuel", 48.5, -108, -112),
        ]
        result = analyze_market(lines)
        assert result is not None
        assert result["bet_type"] == "total"
        assert result["line_dispersion"] is not None
        assert 48.5 in result["line_dispersion"]
        assert 49.0 in result["line_dispersion"]
