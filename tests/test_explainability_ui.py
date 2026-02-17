"""Smoke tests for the explainability UI component."""

from __future__ import annotations

from datetime import datetime

import pytest

from line_tracker.best_bets import recommend_best_bets
from line_tracker.models import BestBetResult, BettingLine, BetType
from line_tracker.ui.components.explainability import (
    _build_metrics_table,
    render_pick_explanation,
)


# ── Helpers ────────────────────────────────────────────────────────────


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


def _sample_lines() -> list[BettingLine]:
    return [
        _ml_line("Pinnacle", -150, 130),
        _ml_line("DraftKings", -145, 125),
        _ml_line("FanDuel", -155, 135),
        _ml_line("BetMGM", -148, 128),
        _ml_line("Caesars", -150, 130),
        _ml_line("BetOnline", -152, 132),
    ]


def _make_bbr() -> BestBetResult:
    """Create a realistic BestBetResult for tests."""
    return BestBetResult(
        edge_pct=2.5,
        consensus_prob=0.55,
        best_odds_american=-150,
        best_odds_decimal=1.6667,
        books_used=["Pinnacle", "DraftKings", "FanDuel"],
        volatility_sigma=0.02,
        recency_weight=0.95,
        outliers_removed=1,
        explanation={
            "consensus_method": "trimmed_mean",
            "edge_breakdown": {
                "consensus_prob": 0.55,
                "breakeven_prob": 0.60,
                "edge_pp": -0.05,
                "edge_pct": 2.5,
                "ev_roi": 0.03,
                "ev_100": 3.0,
            },
            "quality_factors": {
                "edge_score": 70,
                "agreement_score": 80,
                "coverage_score": 75,
                "freshness_score": 90,
                "weights": {"edge": 0.3, "agreement": 0.3},
                "quality_score": 78,
                "quality_tier": "Strong",
            },
            "confidence_reasoning": {
                "edge_z": 2.1,
                "thresholds": {"high": 2.0, "medium": 1.0},
                "result": "High",
            },
            "outlier_info": {
                "outlier_filtered": True,
                "outliers_removed": 1,
                "outlier_rate": 0.14,
            },
            "kelly": {"sizing_note": "1-unit"},
            "ev_edge": {},
            "market_context": {},
            "recency": {},
        },
        market="moneyline",
        selection="Celtics",
        side="home",
        confidence="High",
        quality_score=78,
        quality_tier="Strong",
        edge_z=2.1,
        kelly_suggested=0.02,
        sizing_note="1-unit",
        best_sportsbook="Pinnacle",
        books_used_count=5,
        total_books_count=6,
    )


def _make_slate_entry(bbr: BestBetResult | None = None) -> dict:
    """Create a minimal slate entry dict."""
    entry: dict = {
        "event_id": "lakers_celtics_12345",
        "event": "Lakers @ Celtics",
        "market": "moneyline",
        "selection": "Celtics",
        "side": "home",
        "best_odds": -150,
        "best_sportsbook": "Pinnacle",
        "consensus_prob": 0.55,
        "edge_pct": 3.0,
        "quality_score": 78,
        "quality_tier": "Strong",
        "confidence": "High",
        "edge_z": 2.1,
        "kelly_suggested": 0.02,
        "sizing_note": "1-unit",
    }
    if bbr is not None:
        entry["best_bet_result"] = bbr
    return entry


def _make_shopping_entry() -> dict:
    """Create a minimal Best Lines to Shop standout dict."""
    return {
        "event": "Lakers @ Celtics",
        "market": "ML",
        "selection": "Celtics",
        "sportsbook": "Pinnacle",
        "odds": -150,
        "line": None,
        "book_prob": 0.6000,
        "consensus_prob": 0.5500,
        "edge": 0.0500,
        "dollar_impact": 5.00,
        "median_odds": -155,
        "exec_adv_100": 2.50,
        "books_used_excl": 4,
        "consensus_method": "median",
    }


# ── Tests ──────────────────────────────────────────────────────────────


class TestRenderPickExplanation:
    """Smoke tests for render_pick_explanation."""

    def test_no_output_when_debug_disabled(self) -> None:
        """When debug is off, no Streamlit calls should happen."""
        entry = _make_slate_entry(_make_bbr())
        # Should return None with no side effects
        result = render_pick_explanation(entry, debug_enabled=False)
        assert result is None

    def test_import_succeeds(self) -> None:
        """The component module imports cleanly."""
        from line_tracker.ui.components import explainability  # noqa: F401

    def test_build_metrics_table_with_bbr(self) -> None:
        """_build_metrics_table returns a list of [metric, value] rows."""
        bbr = _make_bbr()
        entry = _make_slate_entry(bbr)
        table = _build_metrics_table(bbr, entry)
        assert isinstance(table, list)
        assert len(table) > 0
        for row in table:
            assert len(row) == 2
            assert isinstance(row[0], str)
            assert isinstance(row[1], str)

    def test_build_metrics_table_contains_expected_keys(self) -> None:
        """All required metrics appear in the table."""
        bbr = _make_bbr()
        entry = _make_slate_entry(bbr)
        table = _build_metrics_table(bbr, entry)
        labels = [row[0] for row in table]
        expected = [
            "Edge (%)",
            "Consensus prob",
            "Best odds (American)",
            "Best book",
            "Quality tier",
            "Quality score",
            "Edge Z",
            "Recency weight",
            "Volatility sigma",
            "Outliers removed",
            "Kelly suggested",
            "Sizing note",
            "Books used",
        ]
        for label in expected:
            assert label in labels, f"Missing metric: {label}"

    def test_build_metrics_table_books_used_formatted(self) -> None:
        """Books used should be a comma-separated string."""
        bbr = _make_bbr()
        entry = _make_slate_entry(bbr)
        table = _build_metrics_table(bbr, entry)
        books_row = [r for r in table if r[0] == "Books used"][0]
        assert "Pinnacle" in books_row[1]
        assert "DraftKings" in books_row[1]

    def test_entry_without_bbr_uses_shopping_path(self) -> None:
        """An entry without best_bet_result should not raise."""
        entry = _make_shopping_entry()
        # Should not raise; debug_enabled=False means no output
        result = render_pick_explanation(entry, debug_enabled=False)
        assert result is None


class TestSlateCarriesBestBetResult:
    """The slate builder attaches best_bet_result to entry dicts."""

    def test_recommend_best_bets_attaches_bbr(self) -> None:
        """Recs from recommend_best_bets should carry best_bet_result."""
        lines = _sample_lines()
        recs = recommend_best_bets(
            lines, top_n=2, now=datetime(2025, 1, 1, 12, 5),
        )
        for rec in recs:
            if rec.skipped_reason:
                continue
            assert hasattr(rec, "best_bet_result")
            assert isinstance(rec.best_bet_result, BestBetResult)

    def test_slate_entry_carries_bbr(self) -> None:
        """build_daily_slate should carry best_bet_result through."""
        from line_tracker.slate import build_daily_slate

        lines = _sample_lines()
        lines_by_event = {"lakers_celtics": lines}
        slate = build_daily_slate(lines_by_event)
        all_entries = (
            slate["tier1a"]
            + slate["tier1b"]
            + slate["tier1"]
            + slate["tier2"]
            + slate["tier3"]
            + slate["stay_away"]
        )
        # At least some entries should exist
        assert len(all_entries) > 0
        for entry in all_entries:
            assert "best_bet_result" in entry
            bbr = entry["best_bet_result"]
            if bbr is not None:
                assert isinstance(bbr, BestBetResult)
