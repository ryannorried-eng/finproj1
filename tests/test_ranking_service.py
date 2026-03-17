"""Tests for Step 4 – Ranking service."""

from __future__ import annotations

from line_tracker.services.ranking_service import (
    format_picks_report,
    select_top_picks,
)


def _make_entry(edge_z, ev_shrunk=0.03, quality=80, **kw) -> dict:
    base = {
        "event_id": "e1",
        "market": "spread",
        "selection": "Chiefs",
        "tier": "tier1b",
        "alpha_label": "Strong",
        "edge_z": edge_z,
        "edge_ev_shrunk": ev_shrunk,
        "quality_score": quality,
        "best_odds": -110,
        "best_sportsbook": "FanDuel",
        "consensus_prob": 0.55,
    }
    base.update(kw)
    return base


class TestSelectTopPicks:
    def test_empty_list(self):
        assert select_top_picks([]) == []

    def test_top_n_default_5(self):
        entries = [_make_entry(i * 0.5 + 1.0) for i in range(7)]
        result = select_top_picks(entries)
        assert len(result) == 5
        # Highest edge_z first
        assert result[0]["edge_z"] == 4.0
        assert result[1]["edge_z"] == 3.5
        assert result[2]["edge_z"] == 3.0

    def test_top_n_custom(self):
        entries = [_make_entry(i * 0.5 + 1.0) for i in range(7)]
        result = select_top_picks(entries, top_n=1)
        assert len(result) == 1
        assert result[0]["edge_z"] == 4.0

    def test_fewer_than_top_n(self):
        entries = [_make_entry(2.0)]
        result = select_top_picks(entries, top_n=5)
        assert len(result) == 1

    def test_tiebreak_by_ev_shrunk(self):
        entries = [
            _make_entry(2.0, ev_shrunk=0.01),
            _make_entry(2.0, ev_shrunk=0.05),
        ]
        result = select_top_picks(entries, top_n=2)
        assert result[0]["edge_ev_shrunk"] == 0.05

    def test_tiebreak_by_quality(self):
        entries = [
            _make_entry(2.0, ev_shrunk=0.03, quality=60),
            _make_entry(2.0, ev_shrunk=0.03, quality=90),
        ]
        result = select_top_picks(entries, top_n=2)
        assert result[0]["quality_score"] == 90

    def test_tiebreak_by_books_used(self):
        entries = [
            _make_entry(2.0, ev_shrunk=0.03, quality=80, books_used=4),
            _make_entry(2.0, ev_shrunk=0.03, quality=80, books_used=8),
        ]
        result = select_top_picks(entries, top_n=2)
        assert result[0]["books_used"] == 8


class TestFormatPicksReport:
    def test_no_picks(self):
        report = format_picks_report([])
        assert "No actionable picks" in report

    def test_report_has_pick_data(self):
        picks = [_make_entry(2.5)]
        report = format_picks_report(picks)
        assert "Best 1 Available Picks" in report
        assert "spread" in report
        assert "Chiefs" in report
        assert "Edge-Z" in report
