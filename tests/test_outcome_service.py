"""Tests for Step 6 – Outcome feedback loop."""

from __future__ import annotations

import csv

import pytest

from line_tracker.services.outcome_service import (
    import_outcomes_csv,
    link_outcomes,
    roi_report,
)
from line_tracker.storage import LineStore


def _write_csv(path, rows: list[dict]) -> str:
    """Write a test CSV file and return its path."""
    filepath = path / "outcomes.csv"
    has_line_value = any("line_value" in r for r in rows)
    fieldnames = ["event_id", "market", "selection", "result", "settled_at"]
    if has_line_value:
        fieldnames.append("line_value")
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(filepath)


def _insert_closed_snapshot(
    store, event_id, market, selection, tier, alpha, odds_dec,
    line=None, odds_american=-110.0,
):
    """Insert a closed snapshot for testing."""
    snap = {
        "created_at": "2026-01-15T12:00:00",
        "event_id": event_id,
        "sport": "nfl",
        "market": market,
        "selection": selection,
        "line": line,
        "book": "FanDuel",
        "odds_american": odds_american,
        "odds_decimal": odds_dec,
        "consensus_prob": 0.55,
        "breakeven_prob": 0.52,
        "edge_pct": 3.0,
        "edge_ev": 0.03,
        "edge_ev_shrunk": 0.025,
        "ev_100": 3.0,
        "edge_z": 2.0,
        "quality_score": 80,
        "confidence_label": "High",
        "tier": tier,
        "meta": None,
        "alpha_score": 75,
        "alpha_label": alpha,
    }
    store.log_rec_snapshots([snap])
    unclosed = store.get_unclosed_snapshots()
    sid = unclosed[-1]["snapshot_id"]
    store.close_snapshot(
        sid,
        close_odds_american=-120.0,
        close_odds_decimal=1.8333,
        close_implied_prob=0.5455,
        open_implied_prob=0.5238,
        clv_delta_american=-10.0,
        clv_delta_implied=0.02,
    )


class TestImportOutcomesCsv:
    def test_basic_import(self, tmp_path):
        csv_path = _write_csv(tmp_path, [
            {"event_id": "e1", "market": "spread", "selection": "Chiefs",
             "result": "win", "settled_at": ""},
            {"event_id": "e2", "market": "moneyline", "selection": "Bills",
             "result": "loss", "settled_at": ""},
        ])
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            count = import_outcomes_csv(store, csv_path)
        assert count == 2

    def test_invalid_result_skipped(self, tmp_path):
        csv_path = _write_csv(tmp_path, [
            {"event_id": "e1", "market": "spread", "selection": "Chiefs",
             "result": "invalid", "settled_at": ""},
        ])
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            count = import_outcomes_csv(store, csv_path)
        assert count == 0

    def test_upsert_idempotent(self, tmp_path):
        csv_path = _write_csv(tmp_path, [
            {"event_id": "e1", "market": "spread", "selection": "Chiefs",
             "result": "win", "settled_at": ""},
        ])
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            count1 = import_outcomes_csv(store, csv_path)
            count2 = import_outcomes_csv(store, csv_path)
        assert count1 == 1
        assert count2 == 1  # upsert still touches the row

    def test_import_with_line_value(self, tmp_path):
        csv_path = _write_csv(tmp_path, [
            {"event_id": "e1", "market": "spread", "selection": "Chiefs",
             "result": "win", "settled_at": "", "line_value": "-3.5"},
            {"event_id": "e1", "market": "total", "selection": "Over",
             "result": "loss", "settled_at": "", "line_value": "45.5"},
        ])
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            count = import_outcomes_csv(store, csv_path)
        assert count == 2

    def test_upsert_with_line_value(self, tmp_path):
        """Same event+market+selection but different line_value → separate rows."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            store.import_outcomes([
                {"event_id": "e1", "market": "spread", "selection": "Chiefs",
                 "result": "win", "line_value": -3.5},
                {"event_id": "e1", "market": "spread", "selection": "Chiefs",
                 "result": "loss", "line_value": -7.0},
            ])
            all_outcomes = store.get_all_outcomes()
        assert len(all_outcomes) == 2

    def test_upsert_same_line_value(self, tmp_path):
        """Same event+market+selection+line_value → single row updated."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            store.import_outcomes([
                {"event_id": "e1", "market": "spread", "selection": "Chiefs",
                 "result": "win", "line_value": -3.5},
            ])
            store.import_outcomes([
                {"event_id": "e1", "market": "spread", "selection": "Chiefs",
                 "result": "loss", "line_value": -3.5},
            ])
            all_outcomes = store.get_all_outcomes()
        assert len(all_outcomes) == 1
        assert all_outcomes[0]["result"] == "loss"


class TestLinkOutcomes:
    def test_link_win(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            _insert_closed_snapshot(
                store, "e1", "spread", "Chiefs", "tier1b", "Strong", 1.9091,
            )
            # Import a win outcome (line_value NULL matches snapshot line NULL)
            store.import_outcomes([{
                "event_id": "e1",
                "market": "spread",
                "selection": "Chiefs",
                "result": "win",
            }])
            linked = link_outcomes(store)
            assert linked == 1

            # Verify actual_roi was set
            row = store._conn.execute(
                "SELECT outcome_result, actual_roi "
                "FROM rec_snapshots WHERE event_id = 'e1'"
            ).fetchone()
            assert row["outcome_result"] == "win"
            assert row["actual_roi"] == pytest.approx(1.9091 - 1.0, abs=0.01)

    def test_link_loss(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            _insert_closed_snapshot(
                store, "e1", "spread", "Chiefs", "tier1b", "Strong", 1.9091,
            )
            store.import_outcomes([{
                "event_id": "e1",
                "market": "spread",
                "selection": "Chiefs",
                "result": "loss",
            }])
            link_outcomes(store)
            row = store._conn.execute(
                "SELECT outcome_result, actual_roi "
                "FROM rec_snapshots WHERE event_id = 'e1'"
            ).fetchone()
            assert row["outcome_result"] == "loss"
            assert row["actual_roi"] == pytest.approx(-1.0, abs=0.01)

    def test_spread_requires_line_value_match(self, tmp_path):
        """Spread outcome with line_value=-3.5 should only link to snapshot
        with matching line=-3.5, not to one with line=-7.0."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            _insert_closed_snapshot(
                store, "e1", "spread", "Chiefs", "tier1b", "Strong", 1.9091,
                line=-3.5,
            )
            _insert_closed_snapshot(
                store, "e1", "spread", "Chiefs", "tier2", "Neutral", 2.0,
                line=-7.0, odds_american=-120.0,
            )
            store.import_outcomes([{
                "event_id": "e1",
                "market": "spread",
                "selection": "Chiefs",
                "result": "win",
                "line_value": -3.5,
            }])
            linked = link_outcomes(store)
            assert linked == 1

            rows = store._conn.execute(
                "SELECT line, outcome_result FROM rec_snapshots "
                "WHERE event_id = 'e1' ORDER BY line"
            ).fetchall()
            matched = {r["line"]: r["outcome_result"] for r in rows}
            assert matched[-7.0] is None  # not linked
            assert matched[-3.5] == "win"  # linked

    def test_total_requires_line_value_match(self, tmp_path):
        """Total market outcome should only link when line_value matches."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            _insert_closed_snapshot(
                store, "e1", "total", "Over", "tier1b", "Strong", 1.9091,
                line=45.5,
            )
            _insert_closed_snapshot(
                store, "e1", "total", "Over", "tier2", "Neutral", 2.0,
                line=48.0, odds_american=-120.0,
            )
            store.import_outcomes([{
                "event_id": "e1",
                "market": "total",
                "selection": "Over",
                "result": "loss",
                "line_value": 48.0,
            }])
            linked = link_outcomes(store)
            assert linked == 1

            rows = store._conn.execute(
                "SELECT line, outcome_result FROM rec_snapshots "
                "WHERE event_id = 'e1' ORDER BY line"
            ).fetchall()
            matched = {r["line"]: r["outcome_result"] for r in rows}
            assert matched[45.5] is None  # not linked
            assert matched[48.0] == "loss"  # linked

    def test_moneyline_ignores_line_value(self, tmp_path):
        """Moneyline outcomes match without requiring line_value."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            _insert_closed_snapshot(
                store, "e1", "moneyline", "Chiefs", "tier1b", "Strong", 1.9091,
            )
            store.import_outcomes([{
                "event_id": "e1",
                "market": "moneyline",
                "selection": "Chiefs",
                "result": "win",
            }])
            linked = link_outcomes(store)
            assert linked == 1

            row = store._conn.execute(
                "SELECT outcome_result FROM rec_snapshots WHERE event_id = 'e1'"
            ).fetchone()
            assert row["outcome_result"] == "win"

    def test_spread_no_match_wrong_line(self, tmp_path):
        """Spread outcome with mismatched line_value should not link."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            _insert_closed_snapshot(
                store, "e1", "spread", "Chiefs", "tier1b", "Strong", 1.9091,
                line=-3.5,
            )
            store.import_outcomes([{
                "event_id": "e1",
                "market": "spread",
                "selection": "Chiefs",
                "result": "win",
                "line_value": -7.0,
            }])
            linked = link_outcomes(store)
            assert linked == 0


class TestRoiReport:
    def test_empty_report(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            report = roi_report(store)
        assert report["overall"]["total"] == 0

    def test_report_with_outcomes(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            # Two picks: one win, one loss
            _insert_closed_snapshot(
                store, "e1", "spread", "Chiefs", "tier1b", "Strong", 1.9091,
            )
            _insert_closed_snapshot(
                store, "e2", "spread", "Bills", "tier2", "Neutral", 2.0,
            )
            store.import_outcomes([
                {"event_id": "e1", "market": "spread",
                 "selection": "Chiefs", "result": "win"},
                {"event_id": "e2", "market": "spread",
                 "selection": "Bills", "result": "loss"},
            ])
            link_outcomes(store)
            report = roi_report(store)

        assert report["overall"]["total"] == 2
        assert report["overall"]["wins"] == 1
        assert report["overall"]["losses"] == 1
        assert "tier1b" in report["by_tier"]
        assert "Strong" in report["by_alpha"]
