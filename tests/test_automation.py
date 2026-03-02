"""Tests for Step 3 – Automation service (cycle)."""

from __future__ import annotations

from line_tracker.services.automation_service import run_cycle
from line_tracker.storage import LineStore


class TestRunCycle:
    def test_no_events_returns_empty(self, tmp_path):
        """Cycle with an empty DB returns zero counts."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            result = run_cycle(store, sport="basketball_nba")

        assert result["slate_entries"] == 0
        assert result["pruned_count"] == 0
        assert result["top_picks"] == []
        assert result["snapshot_count"] == 0
        assert result["closed_count"] == 0
        assert "cycle_ts" in result

    def test_dry_run_skips_writes(self, tmp_path):
        """Dry run doesn't write snapshots."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            result = run_cycle(
                store, sport="basketball_nba", dry_run=True,
            )

        assert result["snapshot_count"] == 0
        assert result["closed_count"] == 0

    def test_cycle_returns_expected_keys(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            result = run_cycle(store, sport="basketball_nba")

        expected_keys = {
            "cycle_ts", "slate_entries", "pruned_count",
            "top_picks", "snapshot_count", "closed_count",
        }
        assert set(result.keys()) == expected_keys
