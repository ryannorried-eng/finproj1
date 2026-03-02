"""Tests for new CLI commands (Steps 1–7)."""

from __future__ import annotations

import csv

from line_tracker.__main__ import main


class TestSlateCommand:
    def test_slate_no_data(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        rc = main(["slate", "--db", db, "--sport", "nba"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No lines available" in out


class TestSnapshotCommand:
    def test_snapshot_no_data(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        rc = main(["snapshot", "--db", db, "--sport", "nba"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No lines available" in out


class TestCycleCommand:
    def test_cycle_empty_db(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        rc = main(["cycle", "--db", db, "--sport", "nba"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Cycle completed" in out
        assert "Slate entries: 0" in out

    def test_cycle_dry_run(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        rc = main(["cycle", "--db", db, "--sport", "nba", "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Cycle completed" in out


class TestImportOutcomesCommand:
    def test_import_csv(self, tmp_path, capsys):
        csv_path = tmp_path / "outcomes.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["event_id", "market", "selection", "result"],
            )
            writer.writeheader()
            writer.writerow({
                "event_id": "e1", "market": "spread",
                "selection": "Chiefs", "result": "win",
            })

        db = str(tmp_path / "test.db")
        rc = main(["import-outcomes", str(csv_path), "--db", db])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Imported 1 outcomes" in out


class TestRoiReportCommand:
    def test_roi_report_empty(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        rc = main(["roi-report", "--db", db])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Overall ROI" in out
        assert "Total: 0" in out


class TestTrainModelCommand:
    def test_train_no_data(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        rc = main(["train-model", "--db", db])
        assert rc == 0
        out = capsys.readouterr().out
        assert "no_data" in out
