"""Tests for the ``kpi`` CLI command and ``kpi_report`` service function."""

import uuid
from datetime import datetime, timezone

from line_tracker.__main__ import main
from line_tracker.services.automation_service import kpi_report
from line_tracker.storage import LineStore


def test_kpi_empty_db(tmp_path, capsys):
    """kpi on a fresh DB prints zeros without errors."""
    db = str(tmp_path / "empty.db")
    ret = main(["kpi", "--db", db])
    assert ret == 0
    out = capsys.readouterr().out
    assert "KPI Report" in out
    assert "Cycles run:          0" in out


def test_kpi_report_with_data(tmp_path):
    """kpi_report returns correct aggregates for seeded snapshots."""
    db = str(tmp_path / "kpi.db")
    now = datetime.now(timezone.utc).isoformat()
    rid = uuid.uuid4().hex

    with LineStore(db) as store:
        # Insert 3 snapshots: 2 closed (1 CLV+, 1 CLV-), 1 open.
        snaps = [
            {
                "created_at": now,
                "event_id": "evt1",
                "sport": "basketball_nba",
                "market": "moneyline",
                "selection": "TeamA",
                "book": "DraftKings",
                "odds_american": -150.0,
                "odds_decimal": 1.6667,
                "consensus_prob": 0.60,
                "tier": "tier1a",
                "run_id": rid,
            },
            {
                "created_at": now,
                "event_id": "evt2",
                "sport": "basketball_nba",
                "market": "moneyline",
                "selection": "TeamB",
                "book": "FanDuel",
                "odds_american": +130.0,
                "odds_decimal": 2.30,
                "consensus_prob": 0.45,
                "tier": "tier2",
                "run_id": rid,
            },
            {
                "created_at": now,
                "event_id": "evt3",
                "sport": "basketball_nba",
                "market": "spread",
                "selection": "TeamC",
                "book": "BetMGM",
                "odds_american": -110.0,
                "odds_decimal": 1.9091,
                "consensus_prob": 0.52,
                "tier": "tier1b",
                "run_id": rid,
            },
        ]
        store.log_rec_snapshots(snaps)

        # Close snapshot 1 with CLV+ (clv_delta_implied > 0)
        store.close_snapshot(
            1,
            close_odds_american=-140.0,
            close_odds_decimal=1.7143,
            close_implied_prob=0.5833,
            open_implied_prob=0.6000,
            clv_delta_american=10.0,
            clv_delta_implied=0.02,
        )
        # Close snapshot 2 with CLV- (clv_delta_implied < 0)
        store.close_snapshot(
            2,
            close_odds_american=+140.0,
            close_odds_decimal=2.40,
            close_implied_prob=0.4167,
            open_implied_prob=0.4348,
            clv_delta_american=-10.0,
            clv_delta_implied=-0.02,
        )

        kpi = kpi_report(store, last_hours=1)

    assert kpi["cycles"] == 1
    assert kpi["total_snapshots"] == 3
    assert kpi["closed_count"] == 2
    assert kpi["clv_positive"] == 1
    assert 66.0 < kpi["pct_closed"] < 67.0  # 2/3 ≈ 66.7%
    assert kpi["clv_rate"] == 50.0  # 1/2
    assert kpi["avg_picks_per_cycle"] == 3.0


def test_kpi_last_hours_flag(tmp_path, capsys):
    """--last-hours is forwarded correctly."""
    db = str(tmp_path / "empty.db")
    ret = main(["kpi", "--last-hours", "48", "--db", db])
    assert ret == 0
    out = capsys.readouterr().out
    assert "last 48h" in out


def test_kpi_multiple_cycles(tmp_path):
    """kpi_report counts distinct run_id values as separate cycles."""
    db = str(tmp_path / "multi.db")
    now = datetime.now(timezone.utc).isoformat()
    rid1 = uuid.uuid4().hex
    rid2 = uuid.uuid4().hex

    with LineStore(db) as store:
        base = {
            "created_at": now,
            "event_id": "evt1",
            "sport": "basketball_nba",
            "market": "moneyline",
            "selection": "TeamA",
            "book": "DraftKings",
            "odds_american": -150.0,
            "odds_decimal": 1.6667,
            "consensus_prob": 0.60,
            "tier": "tier1a",
        }
        store.log_rec_snapshots([{**base, "run_id": rid1}])
        store.log_rec_snapshots([
            {**base, "run_id": rid2, "book": "FanDuel"},
        ])

        kpi = kpi_report(store, last_hours=1)

    assert kpi["cycles"] == 2
    assert kpi["total_snapshots"] == 2
