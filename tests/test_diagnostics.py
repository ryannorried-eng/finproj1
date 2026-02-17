"""Tests for dashboard diagnostics helpers."""

from datetime import datetime, timezone

from line_tracker.models import BettingLine, BetType
from line_tracker.storage import LineStore
from line_tracker.ui.components.diagnostics import (
    get_db_counts,
    get_db_status,
    get_latest_timestamps,
)


def test_diagnostics_status_counts_and_latest(tmp_path):
    db = tmp_path / "diag.db"
    with LineStore(db) as store:
        store.save_lines([
            BettingLine(
                sportsbook="DraftKings",
                sport="americanfootball_nfl",
                event="Bills @ Chiefs",
                bet_type=BetType.MONEYLINE,
                home_team="Chiefs",
                away_team="Bills",
                home_value=-150,
                away_value=130,
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ])
        store.insert_bet({
            "bet_id": "b1",
            "created_at": "2026-01-01T00:00:00Z",
            "sportsbook": "DK",
            "stake": 100.0,
            "total_odds_american": -110,
            "total_odds_decimal": 1.9091,
            "potential_payout": 190.91,
            "profit": 90.91,
            "status": "active",
            "settled_at": None,
            "outcome": None,
        })
        store.save_clv_pick(
            bet_id="b1",
            leg_index=0,
            event="Bills @ Chiefs",
            market="ML",
            pick_side="Home",
            pick_line_value=None,
            pick_odds_american=-110.0,
            pick_odds_decimal=1.9091,
            consensus_prob_at_pick=0.52,
        )
        store.close_clv(
            bet_id="b1",
            leg_index=0,
            consensus_prob_close=0.55,
            best_odds_close_american=-120.0,
            best_odds_close_decimal=1.8333,
        )

        status = get_db_status(store)
        counts = get_db_counts(store)
        latest = get_latest_timestamps(store)

        assert "db_path" in status
        assert status["schema_version"] >= 2
        expected_keys = {"journal_mode", "foreign_keys", "busy_timeout"}
        assert set(status["pragmas"].keys()) == expected_keys

        assert counts["lines"] == 1
        assert counts["bets"] == 1
        assert counts["bet_clv"] == 1

        assert latest["latest_line_timestamp"] is not None
        assert latest["latest_bet_created_at"] is not None
        assert latest["latest_clv_closed_at"] is not None
