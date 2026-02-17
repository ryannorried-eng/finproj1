"""Tests for bet persistence: bets survive DB close/reopen (refresh safety)."""

from __future__ import annotations

from pathlib import Path

import pytest

from line_tracker.bet_history import (
    create_bet,
    delete_bet_persistent,
    load_bets_from_db,
    persist_bet,
    persist_bet_with_snapshot,
    settle_bet_persistent,
    submit_bet,
)
from line_tracker.storage import LineStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leg(**overrides):
    base = {
        "sport": "NFL",
        "event_name": "Bills @ Chiefs",
        "sportsbook": "DraftKings",
        "market": "ML",
        "selection": "Home",
        "line": None,
        "odds": -150,
        "fetched_at": "2026-01-19T18:00:00",
    }
    base.update(overrides)
    return base


def _tmp_store(tmp_path: Path) -> LineStore:
    return LineStore(db_path=tmp_path / "test.db")


# ---------------------------------------------------------------------------
# Round-trip persistence
# ---------------------------------------------------------------------------

class TestBetPersistenceRoundTrip:
    """Bet + legs survive a full write → read cycle."""

    def test_submit_persists_bet_and_legs_round_trip(self, tmp_path):
        """Create a bet with 2 legs, persist, and read back."""
        legs = [
            _leg(odds=-150),
            _leg(odds=130, event_name="Eagles @ Cowboys"),
        ]
        bet = create_bet(stake=100, sportsbook="DraftKings", legs=legs)

        with _tmp_store(tmp_path) as store:
            persist_bet(bet, store)

            # Read back
            bets = load_bets_from_db(store, status="active")
            assert len(bets) == 1
            loaded = bets[0]
            assert loaded.id == bet.id
            assert loaded.sportsbook == "DraftKings"
            assert loaded.stake == 100
            assert loaded.combined_american == bet.combined_american
            assert loaded.combined_decimal == pytest.approx(
                bet.combined_decimal, abs=0.001,
            )
            assert loaded.status == "active"
            assert len(loaded.legs) == 2

            # Verify leg content
            events = {lg["event_name"] for lg in loaded.legs}
            assert "Bills @ Chiefs" in events
            assert "Eagles @ Cowboys" in events

    def test_single_leg_straight_bet(self, tmp_path):
        """Straight bet (1 leg) persists correctly."""
        bet = create_bet(stake=50, sportsbook="FanDuel", legs=[_leg(odds=200)])

        with _tmp_store(tmp_path) as store:
            persist_bet(bet, store)
            bets = load_bets_from_db(store, status="active")
            assert len(bets) == 1
            assert len(bets[0].legs) == 1
            assert bets[0].legs[0]["odds"] == 200
            assert bets[0].sportsbook == "FanDuel"
            assert bets[0].stake == 50


# ---------------------------------------------------------------------------
# Refresh safety: new LineStore on same DB
# ---------------------------------------------------------------------------

class TestRefreshSafety:
    """Simulates app refresh by opening a new LineStore on the same DB file."""

    def test_refresh_safety_simulated(self, tmp_path):
        """Bet survives closing and reopening the store (simulates refresh)."""
        db_path = tmp_path / "refresh.db"
        legs = [_leg(odds=-110), _leg(odds=150, event_name="Eagles @ Cowboys")]
        bet = create_bet(stake=200, sportsbook="BetMGM", legs=legs)

        # Session 1: persist
        with LineStore(db_path=db_path) as store1:
            persist_bet(bet, store1)

        # Session 2: new LineStore (simulates app restart)
        with LineStore(db_path=db_path) as store2:
            bets = load_bets_from_db(store2, status="active")
            assert len(bets) == 1
            assert bets[0].id == bet.id
            assert len(bets[0].legs) == 2
            assert bets[0].stake == 200
            assert bets[0].sportsbook == "BetMGM"

    def test_multiple_bets_survive_refresh(self, tmp_path):
        """Multiple bets persist across DB reconnections."""
        db_path = tmp_path / "multi.db"
        bet1 = create_bet(
            stake=100, sportsbook="DraftKings", legs=[_leg(odds=-150)],
        )
        bet2 = create_bet(
            stake=50, sportsbook="FanDuel",
            legs=[_leg(odds=200, event_name="Lions @ Packers")],
        )

        with LineStore(db_path=db_path) as store:
            persist_bet(bet1, store)
            persist_bet(bet2, store)

        with LineStore(db_path=db_path) as store:
            bets = load_bets_from_db(store)
            assert len(bets) == 2
            ids = {b.id for b in bets}
            assert bet1.id in ids
            assert bet2.id in ids


# ---------------------------------------------------------------------------
# Submit clears slip, not history
# ---------------------------------------------------------------------------

class TestSubmitClearsSlipNotHistory:
    def test_submit_clears_slip_not_history(self, tmp_path):
        """submit_bet clears slip state; persist_bet stores to DB."""
        state = {
            "bet_slip": [_leg()],
            "slip_book": "DraftKings",
            "slip_stake": 100.0,
        }
        bet = submit_bet(state, stake=100)

        # Slip cleared
        assert state["bet_slip"] == []
        assert state["slip_book"] is None

        # Bet returned with an id
        assert bet.id

        # Persist independently
        with _tmp_store(tmp_path) as store:
            persist_bet(bet, store)
            bets = load_bets_from_db(store, status="active")
            assert len(bets) == 1
            assert bets[0].id == bet.id


# ---------------------------------------------------------------------------
# Settlement persistence
# ---------------------------------------------------------------------------

class TestSettlementPersistence:
    def test_settle_won_persisted(self, tmp_path):
        bet = create_bet(stake=100, sportsbook="DK", legs=[_leg(odds=-150)])
        with _tmp_store(tmp_path) as store:
            persist_bet(bet, store)
            settle_bet_persistent(bet.id, "won", store)

            bets = load_bets_from_db(store, status="won")
            assert len(bets) == 1
            assert bets[0].status == "won"
            assert bets[0].settled_at is not None
            # Won: profit stays as original
            assert bets[0].profit == pytest.approx(bet.profit, abs=0.01)

    def test_settle_lost_persisted(self, tmp_path):
        bet = create_bet(stake=100, sportsbook="DK", legs=[_leg(odds=-150)])
        with _tmp_store(tmp_path) as store:
            persist_bet(bet, store)
            settle_bet_persistent(bet.id, "lost", store)

            bets = load_bets_from_db(store, status="lost")
            assert len(bets) == 1
            assert bets[0].status == "lost"
            assert bets[0].profit == pytest.approx(-100.0)
            assert bets[0].total_payout == pytest.approx(0.0)

    def test_settle_push_persisted(self, tmp_path):
        bet = create_bet(stake=100, sportsbook="DK", legs=[_leg(odds=-150)])
        with _tmp_store(tmp_path) as store:
            persist_bet(bet, store)
            settle_bet_persistent(bet.id, "push", store)

            bets = load_bets_from_db(store, status="push")
            assert len(bets) == 1
            assert bets[0].status == "push"
            assert bets[0].profit == pytest.approx(0.0)
            assert bets[0].total_payout == pytest.approx(100.0)

    def test_settled_bet_survives_refresh(self, tmp_path):
        """Settled outcome survives DB reconnection."""
        db_path = tmp_path / "settle_refresh.db"
        bet = create_bet(stake=100, sportsbook="DK", legs=[_leg(odds=200)])

        with LineStore(db_path=db_path) as store:
            persist_bet(bet, store)
            settle_bet_persistent(bet.id, "won", store)

        with LineStore(db_path=db_path) as store:
            bets = load_bets_from_db(store, status="won")
            assert len(bets) == 1
            assert bets[0].status == "won"
            assert bets[0].settled_at is not None


# ---------------------------------------------------------------------------
# Delete persistence
# ---------------------------------------------------------------------------

class TestDeletePersistence:
    def test_delete_removes_from_db(self, tmp_path):
        bet = create_bet(stake=100, sportsbook="DK", legs=[_leg()])
        with _tmp_store(tmp_path) as store:
            persist_bet(bet, store)
            assert len(load_bets_from_db(store)) == 1

            delete_bet_persistent(bet.id, store)
            assert len(load_bets_from_db(store)) == 0
            # Legs also gone
            assert store.get_bet_legs(bet.id) == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_get_bets_empty_db(self, tmp_path):
        with _tmp_store(tmp_path) as store:
            assert load_bets_from_db(store) == []

    def test_get_bets_status_filter(self, tmp_path):
        bet1 = create_bet(stake=100, sportsbook="DK", legs=[_leg()])
        bet2 = create_bet(
            stake=50, sportsbook="FD",
            legs=[_leg(odds=200, event_name="Lions @ Packers")],
        )
        with _tmp_store(tmp_path) as store:
            persist_bet(bet1, store)
            persist_bet(bet2, store)
            settle_bet_persistent(bet2.id, "won", store)

            active = load_bets_from_db(store, status="active")
            assert len(active) == 1
            assert active[0].id == bet1.id

            won = load_bets_from_db(store, status="won")
            assert len(won) == 1
            assert won[0].id == bet2.id

    def test_leg_data_preserved_through_round_trip(self, tmp_path):
        """All leg fields survive the persist → load cycle."""
        leg = _leg(
            sport="NBA",
            event_name="Lakers @ Celtics",
            sportsbook="BetMGM",
            market="Spread",
            selection="Away",
            line=-5.5,
            odds=-110,
            fetched_at="2026-02-01T12:00:00",
        )
        bet = create_bet(stake=100, sportsbook="BetMGM", legs=[leg])
        with _tmp_store(tmp_path) as store:
            persist_bet(bet, store)
            bets = load_bets_from_db(store)
            loaded_leg = bets[0].legs[0]

            assert loaded_leg["sport"] == "NBA"
            assert loaded_leg["event_name"] == "Lakers @ Celtics"
            assert loaded_leg["sportsbook"] == "BetMGM"
            assert loaded_leg["market"] == "Spread"
            assert loaded_leg["selection"] == "Away"
            assert loaded_leg["line"] == pytest.approx(-5.5)
            assert loaded_leg["odds"] == -110
            assert loaded_leg["fetched_at"] == "2026-02-01T12:00:00"

    def test_transaction_atomicity(self, tmp_path):
        """If leg insertion fails, bet should not be persisted."""
        bet = create_bet(stake=100, sportsbook="DK", legs=[_leg()])
        with _tmp_store(tmp_path) as store:
            # First persist succeeds
            persist_bet(bet, store)

            # Trying to persist same bet_id again should fail (PK conflict)
            with pytest.raises(Exception):
                persist_bet(bet, store)

            # Only one bet in DB
            assert len(load_bets_from_db(store)) == 1


def test_settled_record_counts_reconcile_with_db_rows(tmp_path):
    db_path = tmp_path / "reconcile.db"
    won = create_bet(stake=100, sportsbook="DK", legs=[_leg(odds=-150)])
    lost = create_bet(
        stake=50, sportsbook="FD",
        legs=[_leg(odds=120, event_name="A @ B")],
    )
    push = create_bet(
        stake=25, sportsbook="MGM",
        legs=[_leg(odds=-110, event_name="C @ D")],
    )

    with LineStore(db_path=db_path) as store:
        persist_bet(won, store)
        persist_bet(lost, store)
        persist_bet(push, store)
        settle_bet_persistent(won.id, "won", store)
        settle_bet_persistent(lost.id, "lost", store)
        settle_bet_persistent(push.id, "push", store)

        all_rows = store.get_bets()

    statuses = [r["status"] for r in all_rows]
    assert len(all_rows) == 3
    assert statuses.count("won") == 1
    assert statuses.count("lost") == 1
    assert statuses.count("push") == 1


def test_persist_bet_with_snapshot_rolls_back_when_clv_write_fails(
    tmp_path, monkeypatch,
):
    db_path = tmp_path / "atomic_snapshot.db"
    bet = create_bet(stake=100, sportsbook="DraftKings", legs=[_leg()])

    with LineStore(db_path=db_path) as store:
        # Ensure snapshot_pick will attempt a CLV write.
        from datetime import datetime, timezone

        from line_tracker.models import BettingLine, BetType

        ts = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
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
                timestamp=ts,
            ),
            BettingLine(
                sportsbook="FanDuel",
                sport="americanfootball_nfl",
                event="Bills @ Chiefs",
                bet_type=BetType.MONEYLINE,
                home_team="Chiefs",
                away_team="Bills",
                home_value=-145,
                away_value=125,
                timestamp=ts,
            ),
        ])

        original = store.save_clv_pick

        def boom(**kwargs):
            original(**kwargs)
            raise RuntimeError("forced after CLV write")

        monkeypatch.setattr(store, "save_clv_pick", boom)

        with pytest.raises(RuntimeError):
            persist_bet_with_snapshot(bet, store)

        assert store.get_bets() == []
        assert store.get_clv(bet.id) == []


def test_persist_bet_with_snapshot_passes_persisted_bet_id(monkeypatch):
    from line_tracker import bet_history as bh

    bet = create_bet(stake=100, sportsbook="DraftKings", legs=[_leg()])
    calls = []

    def fake_persist_bet(_bet, _store):
        return "persisted-id-123"

    def fake_snapshot_pick(_bet, _store, *, bet_id=None):
        calls.append(bet_id)

    monkeypatch.setattr(bh, "persist_bet", fake_persist_bet)
    monkeypatch.setattr(bh, "snapshot_pick", fake_snapshot_pick)

    class _DummyStore:
        def transaction(self):
            from contextlib import nullcontext
            return nullcontext()

    out = bh.persist_bet_with_snapshot(bet, _DummyStore())

    assert out == "persisted-id-123"
    assert calls == ["persisted-id-123"]
