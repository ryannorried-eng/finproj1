"""Tests for structured, market-aware CLV closing-line matching.

Covers:
- moneyline close match for home / away side
- spread close match for home / away side
- total close match for over / under
- unresolved / invalid selection skips without writing bad CLV data
- attempted / completed counts reflect reality
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from line_tracker.models import BettingLine, BetType
from line_tracker.services.rec_snapshot_service import (
    _find_closing_odds,
    _resolve_side,
    capture_closing_lines,
)
from line_tracker.storage import LineStore

# ── Helpers ──────────────────────────────────────────────────────────────


def _now():
    return datetime.now(timezone.utc)


def _insert_event(store, event_id: str, commence_time: str) -> None:
    """Insert event via save_lines to avoid transaction conflicts."""
    ln = BettingLine(
        sportsbook="__setup__",
        sport="nfl",
        event=f"{event_id} display",
        bet_type=BetType.MONEYLINE,
        home_team="Chiefs",
        away_team="Bills",
        home_value=-100.0,
        away_value=+100.0,
        timestamp=_now(),
        commence_time=datetime.fromisoformat(commence_time) if commence_time else None,
        api_event_id=event_id,
    )
    store.save_lines([ln])


def _insert_snapshot(
    store,
    event_id: str,
    *,
    market: str = "moneyline",
    selection: str = "Chiefs",
    book: str = "FanDuel",
    odds_american: float = -150.0,
    closed: bool = False,
) -> int:
    if odds_american < 0:
        odds_dec = abs(odds_american) / 100.0
    else:
        odds_dec = odds_american / 100.0 + 1
    snap = {
        "created_at": _now().isoformat(),
        "event_id": event_id,
        "sport": "nfl",
        "market": market,
        "selection": selection,
        "line": None,
        "book": book,
        "odds_american": odds_american,
        "odds_decimal": round(odds_dec, 4),
        "consensus_prob": 0.60,
        "breakeven_prob": 0.55,
        "edge_pct": 3.5,
        "edge_ev": 0.035,
        "edge_ev_shrunk": 0.028,
        "ev_100": 3.5,
        "edge_z": 2.1,
        "quality_score": 85,
        "confidence_label": "High",
        "tier": "tier1b",
        "meta": None,
        "alpha_score": None,
        "alpha_label": None,
    }
    store.log_rec_snapshots([snap])
    unclosed = store.rec_snapshots_repo.get_unclosed()
    sid = max(s["snapshot_id"] for s in unclosed if s["event_id"] == event_id)
    if closed:
        store.close_snapshot(
            sid,
            close_odds_american=-170.0,
            close_odds_decimal=1.5882,
            close_implied_prob=0.6296,
            open_implied_prob=0.6000,
            clv_delta_american=-20.0,
            clv_delta_implied=0.0296,
        )
    return sid


def _insert_line(
    store,
    event_id: str,
    *,
    bet_type: BetType = BetType.MONEYLINE,
    sportsbook: str = "FanDuel",
    home_team: str = "Chiefs",
    away_team: str = "Bills",
    home_value: float = -160.0,
    away_value: float = +140.0,
    home_price: float | None = None,
    away_price: float | None = None,
) -> None:
    """Insert a betting line. Also upserts the event via save_lines."""
    ln = BettingLine(
        sportsbook=sportsbook,
        sport="nfl",
        event=f"{home_team} vs {away_team}",
        bet_type=bet_type,
        home_team=home_team,
        away_team=away_team,
        home_value=home_value,
        away_value=away_value,
        timestamp=_now(),
        home_price=home_price,
        away_price=away_price,
        commence_time=_now() + timedelta(hours=2),
        api_event_id=event_id,
    )
    store.save_lines([ln])




# ── _resolve_side unit tests ────────────────────────────────────────────


class TestResolveSideMoneyline:
    """_resolve_side for moneyline uses exact team name matching."""

    def test_home_team_exact_match(self):
        ln = BettingLine(
            sportsbook="FanDuel", sport="nfl", event="X",
            bet_type=BetType.MONEYLINE,
            home_team="Chiefs", away_team="Bills",
            home_value=-150.0, away_value=+130.0,
            timestamp=_now(),
        )
        assert _resolve_side(ln, "moneyline", "Chiefs") == -150.0

    def test_away_team_exact_match(self):
        ln = BettingLine(
            sportsbook="FanDuel", sport="nfl", event="X",
            bet_type=BetType.MONEYLINE,
            home_team="Chiefs", away_team="Bills",
            home_value=-150.0, away_value=+130.0,
            timestamp=_now(),
        )
        assert _resolve_side(ln, "moneyline", "Bills") == +130.0

    def test_case_insensitive_match(self):
        ln = BettingLine(
            sportsbook="FanDuel", sport="nfl", event="X",
            bet_type=BetType.MONEYLINE,
            home_team="Chiefs", away_team="Bills",
            home_value=-150.0, away_value=+130.0,
            timestamp=_now(),
        )
        assert _resolve_side(ln, "moneyline", "chiefs") == -150.0

    def test_substring_does_not_match(self):
        """'Chief' is a substring of 'Chiefs' but must NOT match."""
        ln = BettingLine(
            sportsbook="FanDuel", sport="nfl", event="X",
            bet_type=BetType.MONEYLINE,
            home_team="Chiefs", away_team="Bills",
            home_value=-150.0, away_value=+130.0,
            timestamp=_now(),
        )
        assert _resolve_side(ln, "moneyline", "Chief") is None

    def test_unrelated_selection_returns_none(self):
        ln = BettingLine(
            sportsbook="FanDuel", sport="nfl", event="X",
            bet_type=BetType.MONEYLINE,
            home_team="Chiefs", away_team="Bills",
            home_value=-150.0, away_value=+130.0,
            timestamp=_now(),
        )
        assert _resolve_side(ln, "moneyline", "Lakers") is None


class TestResolveSideSpread:
    """_resolve_side for spread returns home_price / away_price."""

    def test_home_spread(self):
        ln = BettingLine(
            sportsbook="FanDuel", sport="nfl", event="X",
            bet_type=BetType.SPREAD,
            home_team="Chiefs", away_team="Bills",
            home_value=-3.5, away_value=+3.5,
            timestamp=_now(),
            home_price=-110.0, away_price=-110.0,
        )
        assert _resolve_side(ln, "spread", "Chiefs") == -110.0

    def test_away_spread(self):
        ln = BettingLine(
            sportsbook="FanDuel", sport="nfl", event="X",
            bet_type=BetType.SPREAD,
            home_team="Chiefs", away_team="Bills",
            home_value=-3.5, away_value=+3.5,
            timestamp=_now(),
            home_price=-110.0, away_price=-105.0,
        )
        assert _resolve_side(ln, "spread", "Bills") == -105.0

    def test_substring_does_not_match_spread(self):
        ln = BettingLine(
            sportsbook="FanDuel", sport="nfl", event="X",
            bet_type=BetType.SPREAD,
            home_team="Chiefs", away_team="Bills",
            home_value=-3.5, away_value=+3.5,
            timestamp=_now(),
            home_price=-110.0, away_price=-110.0,
        )
        assert _resolve_side(ln, "spread", "Chief") is None


class TestResolveSideTotal:
    """_resolve_side for total returns home_price (over) / away_price (under)."""

    def test_over(self):
        ln = BettingLine(
            sportsbook="FanDuel", sport="nfl", event="X",
            bet_type=BetType.TOTAL,
            home_team="Chiefs", away_team="Bills",
            home_value=47.5, away_value=47.5,
            timestamp=_now(),
            home_price=-108.0, away_price=-112.0,
        )
        assert _resolve_side(ln, "total", "Over") == -108.0

    def test_under(self):
        ln = BettingLine(
            sportsbook="FanDuel", sport="nfl", event="X",
            bet_type=BetType.TOTAL,
            home_team="Chiefs", away_team="Bills",
            home_value=47.5, away_value=47.5,
            timestamp=_now(),
            home_price=-108.0, away_price=-112.0,
        )
        assert _resolve_side(ln, "total", "Under") == -112.0

    def test_team_name_does_not_match_total(self):
        """A team name should NOT resolve for a total market."""
        ln = BettingLine(
            sportsbook="FanDuel", sport="nfl", event="X",
            bet_type=BetType.TOTAL,
            home_team="Chiefs", away_team="Bills",
            home_value=47.5, away_value=47.5,
            timestamp=_now(),
            home_price=-108.0, away_price=-112.0,
        )
        assert _resolve_side(ln, "total", "Chiefs") is None


# ── Integration: _find_closing_odds ─────────────────────────────────────


class TestFindClosingOddsMoneyline:
    def test_home_close(self, tmp_path):
        with LineStore(tmp_path / "t.db") as store:
            ct = (_now() + timedelta(hours=2)).isoformat()
            _insert_event(store, "evt1", ct)
            _insert_line(store, "evt1", home_value=-160.0, away_value=+140.0)

            result = _find_closing_odds(
                store, "evt1", "moneyline", "FanDuel", "Chiefs", -150.0,
            )
            assert result == -160.0

    def test_away_close(self, tmp_path):
        with LineStore(tmp_path / "t.db") as store:
            ct = (_now() + timedelta(hours=2)).isoformat()
            _insert_event(store, "evt1", ct)
            _insert_line(store, "evt1", home_value=-160.0, away_value=+140.0)

            result = _find_closing_odds(
                store, "evt1", "moneyline", "FanDuel", "Bills", +130.0,
            )
            assert result == +140.0


class TestFindClosingOddsSpread:
    def test_home_spread_close(self, tmp_path):
        with LineStore(tmp_path / "t.db") as store:
            ct = (_now() + timedelta(hours=2)).isoformat()
            _insert_event(store, "evt1", ct)
            _insert_line(
                store, "evt1",
                bet_type=BetType.SPREAD,
                home_value=-3.5, away_value=+3.5,
                home_price=-105.0, away_price=-115.0,
            )

            result = _find_closing_odds(
                store, "evt1", "spread", "FanDuel", "Chiefs", -110.0,
            )
            assert result == -105.0

    def test_away_spread_close(self, tmp_path):
        with LineStore(tmp_path / "t.db") as store:
            ct = (_now() + timedelta(hours=2)).isoformat()
            _insert_event(store, "evt1", ct)
            _insert_line(
                store, "evt1",
                bet_type=BetType.SPREAD,
                home_value=-3.5, away_value=+3.5,
                home_price=-105.0, away_price=-115.0,
            )

            result = _find_closing_odds(
                store, "evt1", "spread", "FanDuel", "Bills", -110.0,
            )
            assert result == -115.0


class TestFindClosingOddsTotal:
    def test_over_close(self, tmp_path):
        with LineStore(tmp_path / "t.db") as store:
            ct = (_now() + timedelta(hours=2)).isoformat()
            _insert_event(store, "evt1", ct)
            _insert_line(
                store, "evt1",
                bet_type=BetType.TOTAL,
                home_value=47.5, away_value=47.5,
                home_price=-108.0, away_price=-112.0,
            )

            result = _find_closing_odds(
                store, "evt1", "total", "FanDuel", "Over", -110.0,
            )
            assert result == -108.0

    def test_under_close(self, tmp_path):
        with LineStore(tmp_path / "t.db") as store:
            ct = (_now() + timedelta(hours=2)).isoformat()
            _insert_event(store, "evt1", ct)
            _insert_line(
                store, "evt1",
                bet_type=BetType.TOTAL,
                home_value=47.5, away_value=47.5,
                home_price=-108.0, away_price=-112.0,
            )

            result = _find_closing_odds(
                store, "evt1", "total", "FanDuel", "Under", -110.0,
            )
            assert result == -112.0


class TestFindClosingOddsSkips:
    def test_unknown_market_returns_none(self, tmp_path):
        with LineStore(tmp_path / "t.db") as store:
            result = _find_closing_odds(
                store, "evt1", "props", "FanDuel", "Chiefs", -150.0,
            )
            assert result is None

    def test_unresolved_selection_returns_none(self, tmp_path):
        """If selection doesn't match any team, returns None (no bad data)."""
        with LineStore(tmp_path / "t.db") as store:
            ct = (_now() + timedelta(hours=2)).isoformat()
            _insert_event(store, "evt1", ct)
            _insert_line(store, "evt1", home_value=-160.0, away_value=+140.0)

            result = _find_closing_odds(
                store, "evt1", "moneyline", "FanDuel", "Lakers", -150.0,
            )
            assert result is None

    def test_unchanged_odds_returns_none(self, tmp_path):
        with LineStore(tmp_path / "t.db") as store:
            ct = (_now() + timedelta(hours=2)).isoformat()
            _insert_event(store, "evt1", ct)
            _insert_line(store, "evt1", home_value=-150.0, away_value=+130.0)

            result = _find_closing_odds(
                store, "evt1", "moneyline", "FanDuel", "Chiefs", -150.0,
            )
            assert result is None

    def test_total_with_team_name_returns_none(self, tmp_path):
        """Old bug: total market with team-name selection fell back to home_value."""
        with LineStore(tmp_path / "t.db") as store:
            ct = (_now() + timedelta(hours=2)).isoformat()
            _insert_event(store, "evt1", ct)
            _insert_line(
                store, "evt1",
                bet_type=BetType.TOTAL,
                home_value=47.5, away_value=47.5,
                home_price=-108.0, away_price=-112.0,
            )

            result = _find_closing_odds(
                store, "evt1", "total", "FanDuel", "Chiefs", -110.0,
            )
            assert result is None


# ── Integration: capture_closing_lines metrics ──────────────────────────


class TestCaptureClosingLinesMetrics:
    def test_attempted_reflects_actual_snapshots(self, tmp_path):
        """attempted count should equal actual unclosed snapshots, not batch_size."""
        with LineStore(tmp_path / "t.db") as store:
            ct = (_now() + timedelta(hours=2)).isoformat()
            for i in range(3):
                eid = f"evt_{i}"
                _insert_event(store, eid, ct)
                _insert_snapshot(store, eid)

            result = capture_closing_lines(store, batch_size=200)
            # No lines to match, but attempted should be 3, not 200
            assert result["attempted"] == 3
            assert result["completed"] == 0

    def test_completed_reflects_actual_closes(self, tmp_path):
        with LineStore(tmp_path / "t.db") as store:
            ct = (_now() + timedelta(hours=2)).isoformat()

            # Event with matching close line
            _insert_event(store, "evt_match", ct)
            _insert_snapshot(
                store, "evt_match",
                market="moneyline", selection="Chiefs",
                odds_american=-150.0,
            )
            _insert_line(store, "evt_match", home_value=-160.0, away_value=+140.0)

            # Event without matching close line
            _insert_event(store, "evt_no", ct)
            _insert_snapshot(store, "evt_no")
            # No line inserted for evt_no

            result = capture_closing_lines(store)
            assert result["attempted"] == 2
            assert result["completed"] == 1

    def test_backward_compat_int_comparison(self, tmp_path):
        """Result should compare equal to int(completed) for backward compat."""
        with LineStore(tmp_path / "t.db") as store:
            result = capture_closing_lines(store)
            assert result == 0
            assert int(result) == 0

    def test_unresolved_selection_not_written(self, tmp_path):
        """Snapshot with unresolvable selection must NOT get closed with bad data."""
        with LineStore(tmp_path / "t.db") as store:
            ct = (_now() + timedelta(hours=2)).isoformat()
            _insert_event(store, "evt1", ct)
            sid = _insert_snapshot(
                store, "evt1",
                market="moneyline", selection="NonexistentTeam",
                odds_american=-150.0,
            )
            _insert_line(store, "evt1", home_value=-160.0, away_value=+140.0)

            result = capture_closing_lines(store)
            assert result["completed"] == 0

            # Verify the snapshot is still unclosed
            unclosed = store.get_unclosed_snapshots()
            assert any(s["snapshot_id"] == sid for s in unclosed)
