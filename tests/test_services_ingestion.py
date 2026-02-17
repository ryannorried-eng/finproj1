"""Tests for ingestion service wrapper."""

from datetime import datetime, timezone

from line_tracker.models import BettingLine, BetType
from line_tracker.services import ingestion_service


class _FakeStore:
    def __init__(self):
        self.saved = None

    def save_lines(self, lines):
        self.saved = lines
        return len(lines)


class _FakeClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def get_odds(self, sport):
        return [
            BettingLine(
                sportsbook="DraftKings",
                sport=sport,
                event="Bills @ Chiefs",
                bet_type=BetType.MONEYLINE,
                home_team="Chiefs",
                away_team="Bills",
                home_value=-150,
                away_value=130,
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]


def test_fetch_and_persist_snapshot_uses_client_and_store(monkeypatch):
    monkeypatch.setattr(ingestion_service, "OddsClient", _FakeClient)

    store = _FakeStore()
    out = ingestion_service.fetch_and_persist_snapshot(
        store,
        api_key="k",
        sport="americanfootball_nfl",
    )

    assert out["saved_count"] == 1
    assert len(out["lines"]) == 1
    assert store.saved is out["lines"]
