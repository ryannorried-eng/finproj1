"""Odds ingestion orchestration service."""

from __future__ import annotations

from line_tracker.scraper import OddsClient


def fetch_and_persist_snapshot(
    store,
    api_key: str,
    *,
    sport: str,
) -> dict[str, object]:
    """Fetch latest odds for a sport and persist to storage.

    Returns a summary with fetched lines and persisted row count.
    """
    with OddsClient(api_key=api_key) as client:
        lines = client.get_odds(sport=sport)

    if not lines:
        return {"lines": [], "saved_count": 0}

    saved_count = store.save_lines(lines)
    return {"lines": lines, "saved_count": saved_count}
