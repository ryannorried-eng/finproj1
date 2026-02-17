"""Odds ingestion orchestration service."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter

from line_tracker.core.logging import get_logger
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
    log = get_logger(__name__, source="ingestion_service")
    started = perf_counter()
    log.info("fetch_and_persist_snapshot:start sport=%s", sport)

    fetch_started = perf_counter()
    with OddsClient(api_key=api_key) as client:
        lines = client.get_odds(sport=sport)
    fetch_ms = round((perf_counter() - fetch_started) * 1000.0, 2)

    if not lines:
        log.info(
            "fetch_and_persist_snapshot:end sport=%s fetched=0 fetch_ms=%s",
            sport, fetch_ms,
        )
        return {"lines": [], "saved_count": 0}

    saved_count = store.save_lines(lines)
    snapshot_ts = datetime.now(timezone.utc).isoformat()
    total_ms = round((perf_counter() - started) * 1000.0, 2)
    log.info(
        "fetch_and_persist_snapshot:end sport=%s fetched=%s"
        " saved=%s fetch_ms=%s total_ms=%s snapshot_ts=%s",
        sport,
        len(lines),
        saved_count,
        fetch_ms,
        total_ms,
        snapshot_ts,
    )
    return {"lines": lines, "saved_count": saved_count}
