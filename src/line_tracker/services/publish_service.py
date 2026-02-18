"""Slate publishing service — persist computed slates idempotently."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from line_tracker import __version__
from line_tracker.core.logging import get_logger

_log = get_logger(__name__, source="publish_service")

# Fields used to compute the deterministic slate hash.
_HASH_FIELDS = (
    "event", "market", "selection", "best_odds", "tier", "edge_z",
)


def _compute_slate_hash(picks: list[dict]) -> str:
    """Deterministic SHA-256 hash over the ordered, stable pick fields."""
    canonical = []
    for p in sorted(picks, key=lambda p: (p.get("event", ""), p.get("market", ""), p.get("selection", ""))):
        canonical.append(
            "|".join(str(p.get(f, "")) for f in _HASH_FIELDS)
        )
    payload = "\n".join(canonical).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _parse_sqlite_timestamp(ts: str) -> datetime:
    """Parse SQLite ``CURRENT_TIMESTAMP`` format (``YYYY-MM-DD HH:MM:SS``)
    or ISO-8601 into a timezone-aware UTC datetime."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # Fallback: fromisoformat handles most other valid shapes.
    return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)


def publish_slate(
    store,
    *,
    slate_date: str,
    sport: str,
    mode: str,
    thresholds: dict,
    engine_config: dict | None = None,
    picks: list[dict],
    min_publish_interval_seconds: int | None = None,
) -> int:
    """Persist a computed slate idempotently.

    Parameters
    ----------
    store : LineStore
        Database connection wrapper.
    slate_date : str
        ISO date string (YYYY-MM-DD).
    sport : str
        Sport key (e.g. ``"americanfootball_nfl"``).
    mode : str
        Threshold mode used (``"Standard"`` / ``"Pro"`` / ``"Auto"``).
    thresholds : dict
        Serialisable representation of the TierThresholds used.
    engine_config : dict, optional
        Additional engine configuration snapshot.
    picks : list[dict]
        Slate entries from ``build_daily_slate`` (all tiers including stay_away).
    min_publish_interval_seconds : int, optional
        When set, if a slate for the same (slate_date, sport, mode) was
        published less than this many seconds ago, return the existing
        slate_id even if the slate_hash differs.

    Returns
    -------
    int
        The ``published_slates.id`` (existing or newly created).
    """
    slate_hash = _compute_slate_hash(picks)
    engine_config = engine_config or {}
    code_version = __version__

    # Idempotency check: return existing slate_id if hash matches.
    existing = store.slates_repo.get_by_unique(
        slate_date, sport, mode, slate_hash,
    )
    if existing is not None:
        _log.info(
            "publish_slate idempotent hit slate_id=%s hash=%s",
            existing, slate_hash,
        )
        return existing

    # Throttle check: skip if last publish was too recent.
    if min_publish_interval_seconds is not None:
        latest = store.slates_repo.get_latest_for_day(
            slate_date, sport, mode,
        )
        if latest is not None:
            latest_id, created_at_str, _ = latest
            created_at = _parse_sqlite_timestamp(created_at_str)
            age = (datetime.now(timezone.utc) - created_at).total_seconds()
            if age < min_publish_interval_seconds:
                _log.info(
                    "publish_slate throttled slate_id=%s age=%.0fs limit=%ds",
                    latest_id, age, min_publish_interval_seconds,
                )
                return latest_id

    # New slate — insert in a single transaction.
    with store.transaction():
        slate_id = store.slates_repo.insert(
            slate_date=slate_date,
            sport=sport,
            mode=mode,
            thresholds=thresholds,
            slate_hash=slate_hash,
            engine_config=engine_config,
            code_version=code_version,
        )
        if picks:
            store.slate_picks_repo.insert_many(slate_id, picks)

    _log.info(
        "publish_slate created slate_id=%s picks=%d hash=%s",
        slate_id, len(picks), slate_hash,
    )
    return slate_id
