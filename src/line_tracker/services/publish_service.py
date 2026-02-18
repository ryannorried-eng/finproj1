"""Slate publishing service — persist computed slates idempotently."""

from __future__ import annotations

import hashlib
import json

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


def publish_slate(
    store,
    *,
    slate_date: str,
    sport: str,
    mode: str,
    thresholds: dict,
    engine_config: dict | None = None,
    picks: list[dict],
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
