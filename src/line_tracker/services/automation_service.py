"""Step 3 – Automation scheduler: fetch → slate → prune → snapshot loop.

Provides a single ``run_cycle()`` function that performs one complete
automation cycle.  Intended to be called repeatedly (e.g. every 15 min)
by an external scheduler or cron job via the CLI ``cycle`` command.
"""

from __future__ import annotations

from datetime import datetime, timezone

from line_tracker.core.logging import get_logger
from line_tracker.services.clv_selection_service import build_clv_filter_profile
from line_tracker.services.pick_pruning_service import prune_picks
from line_tracker.services.ranking_service import select_top_picks
from line_tracker.services.rec_snapshot_service import (
    build_snapshot_rows,
    capture_closing_lines,
)

_log = get_logger(__name__, source="automation_service")


def run_cycle(
    store,
    *,
    sport: str,
    mode: str = "Standard",
    top_n: int = 3,
    use_clv_filter: bool = True,
    dry_run: bool = False,
    closing_batch_size: int = 200,
) -> dict:
    """Execute one automation cycle: slate → prune → rank → snapshot.

    This does NOT fetch new odds (caller should fetch first or use a
    separate cron).  It reads whatever lines are in the DB and builds
    a fresh slate + snapshot from them.

    Parameters
    ----------
    store : LineStore
        Database connection wrapper.
    sport : str
        Sport key (e.g. ``"basketball_nba"``).
    mode : str
        Threshold mode (``"Standard"`` / ``"Pro"`` / ``"Auto"``).
    top_n : int
        Maximum top picks to select.
    use_clv_filter : bool
        Whether to apply CLV profile filtering during pruning.
    dry_run : bool
        If True, compute everything but skip DB writes.
    closing_batch_size : int
        Maximum unclosed snapshots to attempt closing per cycle.

    Returns
    -------
    dict with keys: ``cycle_ts``, ``slate_entries``, ``pruned_count``,
    ``top_picks``, ``snapshot_count``, ``closed_count``.
    """
    from line_tracker.services.slate_service import build_daily_slate_service

    cycle_ts = datetime.now(timezone.utc).isoformat()
    _log.info("run_cycle:start sport=%s mode=%s dry_run=%s", sport, mode, dry_run)

    # 1. Gather latest lines grouped by event
    events = store.get_events_rich(sport=sport)
    if not events:
        _log.info("run_cycle:no_events sport=%s", sport)
        return {
            "cycle_ts": cycle_ts,
            "slate_entries": 0,
            "pruned_count": 0,
            "top_picks": [],
            "snapshot_count": 0,
            "closed_count": 0,
        }

    from line_tracker.models import BetType

    lines_by_event: dict[str, list] = {}
    for ev in events:
        eid = ev.get("api_event_id") or ev.get("event_id", "")
        all_lines: list = []
        for bt in BetType:
            try:
                ls = store.get_latest_for_api_event(eid, bt)
                all_lines.extend(ls)
            except Exception:
                pass
        if all_lines:
            lines_by_event[eid] = all_lines

    # 2. Build slate
    filters = {"min_books": 4}
    slate = build_daily_slate_service(
        lines_by_event, filters=filters, mode=mode, store=store,
    )

    # Collect all non-stay-away entries
    all_entries: list[dict] = []
    for tier_key in ("tier1a", "tier1b", "tier2", "tier3"):
        all_entries.extend(slate.get(tier_key, []))
    all_entries.extend(slate.get("closest_candidates", []))

    # 3. Prune
    clv_profile = None
    if use_clv_filter:
        clv_profile = build_clv_filter_profile(store)
        # If no historical data, don't block
        if clv_profile.get("total_closed", 0) == 0:
            clv_profile = None

    pruned = prune_picks(all_entries, clv_profile=clv_profile)

    # 4. Rank and select top
    top_picks = select_top_picks(pruned, top_n=top_n)

    # 5. Snapshot (all tier entries, not just top picks — for CLV tracking)
    snapshot_count = 0
    if not dry_run:
        snap_rows = build_snapshot_rows(slate, sport=sport, run_ts=cycle_ts)
        snapshot_count = store.log_rec_snapshots(snap_rows)

    # 6. Close previously-open snapshots
    closed_count = 0
    if not dry_run:
        closed_count = capture_closing_lines(store, batch_size=closing_batch_size)

    result = {
        "cycle_ts": cycle_ts,
        "slate_entries": len(all_entries),
        "pruned_count": len(pruned),
        "top_picks": top_picks,
        "snapshot_count": snapshot_count,
        "closed_count": closed_count,
    }
    _log.info(
        "run_cycle:end entries=%d pruned=%d top=%d snapped=%d closed=%d",
        len(all_entries), len(pruned), len(top_picks),
        snapshot_count, closed_count,
    )
    return result
