"""Step 3 – Automation scheduler: fetch → slate → prune → snapshot loop.

Provides a single ``run_cycle()`` function that performs one complete
automation cycle.  Intended to be called repeatedly (e.g. every 15 min)
by an external scheduler or cron job via the CLI ``cycle`` command.
"""

from __future__ import annotations

import statistics
import uuid
from collections import Counter
from datetime import datetime, timezone

from line_tracker.core.logging import get_logger
from line_tracker.services.clv_selection_service import build_clv_filter_profile
from line_tracker.services.pick_pruning_service import prune_picks_with_reasons
from line_tracker.services.ranking_service import select_top_picks
from line_tracker.services.rec_snapshot_service import (
    build_snapshot_rows,
    capture_closing_lines,
)

_log = get_logger(__name__, source="automation_service")


def _log_candidate_summary(entries: list[dict]) -> None:
    """Log a compact candidate summary before pruning."""
    tier_counts = dict(Counter(e.get("tier", "?") for e in entries))
    market_counts = dict(Counter(e.get("market", "?") for e in entries))
    alpha_counts = dict(Counter(e.get("alpha_label", "?") for e in entries))

    edge_zs = [e.get("edge_z", 0.0) for e in entries if e.get("edge_z") is not None]
    ev_shrunk = [e.get("edge_ev_shrunk", 0.0) for e in entries if e.get("edge_ev_shrunk") is not None]

    def _mmm(vals: list[float]) -> str:
        if not vals:
            return "n/a"
        return f"{min(vals):.3f}/{statistics.median(vals):.3f}/{max(vals):.3f}"

    tier_str = " ".join(f"{k}={v}" for k, v in sorted(tier_counts.items()))
    mkt_str = " ".join(f"{k}={v}" for k, v in sorted(market_counts.items()))
    alpha_str = " ".join(f"{k}={v}" for k, v in sorted(alpha_counts.items()))

    _log.info(
        "candidate_summary n=%d tiers=[%s] markets=[%s] alpha=[%s] "
        "edge_z(min/med/max)=%s ev_shrunk(min/med/max)=%s",
        len(entries), tier_str, mkt_str, alpha_str,
        _mmm(edge_zs), _mmm(ev_shrunk),
    )


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
    run_id = uuid.uuid4().hex
    _log.info("run_cycle:start sport=%s mode=%s dry_run=%s run_id=%s", sport, mode, dry_run, run_id)

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

    # Debug: log first few slate entries before pruning
    if dry_run:
        _debug_fields = (
            "market", "selection", "tier", "alpha_label", "alpha_score",
            "edge_z", "edge_ev_shrunk", "quality_score", "books_used",
            "market_hold_median",
        )
        for entry in all_entries[:3]:
            vals = " ".join(f"{f}={entry.get(f)}" for f in _debug_fields)
            _log.info("dry_run:slate_entry %s", vals)

    # Compact candidate summary before pruning
    if dry_run and all_entries:
        _log_candidate_summary(all_entries)

    # 3. Prune
    clv_profile = None
    if use_clv_filter:
        clv_profile = build_clv_filter_profile(store)
        # If no historical data, don't block
        if clv_profile.get("total_closed", 0) == 0:
            clv_profile = None

    pruned, prune_reasons = prune_picks_with_reasons(
        all_entries, clv_profile=clv_profile,
    )

    # Log prune breakdown when nothing survives or in dry_run mode
    if dry_run or len(pruned) == 0:
        parts = " ".join(f"{k}={v}" for k, v in prune_reasons.items())
        _log.info("prune_breakdown %s", parts)

    # 4. Rank and select top
    top_picks = select_top_picks(pruned, top_n=top_n)

    # 5. Snapshot (all tier entries, not just top picks — for CLV tracking)
    snapshot_count = 0
    if not dry_run:
        snap_rows = build_snapshot_rows(slate, sport=sport, run_ts=cycle_ts, run_id=run_id)
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


def kpi_report(store, *, last_hours: int = 24) -> dict:
    """Compute KPI summary for recent automation cycles.

    Parameters
    ----------
    store : LineStore
        Database connection wrapper.
    last_hours : int
        Lookback window in hours (default 24).

    Returns
    -------
    dict with keys: ``last_hours``, ``cycles``, ``total_snapshots``,
    ``pct_closed``, ``closed_count``, ``clv_positive``, ``clv_rate``,
    ``avg_picks_per_cycle``.
    """
    row = store._conn.execute(
        """SELECT
               COUNT(*)                                              AS total,
               COUNT(DISTINCT run_id)                                AS cycles,
               SUM(CASE WHEN closed_at IS NOT NULL THEN 1 ELSE 0 END) AS closed,
               SUM(CASE WHEN closed_at IS NOT NULL
                              AND clv_delta_implied > 0
                         THEN 1 ELSE 0 END)                          AS clv_pos
           FROM rec_snapshots
           WHERE created_at >= datetime('now', ?)
             AND run_id IS NOT NULL""",
        (f"-{last_hours} hours",),
    ).fetchone()

    total = row["total"] or 0
    cycles = row["cycles"] or 0
    closed = row["closed"] or 0
    clv_pos = row["clv_pos"] or 0

    return {
        "last_hours": last_hours,
        "cycles": cycles,
        "total_snapshots": total,
        "pct_closed": (closed / total * 100) if total else 0.0,
        "closed_count": closed,
        "clv_positive": clv_pos,
        "clv_rate": (clv_pos / closed * 100) if closed else 0.0,
        "avg_picks_per_cycle": total / cycles if cycles else 0.0,
    }
