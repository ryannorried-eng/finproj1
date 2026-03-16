"""Recommendation snapshot logging and closing-line capture service."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from line_tracker.core.math import (
    american_to_decimal,
    decimal_to_american,
    implied_probability,
)

_log = logging.getLogger(__name__)


def build_snapshot_rows(
    slate: dict,
    *,
    sport: str | None = None,
    run_ts: str | None = None,
    run_id: str | None = None,
) -> list[dict]:
    """Convert a slate result into a list of snapshot dicts for DB insertion.

    Logs every displayed candidate across Tier 1/2/3 and closest_candidates.
    Stay-away entries are excluded (low value for CLV tracking).

    Parameters
    ----------
    slate:
        The dict returned by ``build_daily_slate``.
    sport:
        Sport key (e.g. ``"basketball_nba"``).  Falls back to entry metadata.
    run_ts:
        ISO timestamp for the run (defaults to utcnow).

    Returns
    -------
    list of dicts ready for ``RecSnapshotsRepo.insert_many``.
    """
    ts = run_ts or datetime.now(timezone.utc).isoformat()

    tiers_to_log = ["tier1a", "tier1b", "tier2", "tier3"]
    seen: set[tuple] = set()
    rows: list[dict] = []

    for tier_key in tiers_to_log:
        for entry in slate.get(tier_key, []):
            _add_entry(rows, seen, entry, ts, sport, run_id=run_id)

    # Also log closest_candidates (they may overlap with tier2/tier3)
    for entry in slate.get("closest_candidates", []):
        _add_entry(rows, seen, entry, ts, sport, run_id=run_id)

    return rows


def _add_entry(
    rows: list[dict],
    seen: set[tuple],
    entry: dict,
    ts: str,
    sport: str | None,
    *,
    run_id: str | None = None,
) -> None:
    """Append a snapshot dict if not already seen."""
    dedup_key = (
        entry.get("event_id"),
        entry.get("market"),
        entry.get("selection"),
        entry.get("best_sportsbook"),
        entry.get("best_odds"),
    )
    if dedup_key in seen:
        return
    seen.add(dedup_key)

    best_odds = entry.get("best_odds", 0.0)
    odds_dec = american_to_decimal(best_odds) if best_odds else 0.0

    rows.append({
        "created_at": ts,
        "event_id": entry.get("event_id", ""),
        "sport": sport,
        "market": entry.get("market", ""),
        "selection": entry.get("selection", ""),
        "line": entry.get("line"),
        "book": entry.get("best_sportsbook", ""),
        "odds_american": best_odds,
        "odds_decimal": round(odds_dec, 4),
        "consensus_prob": entry.get("consensus_prob", 0.0),
        "breakeven_prob": entry.get("p_be"),
        "edge_pct": entry.get("edge_pct"),
        "edge_ev": entry.get("edge_ev"),
        "edge_ev_shrunk": entry.get("edge_ev_shrunk"),
        "ev_100": entry.get("ev_100"),
        "edge_z": entry.get("edge_z"),
        "quality_score": entry.get("quality_score"),
        "confidence_label": entry.get("confidence"),
        "tier": entry.get("tier"),
        "meta": None,
        "alpha_score": entry.get("alpha_score"),
        "alpha_label": entry.get("alpha_label"),
        "run_id": run_id,
    })


# ── Closing capture ──────────────────────────────────────────────────


def compute_clv_for_snapshot(
    open_american: float,
    close_american: float,
) -> dict:
    """Compute CLV metrics between open (recommendation) and close odds.

    Returns dict with:
        open_implied_prob, close_implied_prob,
        clv_delta_american, clv_delta_implied.

    Sign convention:
        clv_delta_implied > 0 ⇒ closing line moved toward your pick
        (you beat the close).
    """
    open_impl = implied_probability(open_american)
    close_impl = implied_probability(close_american)

    # For implied-prob CLV: if the close implied prob is *higher* than open,
    # the market moved toward your pick → positive CLV.
    clv_impl = close_impl - open_impl

    # American CLV needs careful handling due to sign flips.
    # Use the decimal difference as a proxy for directional movement.
    clv_american = close_american - open_american

    return {
        "open_implied_prob": round(open_impl, 6),
        "close_implied_prob": round(close_impl, 6),
        "clv_delta_american": round(clv_american, 2),
        "clv_delta_implied": round(clv_impl, 6),
    }


def capture_closing_lines(
    store,
    *,
    window_hours: int = 12,
    batch_size: int = 200,
    prioritize_hours: int = 6,
) -> dict:
    """Find unclosed snapshots and try to close them with latest odds.

    Only considers snapshots whose ``closed_at IS NULL``.  Events with
    ``commence_time`` within *prioritize_hours* (or already started) are
    processed first.  At most *batch_size* snapshots are attempted per
    call to keep each cycle bounded.

    Parameters
    ----------
    store : LineStore
        Database connection wrapper.
    window_hours : int
        (Legacy) Unused but kept for API compat.
    batch_size : int
        Maximum number of unclosed snapshots to process per cycle.
    prioritize_hours : int
        Events whose commence_time is within this many hours from now
        (or in the past) are processed first.

    Returns
    -------
    dict with ``attempted`` and ``completed`` counts.
    For backward compatibility, the dict also supports ``int()`` coercion
    returning the completed count.
    """
    unclosed = store.get_unclosed_snapshots(
        prioritize_hours=prioritize_hours,
        limit=batch_size,
    )
    attempted = len(unclosed)
    closed_count = 0

    for snap in unclosed:
        event_id = snap["event_id"]
        market = snap["market"]
        selection = snap["selection"]
        book = snap["book"]
        open_american = snap["odds_american"]

        close_american = _find_closing_odds(
            store, event_id, market, book, selection, open_american,
        )
        if close_american is None:
            continue

        clv = compute_clv_for_snapshot(open_american, close_american)
        close_dec = american_to_decimal(close_american)

        store.close_snapshot(
            snap["snapshot_id"],
            close_odds_american=close_american,
            close_odds_decimal=round(close_dec, 4),
            close_implied_prob=clv["close_implied_prob"],
            open_implied_prob=clv["open_implied_prob"],
            clv_delta_american=clv["clv_delta_american"],
            clv_delta_implied=clv["clv_delta_implied"],
        )
        closed_count += 1

    _log.info(
        "capture_closing_lines: attempted=%d completed=%d skipped=%d",
        attempted,
        closed_count,
        attempted - closed_count,
    )
    return _ClosingResult(attempted=attempted, completed=closed_count)


class _ClosingResult(dict):
    """Dict-like result that also behaves as ``int`` for backward compat.

    Legacy callers that do ``closed_count = capture_closing_lines(...)``
    and treat the return value as an int will get the *completed* count
    via ``__eq__``, ``__int__``, and comparison dunder methods.
    """

    def __init__(self, *, attempted: int, completed: int):
        super().__init__(attempted=attempted, completed=completed)
        self.attempted = attempted
        self.completed = completed

    def __int__(self) -> int:
        return self.completed

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.completed == other
        return super().__eq__(other)

    def __hash__(self) -> int:  # pragma: no cover
        return hash(self.completed)


def log_snapshot_for_picks(
    store,
    picks: list[dict],
    *,
    sport: str | None = None,
    run_ts: str | None = None,
    run_id: str | None = None,
) -> int:
    """Log snapshot rows for a list of pick dicts (Step 5 volume helper).

    Converts pick dicts (from pruning/ranking) into snapshot rows and
    inserts them.  This is called at each automation cycle; volume comes
    from running frequently, not from relaxed thresholds.

    Returns count of newly inserted rows.
    """
    ts = run_ts or datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    seen: set[tuple] = set()
    for entry in picks:
        _add_entry(rows, seen, entry, ts, sport, run_id=run_id)
    if not rows:
        return 0
    return store.log_rec_snapshots(rows)


def _resolve_side(
    ln,
    market: str,
    selection: str,
) -> float | None:
    """Return closing American odds for *selection* from a BettingLine.

    Market-aware logic — no substring matching, no silent fallbacks.

    Moneyline
        selection must exactly equal home_team or away_team.
        Returns home_value or away_value (the American odds).

    Spread
        selection must exactly equal home_team or away_team.
        Returns home_price or away_price (the juice / American odds).

    Total
        selection must be "Over" or "Under" (case-insensitive).
        Returns home_price (over juice) or away_price (under juice).

    Returns ``None`` if the selection cannot be resolved.
    """
    sel_lower = selection.lower().strip()

    if market == "moneyline":
        home = (ln.home_team or "").strip()
        away = (ln.away_team or "").strip()
        if selection.strip() == home:
            return ln.home_value
        if selection.strip() == away:
            return ln.away_value
        # Case-insensitive exact match as fallback
        if sel_lower == home.lower():
            return ln.home_value
        if sel_lower == away.lower():
            return ln.away_value
        return None

    if market == "spread":
        home = (ln.home_team or "").strip()
        away = (ln.away_team or "").strip()
        if selection.strip() == home:
            return ln.home_price
        if selection.strip() == away:
            return ln.away_price
        if sel_lower == home.lower():
            return ln.home_price
        if sel_lower == away.lower():
            return ln.away_price
        return None

    if market == "total":
        if sel_lower == "over":
            return ln.home_price
        if sel_lower == "under":
            return ln.away_price
        return None

    return None


def _find_closing_odds(
    store,
    event_id: str,
    market: str,
    book: str,
    selection: str,
    open_american: float,
) -> float | None:
    """Find the last observed American odds for the given event/market/book.

    Uses structured, market-aware matching via ``_resolve_side``.
    Returns the closing American odds, or ``None`` if no match is found,
    the selection cannot be resolved, or the odds haven't changed.
    """
    from line_tracker.models import BetType

    bt_map = {
        "moneyline": BetType.MONEYLINE,
        "spread": BetType.SPREAD,
        "total": BetType.TOTAL,
    }
    bet_type = bt_map.get(market)
    if bet_type is None:
        _log.warning(
            "clv_skip: unknown market=%r event=%s selection=%r",
            market, event_id, selection,
        )
        return None

    try:
        lines = store.get_latest_for_api_event(event_id, bet_type)
    except Exception:
        _log.warning(
            "clv_skip: lookup failed event=%s market=%s book=%s",
            event_id, market, book,
        )
        return None

    if not lines:
        return None

    for ln in lines:
        if ln.sportsbook.lower() != book.lower():
            continue

        candidate = _resolve_side(ln, market, selection)

        if candidate is None:
            _log.warning(
                "clv_skip: unresolved selection=%r market=%s "
                "home_team=%r away_team=%r event=%s book=%s",
                selection, market,
                getattr(ln, "home_team", None),
                getattr(ln, "away_team", None),
                event_id, book,
            )
            return None

        if candidate != open_american:
            _log.debug(
                "clv_match: event=%s market=%s selection=%r book=%s "
                "open=%s close=%s",
                event_id, market, selection, book,
                open_american, candidate,
            )
            return candidate

        # Odds unchanged
        return None

    return None
