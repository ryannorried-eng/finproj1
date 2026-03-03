"""Step 6 – Outcome feedback loop: settlement import + ROI reporting.

Imports game outcomes from a CSV file, links them to rec_snapshots,
and produces ROI reports by tier and alpha label.
"""

from __future__ import annotations

import csv
from pathlib import Path


def import_outcomes_csv(store, csv_path: str) -> int:
    """Read a CSV and import outcomes into the database.

    Expected CSV columns: ``event_id, market, selection, result``
    (with optional ``settled_at`` and ``line_value``).  ``result`` must be one of:
    ``win``, ``loss``, ``push``.

    Returns count of rows upserted.
    """
    path = Path(csv_path)
    rows: list[dict] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = row.get("result", "").strip().lower()
            if result not in ("win", "loss", "push"):
                continue
            raw_line = row.get("line_value", "").strip() if "line_value" in row else ""
            line_value = float(raw_line) if raw_line else None
            rows.append({
                "event_id": row["event_id"].strip(),
                "market": row["market"].strip(),
                "selection": row["selection"].strip(),
                "result": result,
                "settled_at": row.get("settled_at", "").strip() or None,
                "line_value": line_value,
            })
    if not rows:
        return 0
    return store.import_outcomes(rows)


def link_outcomes(store) -> int:
    """Link imported outcomes to rec_snapshots and compute actual_roi.

    For spread/total markets, requires line_value to match rec_snapshots.line.
    For moneyline (and other non-line markets), treats None line_value as
    wildcard so only event_id/market/selection need to match.

    Returns count of snapshot rows updated.
    """
    # Non-line markets (moneyline, etc.): match without line_value.
    updated = store._conn.execute(
        """UPDATE rec_snapshots
           SET outcome_result = o.result,
               actual_roi = CASE
                   WHEN o.result = 'win'  THEN (odds_decimal - 1.0)
                   WHEN o.result = 'loss' THEN -1.0
                   ELSE 0.0
               END
           FROM outcomes o
           WHERE rec_snapshots.event_id = o.event_id
             AND rec_snapshots.market   = o.market
             AND rec_snapshots.selection = o.selection
             AND rec_snapshots.outcome_result IS NULL
             AND o.market NOT IN ('spread', 'total')"""
    ).rowcount

    # Line markets (spread, total): require line_value match.
    # SQLite IS handles NULL=NULL correctly.
    updated += store._conn.execute(
        """UPDATE rec_snapshots
           SET outcome_result = o.result,
               actual_roi = CASE
                   WHEN o.result = 'win'  THEN (odds_decimal - 1.0)
                   WHEN o.result = 'loss' THEN -1.0
                   ELSE 0.0
               END
           FROM outcomes o
           WHERE rec_snapshots.event_id = o.event_id
             AND rec_snapshots.market   = o.market
             AND rec_snapshots.selection = o.selection
             AND rec_snapshots.outcome_result IS NULL
             AND o.market IN ('spread', 'total')
             AND rec_snapshots.line IS o.line_value"""
    ).rowcount

    store._conn.commit()
    return updated


def roi_report(store) -> dict:
    """Build an ROI report from snapshots that have linked outcomes.

    Returns a dict with:
      - ``overall``: {total, wins, losses, pushes, net_units, roi_pct}
      - ``by_tier``: {tier: same}
      - ``by_alpha``: {alpha_label: same}
    """
    # Query all snapshots with outcome data
    rows = store._conn.execute(
        """SELECT tier, alpha_label, outcome_result, actual_roi, odds_decimal
           FROM rec_snapshots
           WHERE outcome_result IS NOT NULL"""
    ).fetchall()

    if not rows:
        return {
            "overall": _empty_roi_bucket(),
            "by_tier": {},
            "by_alpha": {},
        }

    all_rows = [dict(r) for r in rows]

    by_tier: dict[str, list[dict]] = {}
    by_alpha: dict[str, list[dict]] = {}
    for r in all_rows:
        t = r.get("tier") or "unknown"
        a = r.get("alpha_label") or "unknown"
        by_tier.setdefault(t, []).append(r)
        by_alpha.setdefault(a, []).append(r)

    return {
        "overall": _compute_roi_bucket(all_rows),
        "by_tier": {k: _compute_roi_bucket(v) for k, v in sorted(by_tier.items())},
        "by_alpha": {k: _compute_roi_bucket(v) for k, v in sorted(by_alpha.items())},
    }


def _compute_roi_bucket(rows: list[dict]) -> dict:
    """Compute ROI stats for a list of outcome-linked snapshot rows."""
    total = len(rows)
    wins = sum(1 for r in rows if r.get("outcome_result") == "win")
    losses = sum(1 for r in rows if r.get("outcome_result") == "loss")
    pushes = sum(1 for r in rows if r.get("outcome_result") == "push")
    net_units = sum(r.get("actual_roi") or 0 for r in rows)
    roi_pct = round(100.0 * net_units / total, 2) if total else 0.0
    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "net_units": round(net_units, 4),
        "roi_pct": roi_pct,
    }


def _empty_roi_bucket() -> dict:
    return {
        "total": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "net_units": 0.0,
        "roi_pct": 0.0,
    }
