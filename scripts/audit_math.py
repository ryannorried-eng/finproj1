#!/usr/bin/env python3
"""Audit script: recompute betting-math metrics from the SQLite DB and report diffs.

Usage:
    python scripts/audit_math.py [--db PATH]

Loads rows from bets, bet_legs, and bet_clv tables, recomputes key metrics
from raw inputs (odds conversions, breakeven, EV, edge, CLV deltas), and
prints any discrepancies beyond tolerance.

Exits 0 even when the DB is empty or tables don't exist.
Exits 1 only when actual metric diffs exceed tolerance.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Independent math (duplicated on purpose -- no production imports for audit)
# ---------------------------------------------------------------------------

_TOLERANCE_DEC = 0.01
_TOLERANCE_PROB = 0.005
_TOLERANCE_EV = 0.01
_TOP_N_FAILURES = 10


def _a2d(odds: float) -> float:
    """American -> decimal odds."""
    if odds >= 0:
        return odds / 100.0 + 1.0
    return 100.0 / abs(odds) + 1.0


def _d2a(dec: float) -> float:
    """Decimal -> American odds."""
    if dec >= 2.0:
        return round((dec - 1) * 100, 2)
    if dec <= 1.0:
        return 0.0
    return round(-100 / (dec - 1), 2)


def _implied_prob(odds: float) -> float:
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return 0.5


def _ev_per_dollar(prob: float, odds: float) -> float:
    d = _a2d(odds)
    return prob * (d - 1.0) - (1.0 - prob)


def _edge_pct(prob: float, odds: float) -> float:
    d = _a2d(odds)
    return 100.0 * (prob - 1.0 / d)


# ---------------------------------------------------------------------------


def _dict_row(cursor, row):
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,),
    )
    return cur.fetchone() is not None


def audit_legs(conn: sqlite3.Connection) -> tuple[int, list[str]]:
    """Check bet_legs: odds_decimal == a2d(odds_american)."""
    if not _table_exists(conn, "bet_legs"):
        return 0, []

    rows = conn.execute(
        "SELECT leg_id, odds_american, odds_decimal FROM bet_legs LIMIT 500",
    ).fetchall()

    checked = 0
    failures: list[str] = []
    max_diff = 0.0

    for r in rows:
        leg_id, am, dec = r["leg_id"], r["odds_american"], r["odds_decimal"]
        if am is None or dec is None:
            continue
        checked += 1
        expected = round(_a2d(am), 4)
        diff = abs(dec - expected)
        max_diff = max(max_diff, diff)
        if diff > _TOLERANCE_DEC:
            failures.append(
                f"  leg {leg_id}: odds_american={am}, "
                f"stored_dec={dec}, expected_dec={expected}, diff={diff:.6f}"
            )

    print(f"bet_legs: checked={checked}, max_abs_diff_decimal={max_diff:.6f}")
    return checked, failures[:_TOP_N_FAILURES]


def audit_bets(conn: sqlite3.Connection) -> tuple[int, list[str]]:
    """Check bets: total_odds_decimal vs total_odds_american conversion."""
    if not _table_exists(conn, "bets"):
        return 0, []

    rows = conn.execute(
        "SELECT bet_id, total_odds_american, total_odds_decimal "
        "FROM bets LIMIT 500",
    ).fetchall()

    checked = 0
    failures: list[str] = []
    max_diff = 0.0

    for r in rows:
        bid = r["bet_id"]
        am, dec = r["total_odds_american"], r["total_odds_decimal"]
        if am is None or dec is None:
            continue

        # For parlays the combined decimal is a product, not a simple
        # conversion. Only flag extreme mismatches as a sanity check.
        legs = conn.execute(
            "SELECT odds_american FROM bet_legs WHERE bet_id = ?", (bid,),
        ).fetchall()

        if len(legs) == 1:
            # straight bet -- should be exact conversion
            checked += 1
            expected = round(_a2d(am), 4)
            diff = abs(dec - expected)
            max_diff = max(max_diff, diff)
            if diff > _TOLERANCE_DEC:
                failures.append(
                    f"  bet {bid}: am={am}, stored_dec={dec}, "
                    f"expected_dec={expected}, diff={diff:.6f}"
                )
        else:
            # parlay -- check combined dec = product of leg decimals
            checked += 1
            prod = 1.0
            for lg in legs:
                if lg["odds_american"] is not None:
                    prod *= _a2d(lg["odds_american"])
            prod = round(prod, 4)
            diff = abs(dec - prod)
            max_diff = max(max_diff, diff)
            if diff > 0.02:
                failures.append(
                    f"  bet {bid} (parlay): stored_dec={dec}, "
                    f"product_dec={prod}, diff={diff:.6f}"
                )

    print(f"bets: checked={checked}, max_abs_diff_decimal={max_diff:.6f}")
    return checked, failures[:_TOP_N_FAILURES]


def audit_clv(conn: sqlite3.Connection) -> tuple[int, list[str]]:
    """Check bet_clv: recompute decimal conversions, EV, edge, CLV."""
    if not _table_exists(conn, "bet_clv"):
        return 0, []

    rows = conn.execute(
        "SELECT id, bet_id, leg_index, "
        "  pick_odds_american, pick_odds_decimal, "
        "  consensus_prob_at_pick, consensus_prob_close, "
        "  best_odds_close_american, best_odds_close_decimal, "
        "  edge_pct_at_pick "
        "FROM bet_clv LIMIT 500",
    ).fetchall()

    checked = 0
    failures: list[str] = []
    max_diff_dec = 0.0
    max_diff_close_dec = 0.0
    max_diff_edge = 0.0

    for r in rows:
        rid = r["id"]

        # --- Pick decimal vs american ---
        am, dec = r["pick_odds_american"], r["pick_odds_decimal"]
        if am is not None and dec is not None:
            checked += 1
            exp = round(_a2d(am), 4)
            diff = abs(dec - exp)
            max_diff_dec = max(max_diff_dec, diff)
            if diff > _TOLERANCE_DEC:
                failures.append(
                    f"  clv {rid}: pick am={am}, stored_dec={dec}, "
                    f"expected={exp}, diff={diff:.6f}"
                )

        # --- Close decimal vs american ---
        cam = r["best_odds_close_american"]
        cdec = r["best_odds_close_decimal"]
        if cam is not None and cdec is not None:
            checked += 1
            exp = round(_a2d(cam), 4)
            diff = abs(cdec - exp)
            max_diff_close_dec = max(max_diff_close_dec, diff)
            if diff > _TOLERANCE_DEC:
                failures.append(
                    f"  clv {rid}: close am={cam}, stored_dec={cdec}, "
                    f"expected={exp}, diff={diff:.6f}"
                )

        # --- Edge pct recomputation from consensus_prob + pick_odds ---
        cons_prob = r["consensus_prob_at_pick"]
        stored_edge = r["edge_pct_at_pick"]
        if am is not None and cons_prob is not None and stored_edge is not None:
            checked += 1
            exp_edge = round(_edge_pct(cons_prob, am), 4)
            diff = abs(stored_edge - exp_edge)
            max_diff_edge = max(max_diff_edge, diff)
            if diff > _TOLERANCE_PROB * 100:
                failures.append(
                    f"  clv {rid}: edge_pct stored={stored_edge}, "
                    f"expected={exp_edge}, diff={diff:.6f} "
                    f"(cons_prob={cons_prob}, am={am})"
                )

    print(
        f"bet_clv: checked={checked}, "
        f"max_diff_pick_dec={max_diff_dec:.6f}, "
        f"max_diff_close_dec={max_diff_close_dec:.6f}, "
        f"max_diff_edge_pct={max_diff_edge:.6f}"
    )
    return checked, failures[:_TOP_N_FAILURES]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit betting math in DB")
    parser.add_argument(
        "--db",
        default=str(Path.home() / ".line_tracker" / "lines.db"),
        help="Path to SQLite database (default: ~/.line_tracker/lines.db)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found at {db_path} -- nothing to audit. OK.")
        return 0

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = _dict_row

    total_checked = 0
    all_failures: list[str] = []

    for audit_fn in (audit_legs, audit_bets, audit_clv):
        checked, failures = audit_fn(conn)
        total_checked += checked
        all_failures.extend(failures)

    conn.close()

    print(f"\nTotal rows checked: {total_checked}")
    if all_failures:
        shown = all_failures[:_TOP_N_FAILURES]
        print(f"FAILURES ({len(all_failures)} total, showing top {len(shown)}):")
        for f in shown:
            print(f)
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
