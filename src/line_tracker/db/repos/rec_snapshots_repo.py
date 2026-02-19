"""Repository for ``rec_snapshots`` table."""

from __future__ import annotations

import json
import sqlite3


class RecSnapshotsRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def insert_snapshot(self, snap: dict) -> int | None:
        """Insert a single recommendation snapshot.

        Uses ``INSERT OR IGNORE`` keyed on the UNIQUE constraint
        (event_id, market, selection, book, odds_american, created_at)
        to ensure idempotent logging per run.

        Returns the row id on insert, or ``None`` if ignored.
        """
        cursor = self._conn.execute(
            """INSERT OR IGNORE INTO rec_snapshots
               (created_at, event_id, sport, market, selection, line,
                book, odds_american, odds_decimal,
                consensus_prob, breakeven_prob,
                edge_pct, edge_ev, edge_ev_shrunk, ev_100,
                edge_z, quality_score, confidence_label, tier, meta,
                alpha_score, alpha_label)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snap["created_at"],
                snap["event_id"],
                snap.get("sport"),
                snap["market"],
                snap["selection"],
                snap.get("line"),
                snap["book"],
                snap["odds_american"],
                snap["odds_decimal"],
                snap["consensus_prob"],
                snap.get("breakeven_prob"),
                snap.get("edge_pct"),
                snap.get("edge_ev"),
                snap.get("edge_ev_shrunk"),
                snap.get("ev_100"),
                snap.get("edge_z"),
                snap.get("quality_score"),
                snap.get("confidence_label"),
                snap.get("tier"),
                json.dumps(snap["meta"], separators=(",", ":"))
                if snap.get("meta")
                else None,
                snap.get("alpha_score"),
                snap.get("alpha_label"),
            ),
        )
        return cursor.lastrowid if cursor.rowcount > 0 else None

    def insert_many(self, snaps: list[dict]) -> int:
        """Bulk-insert snapshots. Returns count of rows inserted."""
        count = 0
        for snap in snaps:
            result = self.insert_snapshot(snap)
            if result is not None:
                count += 1
        return count

    def update_close(
        self,
        snapshot_id: int,
        *,
        close_odds_american: float,
        close_odds_decimal: float,
        close_implied_prob: float,
        open_implied_prob: float,
        clv_delta_american: float,
        clv_delta_implied: float,
    ) -> None:
        """Write closing-line data for a snapshot row."""
        self._conn.execute(
            """UPDATE rec_snapshots
               SET close_odds_american = ?,
                   close_odds_decimal  = ?,
                   close_implied_prob  = ?,
                   open_implied_prob   = ?,
                   clv_delta_american  = ?,
                   clv_delta_implied   = ?,
                   closed_at           = CURRENT_TIMESTAMP
               WHERE snapshot_id = ?""",
            (
                close_odds_american,
                close_odds_decimal,
                close_implied_prob,
                open_implied_prob,
                clv_delta_american,
                clv_delta_implied,
                snapshot_id,
            ),
        )

    def get_unclosed(self, before_iso: str | None = None) -> list[dict]:
        """Return snapshots that haven't been closed yet.

        Optionally filter to snapshots created before *before_iso*.
        """
        query = "SELECT * FROM rec_snapshots WHERE closed_at IS NULL"
        params: list = []
        if before_iso:
            query += " AND created_at <= ?"
            params.append(before_iso)
        query += " ORDER BY created_at"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_closed(self, tier: str | None = None) -> list[dict]:
        """Return closed snapshots, optionally filtered by tier."""
        query = "SELECT * FROM rec_snapshots WHERE closed_at IS NOT NULL"
        params: list = []
        if tier:
            query += " AND tier = ?"
            params.append(tier)
        query += " ORDER BY closed_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_clv_summary_by_tier(self) -> list[dict]:
        """Aggregate CLV metrics grouped by tier.

        Returns rows with: tier, cnt, avg_clv_implied, pct_positive.
        """
        rows = self._conn.execute(
            """SELECT
                   tier,
                   COUNT(*)                                    AS cnt,
                   AVG(clv_delta_implied)                      AS avg_clv_implied,
                   100.0 * SUM(CASE WHEN clv_delta_implied > 0
                                    THEN 1 ELSE 0 END) / COUNT(*) AS pct_positive
               FROM rec_snapshots
               WHERE closed_at IS NOT NULL
               GROUP BY tier
               ORDER BY tier"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_alpha_clv_stats(self) -> list[dict]:
        """Aggregate CLV metrics grouped by alpha_label.

        Returns rows with: alpha_label, cnt, avg_clv_implied,
        avg_clv_american, avg_edge_z, avg_edge_ev_shrunk, beat_close_cnt.
        Only includes closed rows that have a non-NULL alpha_label.
        """
        rows = self._conn.execute(
            """SELECT
                   alpha_label,
                   COUNT(*)                  AS cnt,
                   AVG(clv_delta_implied)    AS avg_clv_implied,
                   AVG(clv_delta_american)   AS avg_clv_american,
                   AVG(edge_z)               AS avg_edge_z,
                   AVG(edge_ev_shrunk)       AS avg_edge_ev_shrunk,
                   SUM(CASE WHEN clv_delta_implied > 0
                            THEN 1 ELSE 0 END) AS beat_close_cnt
               FROM rec_snapshots
               WHERE closed_at IS NOT NULL
                 AND alpha_label IS NOT NULL
               GROUP BY alpha_label
               ORDER BY alpha_label"""
        ).fetchall()
        return [dict(r) for r in rows]
