"""Repository for ``slate_picks`` table."""

from __future__ import annotations

import json
import sqlite3


class SlatePicksRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def insert_many(self, slate_id: int, picks: list[dict]) -> int:
        """Bulk-insert picks for a slate. Returns row count."""
        rows = []
        for rank, p in enumerate(picks, start=1):
            rows.append((
                slate_id,
                rank,
                p["event"],
                p.get("api_event_id"),
                p["market"],
                p["selection"],
                p.get("line"),
                p["best_odds"],
                p["best_sportsbook"],
                p["tier"],
                p["edge_ev_100"],
                p["edge_z"],
                p["consensus_prob"],
                p["quality_score"],
                p["quality_tier"],
                p["confidence"],
                p.get("kelly_suggested"),
                p["slate_score"],
                json.dumps(p.get("avoid_reasons")) if p.get("avoid_reasons") else None,
                json.dumps(p.get("explanation"), separators=(",", ":")) if p.get("explanation") else None,
                p.get("commence_time"),
            ))

        cursor = self._conn.executemany(
            """INSERT INTO slate_picks
               (slate_id, rank, event, api_event_id, market, selection,
                line_value, best_odds, best_sportsbook, tier,
                edge_ev_100, edge_z, consensus_prob, quality_score,
                quality_tier, confidence, kelly_suggested, slate_score,
                avoid_reasons, explanation, commence_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        return cursor.rowcount
