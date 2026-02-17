"""Repository for `bets` and `bet_legs` tables."""

from __future__ import annotations

import sqlite3


class BetsRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def insert_bet(self, bet_row: dict) -> None:
        self._conn.execute(
            """INSERT INTO bets
               (bet_id, created_at, sportsbook, stake,
                total_odds_american, total_odds_decimal,
                potential_payout, profit, status, settled_at, outcome,
                source_page, recommendation_id, rank_at_pick,
                quality_tier_at_pick, edge_pct_at_pick,
                consensus_prob_at_pick, execution_delta_decimal)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?)""",
            (
                bet_row["bet_id"],
                bet_row["created_at"],
                bet_row["sportsbook"],
                bet_row["stake"],
                bet_row["total_odds_american"],
                bet_row["total_odds_decimal"],
                bet_row["potential_payout"],
                bet_row["profit"],
                bet_row.get("status", "active"),
                bet_row.get("settled_at"),
                bet_row.get("outcome"),
                bet_row.get("source_page"),
                bet_row.get("recommendation_id"),
                bet_row.get("rank_at_pick"),
                bet_row.get("quality_tier_at_pick"),
                bet_row.get("edge_pct_at_pick"),
                bet_row.get("consensus_prob_at_pick"),
                bet_row.get("execution_delta_decimal"),
            ),
        )

    def insert_legs(self, bet_id: str, legs: list[dict]) -> None:
        rows = [
            (
                lg["leg_id"],
                bet_id,
                lg.get("sport"),
                lg.get("market"),
                lg.get("event_name"),
                lg.get("selection"),
                lg.get("line_value"),
                lg["odds_american"],
                lg["odds_decimal"],
                lg.get("sportsbook"),
                lg.get("pick_timestamp"),
                lg.get("commence_time"),
            )
            for lg in legs
        ]
        self._conn.executemany(
            """INSERT INTO bet_legs
               (leg_id, bet_id, sport, market, event_name,
                selection, line_value, odds_american, odds_decimal,
                sportsbook, pick_timestamp, commence_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

    def get_bets(self, status: str | None = None, limit: int = 200) -> list[dict]:
        query = "SELECT * FROM bets WHERE 1=1"
        params: list = []
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_bet_legs(self, bet_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM bet_legs WHERE bet_id = ? ORDER BY leg_id",
            (bet_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def settle_bet(
        self,
        bet_id: str,
        outcome: str,
        settled_at: str,
        profit: float,
        potential_payout: float,
    ) -> None:
        self._conn.execute(
            """UPDATE bets
               SET status = ?, outcome = ?, settled_at = ?,
                   profit = ?, potential_payout = ?
               WHERE bet_id = ?""",
            (outcome, outcome, settled_at, profit, potential_payout, bet_id),
        )

    def delete_bet(self, bet_id: str) -> None:
        self._conn.execute("DELETE FROM bet_legs WHERE bet_id = ?", (bet_id,))
        self._conn.execute("DELETE FROM bets WHERE bet_id = ?", (bet_id,))
