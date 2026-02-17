"""Repository for `bet_clv` queries/writes."""

from __future__ import annotations

import sqlite3


class ClvRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def save_pick(self, **kwargs) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO bet_clv
               (bet_id, leg_index, event, market, pick_side,
                pick_line_value, pick_odds_american,
                pick_odds_decimal, consensus_prob_at_pick,
                market_hold_median_at_pick,
                market_volatility_sigma_at_pick,
                pick_sportsbook, sport, confidence_at_pick,
                quality_tier_at_pick, edge_pct_at_pick,
                edge_z_at_pick, books_used_at_pick,
                agreement_score_at_pick)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                kwargs["bet_id"],
                kwargs["leg_index"],
                kwargs["event"],
                kwargs["market"],
                kwargs["pick_side"],
                kwargs["pick_line_value"],
                kwargs["pick_odds_american"],
                kwargs["pick_odds_decimal"],
                kwargs["consensus_prob_at_pick"],
                kwargs["market_hold_median_at_pick"],
                kwargs["market_volatility_sigma_at_pick"],
                kwargs.get("pick_sportsbook"),
                kwargs.get("sport"),
                kwargs.get("confidence_at_pick"),
                kwargs.get("quality_tier_at_pick"),
                kwargs.get("edge_pct_at_pick"),
                kwargs.get("edge_z_at_pick"),
                kwargs.get("books_used_at_pick"),
                kwargs.get("agreement_score_at_pick"),
            ),
        )

    def close_leg(
        self,
        *,
        bet_id: str,
        leg_index: int,
        consensus_prob_close: float,
        best_odds_close_american: float,
        best_odds_close_decimal: float,
    ) -> None:
        self._conn.execute(
            """UPDATE bet_clv
               SET consensus_prob_close = ?,
                   best_odds_close_american = ?,
                   best_odds_close_decimal = ?,
                   closed_at = CURRENT_TIMESTAMP
               WHERE bet_id = ? AND leg_index = ?""",
            (
                consensus_prob_close,
                best_odds_close_american,
                best_odds_close_decimal,
                bet_id,
                leg_index,
            ),
        )

    def get_for_bet(self, bet_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM bet_clv WHERE bet_id = ? ORDER BY leg_index",
            (bet_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_closed(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM bet_clv WHERE closed_at IS NOT NULL ORDER BY closed_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
