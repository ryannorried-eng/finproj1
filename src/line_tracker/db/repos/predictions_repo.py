"""Repository for the ``model_predictions`` table."""

from __future__ import annotations

import sqlite3


class PredictionsRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def save_predictions(self, rows: list[dict]) -> int:
        """Upsert prediction rows. Returns number of rows written."""
        if not rows:
            return 0
        count = 0
        for row in rows:
            self._conn.execute(
                """INSERT INTO model_predictions
                   (event_id, sport, home_team, away_team, game_date,
                    predicted_margin, margin_sigma, home_ml_prob, away_ml_prob,
                    market_spread, home_spread_prob, away_spread_prob,
                    market_total, over_prob, under_prob,
                    model_version, model_confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (home_team, away_team, game_date, model_version)
                   DO UPDATE SET
                       event_id = excluded.event_id,
                       predicted_margin = excluded.predicted_margin,
                       margin_sigma = excluded.margin_sigma,
                       home_ml_prob = excluded.home_ml_prob,
                       away_ml_prob = excluded.away_ml_prob,
                       market_spread = excluded.market_spread,
                       home_spread_prob = excluded.home_spread_prob,
                       away_spread_prob = excluded.away_spread_prob,
                       market_total = excluded.market_total,
                       over_prob = excluded.over_prob,
                       under_prob = excluded.under_prob,
                       model_confidence = excluded.model_confidence,
                       created_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                """,
                (
                    row.get("event_id"),
                    row["sport"],
                    row["home_team"],
                    row["away_team"],
                    row["game_date"],
                    row["predicted_margin"],
                    row["margin_sigma"],
                    row["home_ml_prob"],
                    row["away_ml_prob"],
                    row.get("market_spread"),
                    row.get("home_spread_prob"),
                    row.get("away_spread_prob"),
                    row.get("market_total"),
                    row.get("over_prob"),
                    row.get("under_prob"),
                    row["model_version"],
                    row.get("model_confidence"),
                ),
            )
            count += 1
        return count

    def get_prediction(
        self, home_team: str, away_team: str, game_date: str,
    ) -> dict | None:
        """Fetch the latest prediction for a specific matchup."""
        row = self._conn.execute(
            """SELECT * FROM model_predictions
               WHERE home_team = ? AND away_team = ? AND game_date = ?
               ORDER BY created_at DESC LIMIT 1""",
            (home_team, away_team, game_date),
        ).fetchone()
        return dict(row) if row else None

    def get_predictions_for_date(self, game_date: str) -> list[dict]:
        """Fetch all predictions for a given game date."""
        rows = self._conn.execute(
            """SELECT * FROM model_predictions
               WHERE game_date = ?
               ORDER BY created_at DESC""",
            (game_date,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_event_id(self, event_id: str) -> dict | None:
        """Fetch prediction by Odds API event ID."""
        row = self._conn.execute(
            """SELECT * FROM model_predictions
               WHERE event_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (event_id,),
        ).fetchone()
        return dict(row) if row else None

    def update_event_id(self, prediction_id: int, event_id: str) -> None:
        """Link a prediction to an Odds API event."""
        self._conn.execute(
            "UPDATE model_predictions SET event_id = ? WHERE prediction_id = ?",
            (event_id, prediction_id),
        )

    def update_actual_result(
        self, prediction_id: int, actual_margin: float, actual_total: float,
    ) -> None:
        """Record the actual game result for a prediction."""
        self._conn.execute(
            """UPDATE model_predictions
               SET actual_margin = ?, actual_total = ?
               WHERE prediction_id = ?""",
            (actual_margin, actual_total, prediction_id),
        )
