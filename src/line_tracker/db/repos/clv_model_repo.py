"""Repository for the ``clv_model_scores`` table."""

from __future__ import annotations

import json
import sqlite3


class ClvModelRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def save_scores(self, scores: list[dict]) -> int:
        """Upsert model scores. Returns count of rows written."""
        count = 0
        for s in scores:
            cursor = self._conn.execute(
                """INSERT INTO clv_model_scores
                       (market, selection_type, feature_json,
                        predicted_clv_positive_prob, model_version)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(market, selection_type, model_version)
                   DO UPDATE SET
                       feature_json = excluded.feature_json,
                       predicted_clv_positive_prob =
                           excluded.predicted_clv_positive_prob,
                       created_at = CURRENT_TIMESTAMP""",
                (
                    s["market"],
                    s.get("selection_type"),
                    json.dumps(s.get("features", {}), separators=(",", ":")),
                    s["predicted_clv_positive_prob"],
                    s.get("model_version", "v1"),
                ),
            )
            if cursor.rowcount > 0:
                count += 1
        return count

    def get_score(
        self, market: str, selection_type: str | None,
    ) -> dict | None:
        """Look up a single model score."""
        row = self._conn.execute(
            """SELECT * FROM clv_model_scores
               WHERE market = ? AND selection_type IS ?
               ORDER BY created_at DESC LIMIT 1""",
            (market, selection_type),
        ).fetchone()
        return dict(row) if row else None

    def get_all_scores(self) -> list[dict]:
        """Return all model scores."""
        rows = self._conn.execute(
            "SELECT * FROM clv_model_scores ORDER BY market, selection_type"
        ).fetchall()
        return [dict(r) for r in rows]
