"""Repository functions for the ``mlb_model_predictions`` table."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone


def upsert_prediction(conn: sqlite3.Connection, pred: dict) -> None:
    """Insert or replace a prediction row."""
    conn.execute(
        """INSERT OR REPLACE INTO mlb_model_predictions (
            game_id, game_date, home_team, away_team, commence_time,
            model_home_win_prob, model_run_diff, model_total_runs,
            model_implied_home_odds,
            market_home_ml, market_away_ml, market_spread, market_total,
            ml_edge, total_edge, model_spread_pick, confidence, data_source,
            recommended_prob, model_disagreement, flagged, ensemble_prob
        ) VALUES (
            :game_id, :game_date, :home_team, :away_team, :commence_time,
            :model_home_win_prob, :model_run_diff, :model_total_runs,
            :model_implied_home_odds,
            :market_home_ml, :market_away_ml, :market_spread, :market_total,
            :ml_edge, :total_edge, :model_spread_pick, :confidence, :data_source,
            :recommended_prob, :model_disagreement, :flagged, :ensemble_prob
        )""",
        {
            "game_id": pred.get("game_id", ""),
            "game_date": pred.get("game_date", ""),
            "home_team": pred.get("home_team", ""),
            "away_team": pred.get("away_team", ""),
            "commence_time": pred.get("commence_time"),
            "model_home_win_prob": pred.get("recommended_prob") or pred.get("model_home_win_prob"),
            "model_run_diff": pred.get("model_run_diff"),
            "model_total_runs": pred.get("model_total_runs"),
            "model_implied_home_odds": pred.get("model_implied_home_odds"),
            "market_home_ml": pred.get("market_home_ml"),
            "market_away_ml": pred.get("market_away_ml"),
            "market_spread": pred.get("market_spread"),
            "market_total": pred.get("market_total"),
            "ml_edge": pred.get("ml_edge"),
            "total_edge": pred.get("total_edge"),
            "model_spread_pick": pred.get("model_spread_pick"),
            "confidence": pred.get("confidence"),
            "data_source": pred.get("data_source"),
            "recommended_prob": pred.get("recommended_prob"),
            "model_disagreement": pred.get("model_disagreement"),
            "flagged": 1 if pred.get("flagged") else 0,
            "ensemble_prob": pred.get("ensemble_prob"),
        },
    )
    conn.commit()


def get_predictions_by_date(
    conn: sqlite3.Connection, game_date: str
) -> list[dict]:
    """Fetch all predictions for a given game date."""
    rows = conn.execute(
        "SELECT * FROM mlb_model_predictions WHERE game_date = ? ORDER BY commence_time",
        (game_date,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_unsettled_predictions(conn: sqlite3.Connection) -> list[dict]:
    """Fetch predictions from past dates that have not yet been settled."""
    today = date.today().isoformat()
    rows = conn.execute(
        """SELECT * FROM mlb_model_predictions
           WHERE actual_home_score IS NULL AND game_date < ?""",
        (today,),
    ).fetchall()
    return [dict(r) for r in rows]


def settle_prediction(
    conn: sqlite3.Connection,
    game_id: str,
    home_score: int,
    away_score: int,
) -> None:
    """Record the actual result and compute correctness flags."""
    # Fetch existing prediction to compare
    row = conn.execute(
        "SELECT model_home_win_prob, model_total_runs, market_total FROM mlb_model_predictions WHERE game_id = ?",
        (game_id,),
    ).fetchone()

    if row is None:
        return

    model_prob = row["model_home_win_prob"] or 0.5  # stores recommended_prob (ensemble-adjusted)
    model_total = row["model_total_runs"] or 0.0
    market_total = row["market_total"]

    actual_home_win = home_score > away_score
    model_home_win = model_prob > 0.5
    home_win_correct = 1 if actual_home_win == model_home_win else 0

    actual_total = home_score + away_score
    # Grade totals as over/under direction vs the market line, not as exact prediction.
    # A model total > market total = model predicts OVER; correct if actual also went over.
    if market_total is not None and market_total > 0:
        model_over = model_total > market_total
        actual_over = actual_total > market_total
        total_correct = 1 if model_over == actual_over else 0
    else:
        # No market total available; fall back to within-1-run accuracy
        total_correct = 1 if abs(model_total - actual_total) <= 1.0 else 0

    settled_at = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """UPDATE mlb_model_predictions
           SET actual_home_score = ?,
               actual_away_score = ?,
               home_win_correct = ?,
               total_correct = ?,
               settled_at = ?
           WHERE game_id = ?""",
        (home_score, away_score, home_win_correct, total_correct, settled_at, game_id),
    )
    conn.commit()


def get_model_accuracy(conn: sqlite3.Connection) -> dict:
    """Return accuracy stats from all settled predictions.

    Returns
    -------
    dict with moneyline_acc, total_acc, n_settled.
    """
    rows = conn.execute(
        """SELECT home_win_correct, total_correct
           FROM mlb_model_predictions
           WHERE actual_home_score IS NOT NULL""",
    ).fetchall()

    n = len(rows)
    if n == 0:
        return {"moneyline_acc": 0.0, "total_acc": 0.0, "n_settled": 0}

    ml_correct = sum(r["home_win_correct"] or 0 for r in rows)
    tot_correct = sum(r["total_correct"] or 0 for r in rows)

    return {
        "moneyline_acc": ml_correct / n,
        "total_acc": tot_correct / n,
        "n_settled": n,
    }
