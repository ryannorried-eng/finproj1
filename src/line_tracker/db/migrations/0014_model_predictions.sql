CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    event_id            TEXT,
    sport               TEXT    NOT NULL,
    home_team           TEXT    NOT NULL,
    away_team           TEXT    NOT NULL,
    game_date           TEXT    NOT NULL,
    predicted_margin    REAL    NOT NULL,
    margin_sigma        REAL    NOT NULL,
    home_ml_prob        REAL    NOT NULL,
    away_ml_prob        REAL    NOT NULL,
    market_spread       REAL,
    home_spread_prob    REAL,
    away_spread_prob    REAL,
    market_total        REAL,
    over_prob           REAL,
    under_prob          REAL,
    model_version       TEXT    NOT NULL,
    model_confidence    REAL,
    actual_margin       REAL,
    actual_total        REAL
);

CREATE UNIQUE INDEX IF NOT EXISTS uix_model_predictions_matchup
    ON model_predictions (home_team, away_team, game_date, model_version);

CREATE INDEX IF NOT EXISTS idx_model_predictions_event_id
    ON model_predictions (event_id);

CREATE INDEX IF NOT EXISTS idx_model_predictions_game_date
    ON model_predictions (game_date);
