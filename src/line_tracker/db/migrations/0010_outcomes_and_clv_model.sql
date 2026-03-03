PRAGMA foreign_keys=ON;

-- Stores settled game outcomes imported from CSV.
CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id       TEXT NOT NULL,
    market         TEXT NOT NULL,
    selection      TEXT NOT NULL,
    line_value     REAL,
    result         TEXT NOT NULL CHECK(result IN ('win', 'loss', 'push')),
    settled_at     TEXT,
    created_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Partial unique indexes to correctly handle NULL line_value.
CREATE UNIQUE INDEX IF NOT EXISTS idx_outcomes_unique_with_line
ON outcomes (event_id, market, selection, line_value) WHERE line_value IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_outcomes_unique_no_line
ON outcomes (event_id, market, selection) WHERE line_value IS NULL;

CREATE INDEX IF NOT EXISTS idx_outcomes_event
ON outcomes (event_id, market, selection);

-- Extend rec_snapshots with outcome linkage columns.
ALTER TABLE rec_snapshots ADD COLUMN outcome_result TEXT;
ALTER TABLE rec_snapshots ADD COLUMN actual_roi REAL;

-- CLV model feature table: stores per-(market, selection_type) predicted
-- CLV-positive probability from historical rec_snapshots analysis.
CREATE TABLE IF NOT EXISTS clv_model_scores (
    score_id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at                  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    market                      TEXT NOT NULL,
    selection_type              TEXT,
    feature_json                TEXT NOT NULL,
    predicted_clv_positive_prob REAL NOT NULL,
    model_version               TEXT NOT NULL DEFAULT 'v1',
    UNIQUE(market, selection_type, model_version)
);

CREATE INDEX IF NOT EXISTS idx_clv_model_market
ON clv_model_scores (market, selection_type, model_version);
