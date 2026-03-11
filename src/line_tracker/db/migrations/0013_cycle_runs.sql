CREATE TABLE IF NOT EXISTS cycle_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at              TEXT    NOT NULL,
    finished_at             TEXT,
    sport                   TEXT    NOT NULL,
    success                 INTEGER NOT NULL DEFAULT 0,
    lines_fetched           INTEGER NOT NULL DEFAULT 0,
    events_processed        INTEGER NOT NULL DEFAULT 0,
    picks_generated         INTEGER NOT NULL DEFAULT 0,
    snapshots_written       INTEGER NOT NULL DEFAULT 0,
    clv_updates_attempted   INTEGER NOT NULL DEFAULT 0,
    clv_updates_completed   INTEGER NOT NULL DEFAULT 0,
    warnings_json           TEXT,
    errors_json             TEXT
);

CREATE INDEX IF NOT EXISTS idx_cycle_runs_started
    ON cycle_runs (started_at DESC);

CREATE INDEX IF NOT EXISTS idx_cycle_runs_sport
    ON cycle_runs (sport, started_at DESC);
