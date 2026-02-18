PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS slate_picks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    slate_id        INTEGER NOT NULL REFERENCES published_slates(id) ON DELETE CASCADE,
    rank            INTEGER NOT NULL,
    event           TEXT NOT NULL,
    api_event_id    TEXT,
    market          TEXT NOT NULL,
    selection       TEXT NOT NULL,
    line_value      REAL,
    best_odds       INTEGER NOT NULL,
    best_sportsbook TEXT NOT NULL,
    tier            TEXT NOT NULL,
    edge_ev_100     REAL NOT NULL,
    edge_z          REAL NOT NULL,
    consensus_prob  REAL NOT NULL,
    quality_score   INTEGER NOT NULL,
    quality_tier    TEXT NOT NULL,
    confidence      TEXT NOT NULL,
    kelly_suggested REAL,
    slate_score     REAL NOT NULL,
    avoid_reasons   TEXT,
    explanation     TEXT,
    commence_time   TEXT,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_slate_picks_slate ON slate_picks (slate_id);
CREATE INDEX IF NOT EXISTS idx_slate_picks_tier  ON slate_picks (tier);
CREATE INDEX IF NOT EXISTS idx_slate_picks_api_event ON slate_picks (api_event_id);
