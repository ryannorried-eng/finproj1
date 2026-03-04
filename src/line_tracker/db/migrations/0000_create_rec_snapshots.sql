-- Baseline: create rec_snapshots with the core schema matching migration 0008.
-- Later migrations (0009-0011) add columns via ALTER TABLE.
CREATE TABLE IF NOT EXISTS rec_snapshots (
    snapshot_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_id            TEXT NOT NULL,
    sport               TEXT,
    market              TEXT NOT NULL,
    selection           TEXT NOT NULL,
    line                REAL,
    book                TEXT NOT NULL,
    odds_american       REAL NOT NULL,
    odds_decimal        REAL NOT NULL,
    consensus_prob      REAL NOT NULL,
    breakeven_prob      REAL,
    edge_pct            REAL,
    edge_ev             REAL,
    edge_ev_shrunk      REAL,
    ev_100              REAL,
    edge_z              REAL,
    quality_score       INTEGER,
    confidence_label    TEXT,
    tier                TEXT,
    meta                TEXT,
    close_odds_american REAL,
    close_odds_decimal  REAL,
    close_implied_prob  REAL,
    open_implied_prob   REAL,
    clv_delta_american  REAL,
    clv_delta_implied   REAL,
    closed_at           TEXT,
    UNIQUE(event_id, market, selection, book, odds_american, created_at)
);

CREATE INDEX IF NOT EXISTS idx_rec_snap_created
ON rec_snapshots (created_at);

CREATE INDEX IF NOT EXISTS idx_rec_snap_event_market
ON rec_snapshots (event_id, market, selection, created_at);
