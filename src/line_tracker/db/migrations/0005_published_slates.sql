PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS published_slates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    slate_date    TEXT NOT NULL,
    sport         TEXT NOT NULL,
    mode          TEXT NOT NULL,
    thresholds    TEXT NOT NULL,
    slate_hash    TEXT NOT NULL,
    engine_config TEXT NOT NULL,
    code_version  TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(slate_date, sport, mode, slate_hash)
);

CREATE INDEX IF NOT EXISTS idx_published_slates_date
ON published_slates (slate_date);

CREATE INDEX IF NOT EXISTS idx_published_slates_sport
ON published_slates (sport);
