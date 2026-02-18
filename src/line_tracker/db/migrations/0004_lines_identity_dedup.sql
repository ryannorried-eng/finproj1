PRAGMA foreign_keys=ON;

ALTER TABLE lines ADD COLUMN api_event_id TEXT;
ALTER TABLE lines ADD COLUMN ingested_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP);

CREATE INDEX IF NOT EXISTS idx_lines_api_event_id
ON lines (api_event_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_lines_snapshot
ON lines (api_event_id, bet_type, sportsbook, timestamp);

CREATE INDEX IF NOT EXISTS idx_lines_event_type_book_id
ON lines (event, bet_type, sportsbook, id);

CREATE INDEX IF NOT EXISTS idx_lines_event
ON lines (event);

CREATE INDEX IF NOT EXISTS idx_lines_event_ts
ON lines (event, timestamp DESC);

DROP INDEX IF EXISTS idx_event_type_book;
