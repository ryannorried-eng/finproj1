CREATE TABLE IF NOT EXISTS events (
    api_event_id   TEXT PRIMARY KEY,
    sport          TEXT NOT NULL,
    commence_time  TEXT NOT NULL,
    home_team      TEXT NOT NULL,
    away_team      TEXT NOT NULL,
    event_display  TEXT NOT NULL,
    first_seen_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_events_sport_time
ON events (sport, commence_time);
