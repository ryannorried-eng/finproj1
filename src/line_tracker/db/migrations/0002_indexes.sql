CREATE INDEX IF NOT EXISTS idx_bet_clv_bet_id
ON bet_clv (bet_id);

CREATE INDEX IF NOT EXISTS idx_event_type
ON lines (event, bet_type);

CREATE INDEX IF NOT EXISTS idx_timestamp
ON lines (timestamp);

CREATE INDEX IF NOT EXISTS idx_sportsbook
ON lines (sportsbook);

CREATE INDEX IF NOT EXISTS idx_event_type_book
ON lines (event, bet_type, sportsbook);

CREATE INDEX IF NOT EXISTS idx_bet_legs_bet_id
ON bet_legs (bet_id);

CREATE INDEX IF NOT EXISTS idx_bets_status
ON bets (status);
