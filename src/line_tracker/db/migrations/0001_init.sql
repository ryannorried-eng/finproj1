CREATE TABLE IF NOT EXISTS lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sportsbook TEXT NOT NULL,
    sport TEXT NOT NULL,
    event TEXT NOT NULL,
    bet_type TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_value REAL NOT NULL,
    away_value REAL NOT NULL,
    home_price REAL,
    away_price REAL,
    timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    commence_time TEXT
);

CREATE TABLE IF NOT EXISTS bet_clv (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bet_id TEXT NOT NULL,
    leg_index INTEGER NOT NULL DEFAULT 0,
    event TEXT NOT NULL,
    market TEXT NOT NULL,
    pick_side TEXT NOT NULL,
    pick_line_value REAL,
    pick_odds_american REAL NOT NULL,
    pick_odds_decimal REAL NOT NULL,
    consensus_prob_at_pick REAL NOT NULL,
    market_hold_median_at_pick REAL DEFAULT 0.0,
    market_volatility_sigma_at_pick REAL DEFAULT 0.0,
    consensus_prob_close REAL,
    best_odds_close_american REAL,
    best_odds_close_decimal REAL,
    closed_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    pick_sportsbook TEXT,
    sport TEXT,
    confidence_at_pick TEXT,
    quality_tier_at_pick TEXT,
    edge_pct_at_pick REAL,
    edge_z_at_pick REAL,
    books_used_at_pick INTEGER,
    agreement_score_at_pick REAL,
    UNIQUE(bet_id, leg_index)
);

CREATE TABLE IF NOT EXISTS calibration_thresholds (
    key TEXT PRIMARY KEY,
    json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bets (
    bet_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    sportsbook TEXT NOT NULL,
    stake REAL NOT NULL,
    total_odds_american INTEGER NOT NULL,
    total_odds_decimal REAL NOT NULL,
    potential_payout REAL NOT NULL,
    profit REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    settled_at TEXT,
    outcome TEXT
);

CREATE TABLE IF NOT EXISTS bet_legs (
    leg_id TEXT PRIMARY KEY,
    bet_id TEXT NOT NULL REFERENCES bets(bet_id),
    sport TEXT,
    market TEXT,
    event_name TEXT,
    selection TEXT,
    line_value REAL,
    odds_american INTEGER NOT NULL,
    odds_decimal REAL NOT NULL,
    sportsbook TEXT,
    pick_timestamp TEXT,
    commence_time TEXT
);
