# Codebase Handoff Brief

> **Target extensions:** near real-time ingestion, daily slate publishing, bankroll + portfolio risk sizing, Discord/subscription distribution.

---

## 1. Ingestion Execution

### How ingestion is initiated today

**Manual button click only.** The Streamlit dashboard (`src/line_tracker/dashboard.py:699`) renders a "Fetch Latest [Sport] Odds" button. When clicked, it calls the local helper `_fetch_odds()` (dashboard.py:619–684), which orchestrates:

```
User clicks button
  → dashboard._fetch_odds()
    → ingestion_service.fetch_and_persist_snapshot(store, api_key, sport=...)
      → OddsClient(api_key).get_odds(sport=...)          # HTTP GET to The Odds API
        → scraper._parse_events(response_json, sport)     # normalize JSON → BettingLine[]
      → store.save_lines(lines)                           # INSERT INTO lines
    → arbitrage.find_moneyline_arbs(lines)                # auto-compute arbs
    → arbitrage.find_spread_arbs(lines)                   # auto-compute spread arbs
    → movements.detect_moves(oldest, newest)              # diff oldest vs newest snapshot
  → results stored in st.session_state["last_fetch"], ["last_arbs"], ["last_moves"]
```

**There is no scheduled job, no cron, no background polling, and no CLI ingestion command.** The `__main__.py` exists but only launches the Streamlit app.

### Key functions / entrypoints

| Function | File | Purpose |
|----------|------|---------|
| `_fetch_odds()` | `dashboard.py:619` | UI orchestrator — calls service, computes arbs/moves, updates session_state |
| `fetch_and_persist_snapshot()` | `services/ingestion_service.py:12` | Service layer — fetch + persist + log timing |
| `OddsClient.get_odds()` | `scraper.py:46` | HTTP client — calls The Odds API v4 `/sports/{sport}/odds/` |
| `_parse_events()` | `scraper.py:85` | Parses raw JSON → `BettingLine[]`, dispatches to `_parse_moneyline`, `_parse_spread`, `_parse_total` |
| `LineStore.save_lines()` | `storage.py:100` | Batch INSERT into SQLite `lines` table |

### Polling / refresh behavior

**None.** Each fetch is a one-shot GET. No auto-refresh, no WebSocket, no SSE. Stale data stays in `st.session_state["last_fetch"]` until the user clicks the button again. There is no TTL or cache invalidation.

### Rate limit handling / backoff

**None.** `OddsClient` uses `httpx.Client(timeout=30)` with no retry, no exponential backoff, no 429 handling. If the API returns 401/403, it raises `ValueError`. If it returns 422, it raises `ValueError`. All other HTTP errors propagate as `httpx.HTTPStatusError` unhandled.

---

## 2. Data Identity & Normalization

### Event identity

Events are identified by a **synthetic string**: `"{away_team} @ {home_team}"` constructed in `scraper._parse_events()` (scraper.py:91). There is **no API event ID** stored. The Odds API returns an `id` field per event, but it is discarded during parsing. This means:

- If team names change (e.g., city relocation, API spelling changes), event identity breaks.
- Cross-sport or cross-source joins are impossible without the upstream ID.

### Market / line normalization

Markets are mapped 1:1 via `MARKET_TO_BET_TYPE` (scraper.py:14):

| API `market.key` | `BetType` enum | Field semantics |
|---|---|---|
| `"h2h"` | `MONEYLINE` | `home_value`/`away_value` = American odds |
| `"spreads"` | `SPREAD` | `home_value`/`away_value` = spread number; `home_price`/`away_price` = juice (American) |
| `"totals"` | `TOTAL` | `home_value`/`away_value` = O/U number; `home_price`/`away_price` = juice (American) |

**Props do not exist.** Any API market key not in `{"h2h", "spreads", "totals"}` is silently skipped (`scraper.py:101`: `if bet_type is None: continue`).

Draw outcomes (3-way moneylines, e.g., soccer) are **detected but not modeled**. `best_bets._is_three_way_market()` (best_bets.py:403) checks if the sport starts with `"soccer"` and emits a skipped `BetRecommendation` with `skipped_reason="3-way market"`.

### Deduplication

**There is no dedup.** Every call to `store.save_lines()` does a raw `INSERT INTO lines ...` (lines_repo.py:12–21). If you fetch the same sport twice within a minute, you get duplicate rows with identical data but different `id` and `created_at`. There is no UPSERT, no unique constraint on `(event, bet_type, sportsbook, timestamp)`.

---

## 3. Persistence Details

### Schema

**5 tables** defined in `db/migrations/0001_init.sql`:

```sql
-- lines: core odds snapshot table (HIGHEST GROWTH)
CREATE TABLE lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sportsbook TEXT NOT NULL,
    sport TEXT NOT NULL,
    event TEXT NOT NULL,         -- "{away} @ {home}" string
    bet_type TEXT NOT NULL,      -- "moneyline" | "spread" | "total"
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_value REAL NOT NULL,
    away_value REAL NOT NULL,
    home_price REAL,
    away_price REAL,
    timestamp TEXT NOT NULL,     -- ISO 8601 from API bookmaker.last_update
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    commence_time TEXT
);

-- bet_clv: closing line value snapshots
CREATE TABLE bet_clv (
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

-- calibration_thresholds: JSON blobs for auto-derived tier thresholds
CREATE TABLE calibration_thresholds (
    key TEXT PRIMARY KEY,         -- e.g., "global"
    json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- bets: placed bets
CREATE TABLE bets (
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
    -- Migration 0003 adds: source_page, recommendation_id, rank_at_pick,
    --   quality_tier_at_pick, edge_pct_at_pick, consensus_prob_at_pick,
    --   execution_delta_decimal
);

-- bet_legs: individual legs of parlays/straights
CREATE TABLE bet_legs (
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
```

### Indexes actually created (`db/migrations/0002_indexes.sql`)

```sql
CREATE INDEX idx_bet_clv_bet_id     ON bet_clv (bet_id);
CREATE INDEX idx_event_type         ON lines (event, bet_type);
CREATE INDEX idx_timestamp          ON lines (timestamp);
CREATE INDEX idx_sportsbook         ON lines (sportsbook);
CREATE INDEX idx_event_type_book    ON lines (event, bet_type, sportsbook);
CREATE INDEX idx_bet_legs_bet_id    ON bet_legs (bet_id);
CREATE INDEX idx_bets_status        ON bets (status);
```

### Highest-growth table

**`lines`** — every fetch inserts `(num_events × num_books × 3 markets)` rows. A single NFL fetch produces ~350–500 rows. With near-real-time polling (every 60s), this grows to ~500K rows/day for one sport.

### Most frequent queries

| Query | File | Frequency |
|-------|------|-----------|
| `INSERT INTO lines ...` (batch) | `lines_repo.py:12` | Every fetch |
| `SELECT * FROM lines WHERE event=? AND bet_type=? ... GROUP BY sportsbook` | `lines_repo.py:45` | Detail page (history tab) |
| `SELECT * FROM lines WHERE 1=1 ... ORDER BY timestamp DESC LIMIT ?` | `lines_repo.py:30` | Filtered line queries |
| `SELECT * FROM bets WHERE status=? LIMIT ?` | `bets_repo.py` | Sidebar active/settled counts |
| `SELECT * FROM bet_clv ... WHERE closed_at IS NOT NULL` | `clv_repo.py` | Performance page |

### WAL / busy_timeout settings

Configured in `storage.py:64-68` (`LineStore._configure_connection()`):

```python
self._conn.execute("PRAGMA foreign_keys = ON")
self._conn.execute("PRAGMA journal_mode = WAL")
self._conn.execute("PRAGMA busy_timeout = 5000")  # 5 seconds
```

Single-writer with `BEGIN IMMEDIATE` transactions (storage.py:78). No connection pooling.

---

## 4. Compute Pipeline: Best Bets (End-to-End)

### Pipeline flow

```
raw BettingLine[]  (from API or session_state)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│ recommend_best_bets(lines_for_event)                │  best_bets.py:1481
│   Split by BetType → ML / Spread / Total            │
│   Guard: skip 3-way markets (soccer ML)             │
└─────┬───────────────┬──────────────────┬────────────┘
      │               │                  │
      ▼               ▼                  ▼
  _moneyline_     _spread_           _total_
  recommendations recommendations   recommendations
  (py:1025)       (py:1170)         (py:1328)
      │               │                  │
      │  Each does:                      │
      │  1. _remove_vig() → de-vig each book's odds → per-book prob
      │  2. _filter_outliers() → keep/drop indices, market_unstable flag
      │  3. _consensus_prob() → trimmed_mean (≥5 books) or median (<5)
      │  4. consensus_prob_excluding_book() → leave-one-out for best-book
      │  5. _line_weight() = BOOK_WEIGHTS[book] × recency_multiplier
      │  6. _best_line_group() for spread/total (most-common spread number)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│ _build_rec()                                        │  best_bets.py:707
│   Computes ALL metrics for one side:                │
│   • ev_roi = p × d - 1                             │
│   • edge_pct = 100 × (p - 1/d)                     │
│   • weighted_robust_consensus() → p_cons_w          │
│   • compute_ev_edge(p_cons_w, d) → edge_ev          │
│   • compute_shrinkage(edge_ev, n_eff) → edge_ev_shrunk │
│   • compute_edge_z(edge_ev_shrunk, ev_sigma)        │
│   • _quality_score() = 0.45×edge + 0.25×agreement   │
│       + 0.20×coverage + 0.10×freshness              │
│   • _quality_tier() → Elite/Strong/Moderate/Thin    │
│   • kelly_suggested(p, d, confidence)               │
│   • Confidence: High (Z≥2.5), Medium (Z≥1.5), Low  │
│   • Attaches BestBetResult to rec                   │
└─────┬───────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│ assign_tiers(recs, method="hybrid")                 │  tiering.py:309
│   Methods: absolute / percentile / hybrid / composite│
│   hybrid (default):                                  │
│     Tier 1: Z >= max(1.5, p90 of slate Z)           │
│     Tier 2: Z >= p75                                │
│     Tier 3: Z >= p50                                │
│     Stay Away: below p50 or edge ≤ 0                │
│   Sets rec.bet_tier on each BetRecommendation       │
└─────┬───────────────────────────────────────────────┘
      │
      ▼
  Sort by EV desc → return top_n BetRecommendation[]
```

### Daily slate wrapper

```
lines_by_event: dict[str, list[BettingLine]]  (from session_state, grouped)
      │
      ▼
build_daily_slate_service()          services/slate_service.py:8
  → (Auto mode) load calibration from SQLite
  → build_daily_slate()              slate.py:686
      For each event:
        → recommend_best_bets()      (pipeline above)
        → assign_tiers()             (hybrid/absolute/percentile/composite)
        → classify_rec()             slate.py:287  ← detailed tier cascade
            Hard gates → Stay Away (unstable, <4 books, stale, hold≥8%)
            Tier 1A (Institutional): High conf, Elite/Strong, ≥6 books, ≤6% hold, edge ≥ dyn_floor
            Tier 1B (Standard): High/Med conf, ≥5 books, ≤7.5% hold, edge ≥ dyn_floor_1b
            Tier 2: positive edge, ≥4 books
            Tier 3: positive edge, anything else
        → _avoid_score()             (composite badness score for Stay Away ranking)
        → compute_distance_to_1b()   (how close Tier 2/3 entries are to Tier 1B)
      → bucket, filter, sort, cap Stay Away at 15
      → return {tier1a, tier1b, tier1, tier2, tier3, stay_away, counts, debug_stats}
```

### Where to insert extensions

**(a) Bankroll sizing** — Insert after `kelly_suggested` in `_build_rec()` (best_bets.py:820). Currently Kelly fraction is computed but not applied to any bankroll. You need:
- A `bankroll` parameter threaded from `st.session_state["bankroll"]` through `recommend_best_bets()` → `_build_rec()`
- A new field on `BetRecommendation`: `suggested_stake = bankroll × kelly_suggested`
- Alternate sizing modes: fractional Kelly (¼, ½), flat unit, risk-parity

**(b) Exposure caps** — Insert in `build_daily_slate()` (slate.py:686) after the per-event loop completes but before bucketing. At this point all entries exist with Kelly sizing. Add:
- Per-event max exposure (sum of suggested_stakes for all recs on same event)
- Per-sport max exposure
- Per-market-type max exposure
- Total portfolio exposure cap (% of bankroll)
- A new function `apply_exposure_caps(all_entries, bankroll, caps_config)` in a new `src/line_tracker/risk.py`

**(c) Correlation penalties** — Insert inside `_build_rec()` (best_bets.py:707) or as a post-processing step in `build_daily_slate()`. Correlated legs (e.g., same-game ML + spread, or two sides of same total) should have Kelly fractions reduced. This requires:
- Building a correlation matrix from `(event, market, side)` tuples
- Applying a penalty multiplier to `kelly_suggested` for correlated positions
- A new function `compute_correlation_penalty(entry, existing_positions)` in `risk.py`

---

## 5. Streamlit Interaction Model

### Pages that trigger compute and/or ingestion

| Page | Triggers Ingestion? | Triggers Compute? | Compute Functions |
|------|--------------------|--------------------|-------------------|
| **Dashboard** | YES — "Fetch Latest Odds" button → `_fetch_odds()` | YES — arbs + moves computed on every fetch | `find_moneyline_arbs()`, `find_spread_arbs()`, `detect_moves()` |
| **Game Detail** | No | YES — on render | `recommend_best_bets()`, `analyze_market()` (dashboard.py:881, 1068) |
| **Best Lines** | No | YES — on render | `compute_standouts()` (from `_lines_to_shopping_entries()`) |
| **Daily Slate** | No (reads calibration from SQLite) | YES — on render | `build_daily_slate_service()` → full pipeline |
| **Performance** | No | YES — on render | `load_clv_df()`, `summary_kpis()`, `all_breakdowns()`, `rolling_clv_series()` |

### What depends on `st.session_state` vs SQLite

| Data | Source | Lifetime |
|------|--------|----------|
| Current odds snapshot (`last_fetch`) | `session_state` | Lost on page refresh / rerun |
| Arb opportunities (`last_arbs`) | `session_state` | Lost on page refresh |
| Line movements (`last_moves`) | `session_state` | Lost on page refresh |
| Bet slip (`bet_slip`) | `session_state` | Lost on page refresh |
| Historical snapshots (History tab) | SQLite `lines` | Permanent |
| Placed bets | SQLite `bets` + `bet_legs` | Permanent |
| CLV data | SQLite `bet_clv` | Permanent |
| Calibration thresholds | SQLite `calibration_thresholds` | Permanent |
| User settings (bankroll, sport, API key) | `session_state` | Lost on page refresh |

### Known state bugs / mismatches

1. **Data loss on refresh.** All compute results (`last_fetch`, `last_arbs`, `last_moves`) and the entire bet slip live only in `session_state`. A browser refresh or Streamlit rerun wipes them. The user must re-fetch.

2. **Widget key corruption.** dashboard.py:1904–1906, 1785–1787, 2104–2106 explicitly delete invalid widget keys from `session_state` to prevent Streamlit crashes. If a widget value becomes a wrong type, it silently resets to default.

3. **No bet slip persistence.** The bet slip (`session_state["bet_slip"]`) is not saved to SQLite until "Mock Submit." If the browser crashes mid-session, all pending legs are lost.

4. **Recommendation metadata one-shot.** `_slip_rec_meta` is popped from `session_state` on submit (dashboard.py:1833). If the dialog reopens without re-adding the leg, the metadata link to the original recommendation is lost.

5. **Stale compute after fetch.** Best Lines and Daily Slate pages re-compute on every render from `session_state["last_fetch"]` but do not know when the data was fetched. There is no staleness indicator.

---

## 6. Current Gaps vs Goals

### Gap 1: No automated ingestion loop

**Impact:** Blocks near-real-time.
**Current state:** Manual button click only (`dashboard.py:699`).
**Proposed change:** Create `src/line_tracker/ingestion/scheduler.py` with an `asyncio` loop (or APScheduler) that calls `fetch_and_persist_snapshot()` on a configurable interval (default 60s). Add a CLI entrypoint: `python -m line_tracker ingest --sport=nfl --interval=60`. Decouple ingestion from the Streamlit process entirely.

### Gap 2: No event ID from upstream API

**Impact:** Blocks reliable dedup, cross-source joins, and event lifecycle tracking.
**Current state:** Events identified by `"{away} @ {home}"` string (scraper.py:91). The Odds API `id` field is discarded.
**Proposed change:** In `scraper._parse_events()`, capture `event["id"]` and add it to `BettingLine` as `api_event_id: str`. Add column `api_event_id TEXT` to `lines` table. Add a unique index on `(api_event_id, bet_type, sportsbook, timestamp)` for dedup.

### Gap 3: No snapshot deduplication

**Impact:** `lines` table grows unbounded with duplicate data on repeated fetches.
**Current state:** Raw INSERT every time (lines_repo.py:12). No UPSERT, no unique constraint.
**Proposed change:** Add `INSERT OR IGNORE` with a unique index on `(api_event_id, bet_type, sportsbook, timestamp)` in `lines_repo.py:12`. Or use `INSERT ... ON CONFLICT DO UPDATE SET` to update values if the timestamp matches.

### Gap 4: No rate limiting or retry logic

**Impact:** Near-real-time polling will hit API rate limits, leading to silent failures or bans.
**Current state:** No retry, no backoff, no 429 handling in `scraper.py`.
**Proposed change:** Add `tenacity` retry decorator (already a transitive dependency via Streamlit) to `OddsClient.get_odds()` in `scraper.py:46`: `@retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5), retry=retry_if_exception_type(httpx.HTTPStatusError))`. Track remaining API quota from response headers (`x-requests-remaining`).

### Gap 5: No bankroll sizing or risk management

**Impact:** Kelly fractions are computed but never applied to any bankroll amount.
**Current state:** `kelly_suggested` is calculated in `_build_rec()` (best_bets.py:821) but only displayed as a percentage. `st.session_state["bankroll"]` exists in the sidebar but is unused by the compute pipeline.
**Proposed change:** Create `src/line_tracker/risk.py` with: `compute_position_size(bankroll, kelly_frac, mode="quarter_kelly")`, `apply_exposure_caps(positions, bankroll, caps)`, `correlation_penalty(positions)`. Thread `bankroll` from session_state through `build_daily_slate_service()`.

### Gap 6: No portfolio exposure tracking

**Impact:** Can't enforce per-event, per-sport, or total portfolio limits.
**Current state:** Each bet is sized independently. No concept of "portfolio" or "existing positions."
**Proposed change:** Add a `positions` table to SQLite tracking active exposure by event/sport/market. Add `src/line_tracker/risk.py::PortfolioManager` class that queries active positions and applies caps before final sizing in `build_daily_slate()`.

### Gap 7: No publishing / distribution layer

**Impact:** Blocks Discord bot and subscription distribution.
**Current state:** All output is rendered in Streamlit only. No export, no API, no webhook.
**Proposed change:** Create `src/line_tracker/publishing/` package with: `formatter.py` (slate → Markdown/embed), `discord_bot.py` (discord.py webhook or bot), `api.py` (FastAPI endpoint for subscription tier filtering). The Daily Slate's `build_daily_slate()` return dict already has the right structure — just needs serialization.

### Gap 8: SQLite won't scale for concurrent writes

**Impact:** Near-real-time ingestion + Streamlit reads + potential API server = write contention.
**Current state:** Single `sqlite3.connect()`, WAL mode, `busy_timeout=5000`. No connection pooling.
**Proposed change (short-term):** Add connection pooling via a thread-local pattern in `storage.py`. Move to `aiosqlite` for async ingestion. **(Medium-term):** Migrate to PostgreSQL. The repo pattern (`db/repos/`) already abstracts queries — swap the connection backend.

### Gap 9: No event lifecycle / game state tracking

**Impact:** Can't publish a "daily slate" on a schedule without knowing which games are today, which have started, which are final.
**Current state:** `commence_time` is stored but never used for filtering or lifecycle management. There is no concept of "today's games" vs. "past games."
**Proposed change:** Add `src/line_tracker/events.py` with: `get_todays_events(lines)`, `is_live(event)`, `is_final(event)`. Use `commence_time` to filter. Add a `game_status` column to `lines` or a separate `events` table that tracks status (pre-game / live / final / postponed).

### Gap 10: No closing line capture for CLV

**Impact:** CLV analysis requires manual settlement. Can't auto-close at game time.
**Current state:** CLV closing snapshot is written manually via `store.close_clv()` (storage.py:200) when a user settles a bet in the UI. There is no automated closing-line capture at game start.
**Proposed change:** In the ingestion scheduler (Gap 1), add a `close_approaching_games()` job that runs 5 minutes before `commence_time` for each event. This fetches one final snapshot, computes consensus closing probability, and calls `store.close_clv()` for all active bets on that event. Implement in `src/line_tracker/services/clv_service.py`.

---

## Appendix: File Map

```
src/line_tracker/
├── __init__.py
├── __main__.py                    # Entry: launches Streamlit
├── dashboard.py                   # 3,284-line Streamlit app (ALL UI)
├── models.py                      # BettingLine, BestBetResult, BetType
├── scraper.py                     # OddsClient (The Odds API v4 HTTP)
├── storage.py                     # LineStore (SQLite facade, WAL, transactions)
├── best_bets.py                   # Consensus EV pipeline, BetRecommendation
├── slate.py                       # Daily slate builder, classify_rec, tier thresholds
├── tiering.py                     # Pluggable tiering (absolute/percentile/hybrid/composite)
├── arbitrage.py                   # Arb detection (moneyline + spread)
├── movements.py                   # Line movement detection (diff snapshots)
├── market_structure.py            # Market efficiency analysis, sharp-retail divergence
├── calibration.py                 # Auto-derive tier thresholds from CLV history
├── performance.py                 # CLV analytics (breakdowns, rolling, distribution)
├── bet_slip.py                    # Bet slip math (odds conversion, payouts)
├── bet_history.py                 # Bet history helpers
├── tracker.py                     # (Legacy/unused tracker)
├── alerts.py                      # Alert definitions (unused in current UI)
├── core/
│   ├── math.py                    # Odds math, Kelly, parlay, recommendation IDs
│   ├── clv.py                     # CLV calculation helpers
│   └── logging.py                 # Structured logger
├── db/
│   ├── migrate.py                 # Schema migration runner
│   ├── migrations/
│   │   ├── 0001_init.sql          # Tables: lines, bet_clv, calibration_thresholds, bets, bet_legs
│   │   ├── 0002_indexes.sql       # 7 indexes
│   │   └── 0003_recommendation_metadata.sql  # ALTER TABLE bets ADD columns
│   └── repos/
│       ├── lines_repo.py          # lines table CRUD
│       ├── bets_repo.py           # bets + bet_legs CRUD
│       ├── clv_repo.py            # bet_clv CRUD
│       └── calibration_repo.py    # calibration_thresholds CRUD
├── services/
│   ├── ingestion_service.py       # fetch_and_persist_snapshot()
│   ├── slate_service.py           # build_daily_slate_service() (thin wrapper)
│   ├── bet_service.py             # Bet placement orchestration
│   ├── performance_service.py     # Performance page data loading
│   └── explanation_service.py     # BestBetResult extraction for debug UI
└── ui/
    └── components/
        ├── explainability.py      # "Why this pick?" debug component
        └── diagnostics.py         # DB status/counts sidebar component
```
