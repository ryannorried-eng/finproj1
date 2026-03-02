# Project Tech Brief

## Repo Inventory

```
finproj1/
├── pyproject.toml                       # Project config, dependencies, entry points
├── src/line_tracker/
│   ├── __init__.py                      # Package init, __version__ = "0.1.0"
│   ├── __main__.py                      # CLI entry point (fetch/lines/arbs/moves/events/sports)
│   ├── models.py                        # BetType enum, BettingLine, BestBetResult dataclasses
│   ├── config.py                        # Centralized config (Streamlit secrets → env fallback)
│   ├── scraper.py                       # OddsClient – The Odds API v4 client (httpx)
│   ├── storage.py                       # LineStore – SQLite persistence facade
│   ├── tracker.py                       # LineTracker – in-memory line tracking
│   ├── best_bets.py                     # Core recommendation engine (consensus, EV, quality)
│   ├── slate.py                         # Daily slate builder + tier classification cascade
│   ├── scoring.py                       # Alpha/hybrid scoring, ranking, Kelly effective
│   ├── alpha.py                         # Alpha v1.1 – rule-based robustness overlay (0–100)
│   ├── tiering.py                       # 5 pluggable tiering methods (absolute/percentile/hybrid/composite/quantile)
│   ├── calibration.py                   # Tier threshold auto-calibration via grid search + rolling CLV
│   ├── performance.py                   # CLV analytics, breakdowns, rolling series, calibration stats
│   ├── bet_history.py                   # Bet model, session state, DB persistence, CLV tracking
│   ├── bet_slip.py                      # Odds conversion, EV, parlay math (mirrors core/math.py)
│   ├── market_structure.py              # Sharp/retail divergence, hold spread, market tagging
│   ├── movements.py                     # Line movement detection
│   ├── arbitrage.py                     # ML and spread arb scanning
│   ├── alerts.py                        # Alert system (arbs, movements)
│   ├── dashboard.py                     # [NOT PRESENT] – Streamlit UI (referenced but not in repo)
│   ├── core/
│   │   ├── math.py                      # Shared betting math: odds conversion, Kelly, parlay, recommendation ID
│   │   ├── clv.py                       # Canonical CLV formulas (clv_decimal, clv_prob)
│   │   └── logging.py                   # Structured logging with run_id/source context
│   ├── db/
│   │   ├── migrate.py                   # Schema migration runner (sequential SQL files)
│   │   ├── repos/                       # Repository classes (one per table group)
│   │   │   ├── lines_repo.py            # LinesRepo
│   │   │   ├── bets_repo.py             # BetsRepo
│   │   │   ├── clv_repo.py              # ClvRepo
│   │   │   ├── events_repo.py           # EventsRepo
│   │   │   ├── slates_repo.py           # SlatesRepo
│   │   │   ├── slate_picks_repo.py      # SlatePicksRepo
│   │   │   ├── rec_snapshots_repo.py    # RecSnapshotsRepo
│   │   │   └── calibration_repo.py      # CalibrationRepo
│   │   └── migrations/
│   │       ├── 0001_init.sql            # lines, bet_clv, calibration_thresholds, bets, bet_legs
│   │       ├── 0002_indexes.sql         # Performance indexes
│   │       ├── 0003_recommendation_metadata.sql  # Rec linkage columns on bets
│   │       ├── 0004_lines_identity_dedup.sql     # Unique constraint on lines
│   │       ├── 0005_published_slates.sql          # published_slates table
│   │       ├── 0006_slate_picks.sql               # slate_picks table
│   │       ├── 0007_events.sql                    # events lookup table
│   │       ├── 0008_rec_snapshots.sql             # rec_snapshots table (CLV logging)
│   │       └── 0009_alpha_columns.sql             # alpha_score/alpha_label on rec_snapshots
│   └── services/
│       ├── ingestion_service.py         # fetch_and_persist_snapshot()
│       ├── slate_service.py             # Slate orchestration
│       ├── publish_service.py           # Slate publishing
│       ├── bet_service.py               # Bet placement orchestration
│       ├── performance_service.py       # Performance data loading
│       ├── explanation_service.py       # Human-readable explanations
│       └── rec_snapshot_service.py       # Rec snapshot logging + closing-line capture
├── tests/
│   ├── conftest.py                      # Path setup for deterministic imports
│   ├── test_best_bets.py (2189 lines)   # Consensus, EV, quality scoring tests
│   ├── test_slate.py (2048 lines)        # Tier classification, slate building
│   ├── test_scoring.py (848 lines)       # Alpha, hybrid, ranking, Kelly effective
│   ├── test_tiering.py (797 lines)       # All 5 tiering methods
│   └── ... (49 test files, 2187 tests total)
```

## Architecture & Components

**Runtime:** Python 3.11 · **Framework:** Streamlit (dashboard) · **Data source:** [The Odds API](https://the-odds-api.com) v4 · **HTTP client:** httpx · **DB:** SQLite (WAL mode) · **ORM:** None (raw SQL via repository pattern)

**Architecture pattern:** Layered service architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Dashboard                  │  ← UI layer (dashboard.py, not in repo)
├─────────────────────────────────────────────────────┤
│                   Services Layer                     │  ← ingestion, slate, bet, publish, perf
├─────────────┬──────────┬──────────┬─────────────────┤
│ best_bets   │  slate   │ scoring  │  tiering        │  ← Business logic
│ alpha       │  calibr  │ perf     │  market_struct   │
├─────────────┴──────────┴──────────┴─────────────────┤
│  core/math   core/clv   core/logging                │  ← Shared math/utilities
├─────────────────────────────────────────────────────┤
│          storage.py → db/repos/* → SQLite            │  ← Persistence
├─────────────────────────────────────────────────────┤
│          scraper.py (OddsClient → The Odds API)      │  ← Data ingestion
└─────────────────────────────────────────────────────┘
```

**Entrypoints:**
- **CLI:** `python -m line_tracker` → `src/line_tracker/__main__.py:main()`
- **Dashboard:** `streamlit run dashboard.py` (file not in repo, but config references it)
- **Services:** Called from dashboard or CLI, no standalone HTTP server

## How To Run (Golden Path)

### Install
```bash
pip install -e ".[dev]"        # Installs line-tracker + dev deps (pytest, ruff)
```

### Local dev (dashboard)
```bash
export ODDS_API_KEY=<your-key>
streamlit run dashboard.py     # Dashboard file not in repo – needs to be created/restored
```

### CLI usage
```bash
python -m line_tracker fetch --sport nba        # Fetch + persist odds
python -m line_tracker lines --event "Team @ Team" --type moneyline
python -m line_tracker arbs --sport nfl         # Scan arbitrage
python -m line_tracker moves --event "..."      # Detect line movements
python -m line_tracker events                   # List tracked events
python -m line_tracker sports                   # List API-available sports
```

### Database setup
No manual migration needed. `LineStore.__init__()` auto-runs all migrations in `db/migrations/` via `ensure_latest()` on first connection. Default DB path: `~/.line_tracker/lines.db`.

### Running tests
```bash
pytest tests/                                   # 2187 tests, ~3s
ruff check src/ tests/                          # Linting
```

### No schedulers/pipelines found
There is no cron, Celery, APScheduler, or similar automation in the repo. Ingestion is manual (CLI `fetch` or dashboard button).

## Config & Environment Variables

All config is read via `config.py:_read_secret()` — checks Streamlit secrets first, then `os.environ`.

| Variable | Purpose | Default | File |
|---|---|---|---|
| `ODDS_API_KEY` | The Odds API key (required) | — | `config.py:get_api_key()` |
| `DB_PATH` | Custom SQLite path | `~/.line_tracker/lines.db` | `config.py:get_db_path()` |
| `DATABASE_URL` | Persistent DB URL | None | `config.py:get_database_url()` |
| `DISPLAY_TZ` | Timezone for UI | `America/Chicago` | `config.py:get_display_timezone()` |
| `TIER_METHOD` | Tiering method | `quantile` | `config.py:get_tier_method()` |
| `TIER1_Q` | Tier 1 quantile share | `0.10` | `config.py:get_tier1_quantile()` |
| `TIER2_Q` | Tier 2 quantile share | `0.35` | `config.py:get_tier2_quantile()` |
| `TIER_MIN_CANDIDATES` | Min candidates for quantile tiers | `10` | `config.py:get_tier_min_candidates()` |
| `MARKET_W_SPREAD` | Spread market weight | `1.00` | `config.py` |
| `MARKET_W_TOTAL` | Total market weight | `0.95` | `config.py` |
| `MARKET_W_ML_FAV` | ML favorite weight | `0.90` | `config.py` |
| `MARKET_W_ML_DOG` | ML underdog weight | `0.75` | `config.py` |
| `MARKET_W_LONGSHOT` | Longshot penalty mult | `0.90` | `config.py` |
| `KELLY_MULT_HIGH` | Kelly mult for High confidence | `1.00` | `config.py` |
| `KELLY_MULT_MED` | Kelly mult for Medium | `0.70` | `config.py` |
| `KELLY_MULT_LOW` | Kelly mult for Low | `0.40` | `config.py` |
| `PRO_HYBRID_MARKET_CONF` | Enable conf-weighted market weight | `0` (off) | `config.py` |
| `CONF_W_HIGH/MED/LOW/MIN/MAX` | Confidence weight factors | 1.15/1.00/0.85/0.50/1.10 | `config.py` |
| `LINE_TRACKER_DEBUG` | Enable debug output in slate | `false` | `slate.py` |
| `DEBUG_RANKING` | Print top-5 ranking debug | unset | `scoring.py` |

## Data Model & Storage

### Engine
**SQLite** with WAL journal mode, foreign keys ON, 5000ms busy timeout. No ORM — raw SQL via repository pattern (`db/repos/*.py`). Migrations in `db/migrations/0001-0009.sql`, applied sequentially by `db/migrate.py:ensure_latest()`.

### Key Tables

**`lines`** — Historical odds snapshots
- `id`, `sportsbook`, `sport`, `event`, `bet_type`, `home_team`, `away_team`, `home_value`, `away_value`, `home_price`, `away_price`, `timestamp`, `commence_time`, `api_event_id`
- Unique constraint on `(sportsbook, event, bet_type, home_value, away_value, home_price, away_price, timestamp)` (migration 0004)

**`events`** — Stable event lookup
- `api_event_id` (PK), `sport`, `commence_time`, `home_team`, `away_team`, `event_display`

**`bets`** — Placed bets
- `bet_id` (PK), `sportsbook`, `stake`, `total_odds_american/decimal`, `potential_payout`, `profit`, `status` (active/won/lost/push), `settled_at`, `outcome`
- Recommendation metadata: `source_page`, `recommendation_id`, `rank_at_pick`, `quality_tier_at_pick`, `edge_pct_at_pick`, `consensus_prob_at_pick`, `execution_delta_decimal` (migration 0003)

**`bet_legs`** — Individual legs of a bet
- `leg_id` (PK), `bet_id` (FK→bets), `sport`, `market`, `event_name`, `selection`, `line_value`, `odds_american/decimal`, `sportsbook`, `pick_timestamp`, `commence_time`

**`bet_clv`** — Closing Line Value tracking per leg
- `bet_id`, `leg_index`, `event`, `market`, `pick_side`, pick-time snapshot (odds, consensus_prob, hold, sigma, confidence, quality_tier, edge_pct, edge_z, books_used, agreement_score), close-time snapshot (consensus_prob_close, best_odds_close_american/decimal), `closed_at`

**`rec_snapshots`** — Recommendation snapshots for CLV logging (migration 0008-0009)
- `snapshot_id`, `created_at`, `event_id`, `sport`, `market`, `selection`, `line`, `book`, `odds_american/decimal`, `consensus_prob`, `breakeven_prob`, `edge_pct`, `edge_ev`, `edge_ev_shrunk`, `ev_100`, `edge_z`, `quality_score`, `confidence_label`, `tier`, `alpha_score`, `alpha_label`, closing fields (`close_odds_american/decimal`, `close_implied_prob`, `open_implied_prob`, `clv_delta_american`, `clv_delta_implied`, `closed_at`)

**`published_slates`** / **`slate_picks`** — Published slate audit trail (migrations 0005-0006)

**`calibration_thresholds`** — Persisted calibration JSON by scope key

### Data Ingestion
- **Source:** The Odds API v4 (`scraper.py:OddsClient.get_odds()`)
- **Sports supported:** NFL, NBA, MLB, NHL, NCAAF, NCAAB, MMA, Soccer EPL (`__main__.py:SPORTS`)
- **Markets:** h2h (moneyline), spreads, totals
- **Flow:** `OddsClient.get_odds()` → `_parse_events()` → list[`BettingLine`] → `LineStore.save_lines()` (deduped upsert)

## Core Business Logic (EV/CLV/Unlocks/Parlays)

### What "Picks" Are
A **pick** (internally: `BetRecommendation` in `best_bets.py`) is a recommended wager on one side of a market. Lifecycle: raw odds → vig removal → consensus probability → EV computation → quality scoring → tier classification → slate display.

Key `BetRecommendation` fields: `market`, `selection`, `side`, `line`, `consensus_prob`, `best_odds`, `ev_roi`, `ev_100`, `edge_pct`, `edge_ev`, `edge_ev_shrunk`, `edge_z`, `confidence`, `quality_score`, `quality_tier`, `kelly_base`, `kelly_suggested`, `bet_tier`, `alpha_score`, `alpha_label`.

### Odds/Books/Markets Representation
- **BetType enum** (`models.py`): `MONEYLINE`, `SPREAD`, `TOTAL`
- **BettingLine** dataclass: `sportsbook`, `sport`, `event`, `bet_type`, `home_team`, `away_team`, `home_value`, `away_value`, `home_price`, `away_price`, `timestamp`
  - ML: `home_value`/`away_value` = American odds
  - Spread: `home_value`/`away_value` = spread number, `home_price`/`away_price` = juice
  - Total: `home_value` = over number, `away_value` = under number, prices = juice

### Book Weights (`best_bets.py:BOOK_WEIGHTS`)
Sharp/low-hold books get higher influence on consensus:
- Pinnacle: 3.0, Circa: 2.5, Bookmaker: 2.0, BetOnline/SuperBook: 1.5
- BetMGM: 1.2, DraftKings/FanDuel/Caesars: 1.0
- PointsBet/WynnBET/Unibet/BetRivers: 0.8

Weight capping: no single book > 40% of total weight (`_MAX_BOOK_WEIGHT_SHARE = 0.40`, `best_bets.py:_cap_weights()`).

### Vig Removal Formula (`best_bets.py:_remove_vig()`)
```
pA_raw = implied_prob(odds_A)
pB_raw = implied_prob(odds_B)
pA_no_vig = pA_raw / (pA_raw + pB_raw)
pB_no_vig = pB_raw / (pA_raw + pB_raw)
```
Where `implied_prob(odds)`:
- Negative odds: `|odds| / (|odds| + 100)`
- Positive odds: `100 / (odds + 100)`

### Consensus Probability Computation

**Step 1: Per-book de-vigged probabilities** — for each sportsbook, remove vig to get true probability estimate.

**Step 2: Outlier filtering** (`best_bets.py:_filter_outliers()`)
- Remove books where `|p - median| / median > 0.15` (relative threshold)
- Remove books with `p < 0.01` or `p > 0.99`
- If > 30% filtered or < 4 remain → `market_unstable = True`

**Step 3: Consensus** (`best_bets.py:_consensus_prob()`)
- >= 5 books: trimmed mean (trim=0.2 — drop 20% from each tail)
- < 5 books: plain median

**Step 4: Weighted robust consensus** (`best_bets.py:weighted_robust_consensus()`)
```
w_i = (1 / (eps + |hold_i|)) * (1 / (eps + |p_i - median(p)|))
consensus = Σ(w_i * p_i) / Σ(w_i)
```
Combines inverse hold (sharp preference) with outlier resistance.

**Step 5: Exclusion consensus** — for each candidate book, consensus is recomputed *excluding* that book to avoid self-referencing bias (`consensus_prob_excluding_book()`).

### EV Calculation (`best_bets.py`)

```
decimal_odds = american_to_decimal(best_odds)
    if odds >= 0: dec = odds/100 + 1
    if odds < 0:  dec = 100/|odds| + 1

# Breakeven probability
p_be = 1 / decimal_odds

# Probability-point edge
edge_pp = consensus_prob - p_be

# Edge percentage (prob points * 100)
edge_pct = 100 * edge_pp

# EV per $1 staked (ROI)
ev_roi = consensus_prob * decimal_odds - 1

# EV per $100 wagered
ev_100 = 100 * ev_roi

# EV-based edge
edge_ev = consensus_prob * decimal_odds - 1    (same as ev_roi)
```

### Shrinkage & Edge Z-Score (`best_bets.py`)

```
# Effective sample size
n_eff = books_used * (1 - outlier_rate)

# Bayesian shrinkage toward zero (k = 5.0)
edge_ev_shrunk = edge_ev * n_eff / (n_eff + 5.0)

# EV sigma (robust, IQR-based)
robust_sigma = IQR / 1.349   (in probability space)
ev_sigma = robust_sigma * decimal_odds

# Edge z-score
edge_z = edge_ev_shrunk / max(ev_sigma, 0.002)
```

### Confidence Label (`best_bets.py:_confidence_label()`)
Based on IQR of de-vigged probabilities across books:
- IQR <= 0.03 → **High** (books agree within 3 pp)
- IQR <= 0.06 → **Medium**
- IQR > 0.06 → **Low**

### Quality Score (0–100) (`best_bets.py`)
Composite of four subscores:

| Subscore | Weight | Source |
|---|---|---|
| `edge_score` | Piecewise linear on edge_pct (0→0, 0.5→35, 1→55, 2→75, 3→85, 5→95, 7→100) | `_edge_score()` |
| `agreement_score` | IQR-based (90 if IQR<=0.03, 70 if <=0.06, else 40) with penalties for high stdev, few books, staleness | `_agreement_score()` |
| `coverage_score` | `books_used/books_total * 100`, capped at 70 if total < 5 | `_coverage_score()` |
| `freshness_score` | Age-based (95 if <=10min, 75 if <=30min, 50 if <=60min, 30 if <=120min, else 10) | `_freshness_score()` |

Final: `quality_score = 0.30 * edge + 0.30 * agreement + 0.20 * coverage + 0.20 * freshness`

Quality tiers: **Elite** (>=85), **Strong** (>=70), **Moderate** (>=55), **Thin** (<55)

### Kelly Criterion (`core/math.py:kelly_fraction()`)
```
kelly_fraction = (p * decimal_odds - 1) / (decimal_odds - 1)
kelly_fraction = clamp(kelly_fraction, 0, 0.25)   # 25% cap

# Confidence-scaled Kelly
kelly_suggested = kelly_fraction * confidence_multiplier
    High: 1.0, Medium: 0.5, Low: 0.25
```

### Kelly Effective (`scoring.py:compute_kelly_effective()`)
Based on `confidence_label` (quality/prob/hold/alpha), NOT edge-z confidence:
```
kelly_effective = kelly_base * confidence_label_multiplier
    High: 1.00, Medium: 0.70, Low: 0.40    (configurable via env)
kelly_effective = clamp(kelly_effective, 0, 0.25)
```

### Alpha Score (0–100) (`alpha.py:alpha_score()`)
Rule-based robustness overlay using smoothstep curves:

| Component | Range | Formula |
|---|---|---|
| **A) Market robustness** | 0–30 | `18 * smoothstep(4, 10, books) + 12 * (1 - smoothstep(4.5, 8, hold))` |
| **B) Edge robustness** | 0–40 | `20 * smoothstep(0.75, 2.25, edge_z) + 20 * smoothstep(0.50, 3.00, shrunk_pct)` |
| **C) Agreement/stability** | 0–20 | `12 * smoothstep(70, 95, agreement) + 8 * (1 - smoothstep(0.006, 0.016, sigma))` |
| **D) Consensus longshot penalty** | 0 to −18 | `-18 * (1 - smoothstep(0.18, 0.30, consensus_prob))` |
| **E) Implied longshot penalty** | 0 to −12 | Stepped: 0 if p>=0.25, −2 if >=0.20, −5 if >=0.15, −9 if >=0.10, −12 else |
| **F) ML penalty** | 0 to −7 | −4 base for ML markets, −7 if implied_prob < 0.20 |

`smoothstep(lo, hi, x) = t²(3 - 2t)` where `t = clamp((x - lo)/(hi - lo), 0, 1)`

Labels: **Strong** (>=72), **Neutral** (>=45), **Weak** (<45)

Alpha gating: when `ALPHA_GATE_ENABLED = True`, tier1b recs with Weak alpha → downgraded to tier2 (`slate.py`).

### Hybrid Score (`scoring.py:compute_hybrid_fields()`)
```
alpha_norm  = alpha_score / 100
kelly_norm  = clamp(kelly_suggested / 0.05, 0, 1)
prob_norm   = consensus_prob

hybrid_score_raw = 0.40 * alpha_norm + 0.40 * kelly_norm + 0.20 * prob_norm
hybrid_score = hybrid_score_raw * market_weight(entry)
```

Market weights (soft bias): spreads=1.00, totals=0.95, ML favorite=0.90, ML underdog=0.75. Longshot penalty (ML + prob < 0.20): ×0.90.

### Confidence Label (Gating) (`scoring.py:compute_confidence_label()`)
Separate from edge-z confidence — used for Tier 1 eligibility:
- **High:** quality >= 70, prob >= 0.45, hold <= 9%, AND (alpha Strong OR edge_z >= 0.8)
- **Medium:** quality >= 60, prob >= 0.35, hold <= 10%
- **Low:** everything else

### CLV Computation (`core/clv.py:compute_clv_metrics()`)
```
clv_decimal = pick_decimal_odds - close_decimal_odds    (positive = got better odds)
clv_prob    = close_consensus_prob - pick_consensus_prob (positive = market moved toward your pick)
```
When `close_prob` is None/NaN, falls back to `pick_prob` (no movement assumed).

### CLV Tracking Flow (`bet_history.py`)
1. **At pick time** (`snapshot_pick()`): for each leg, look up latest lines, run consensus, store pick-time odds + consensus_prob + metadata in `bet_clv` table.
2. **At close** (`close_bet_clv()`): re-fetch latest lines, recompute consensus, store closing odds/prob.
3. **Analytics** (`performance.py`): build DataFrame, compute `clv_decimal`/`clv_prob`, rolling series, group-by breakdowns, distribution histograms.

### Rec-Snapshot CLV (`rec_snapshot_service.py`)
Logs every displayed recommendation (tier1-3 + closest_candidates) with pick-time metrics. Closing capture (`capture_closing_lines()`) finds latest odds for same event/market/book and computes CLV delta.

### Parlay Logic (`core/math.py:parlay_payout()`)
```
combined_decimal = product(american_to_decimal(odds_i) for each leg)
total_return = stake * combined_decimal
profit = total_return - stake
```
Parlay creation exists in `bet_history.py:create_bet()` — supports multi-leg bets. No parlay *recommendation* or *generation* logic found (only payout math).

## API Surface (Routes/Endpoints)

**There is no HTTP API/REST server in this repo.** The application is a Streamlit dashboard + CLI tool.

### CLI Commands (`__main__.py`)

| Command | Description | Key Args |
|---|---|---|
| `fetch` | Fetch live odds from API, save to DB | `--sport` (nfl/nba/mlb/nhl/ncaaf/ncaab/mma/soccer), `--db` |
| `lines` | Query stored lines | `--event`, `--type`, `--limit`, `--db` |
| `arbs` | Scan for arbitrage opportunities | `--sport`, `--db` |
| `moves` | Detect line movements between snapshots | `--event`, `--threshold`, `--db` |
| `events` | List tracked events | `--db` |
| `sports` | List available sports from API | — |

### Service Layer (called by dashboard)

| Service | Function | Purpose |
|---|---|---|
| `ingestion_service.py` | `fetch_and_persist_snapshot(store, api_key, sport)` | Fetch + save odds |
| `slate_service.py` | — | Slate orchestration (builds daily slate) |
| `bet_service.py` | — | Bet placement orchestration |
| `publish_service.py` | — | Slate publishing to DB |
| `performance_service.py` | — | Load CLV/performance data |
| `explanation_service.py` | — | Human-readable bet explanations |
| `rec_snapshot_service.py` | `build_snapshot_rows()`, `capture_closing_lines()` | CLV snapshot logging + closing |

## Pipeline & Automation

### No Automated Pipeline
There is no cron job, scheduler, or background worker in the repo. The full pipeline is triggered manually:

### Manual Pipeline Flow
```
1. Fetch odds        → OddsClient.get_odds(sport)           [scraper.py]
2. Parse + persist   → LineStore.save_lines(lines)           [storage.py]
3. Group by event    → lines_by_event dict                   [slate.py]
4. Best bets         → recommend_best_bets(lines)            [best_bets.py]
   a. De-vig each book pair → _remove_vig()
   b. Filter outliers → _filter_outliers()
   c. Compute consensus prob → _consensus_prob() + weighted_robust_consensus()
   d. Exclusion consensus per book → consensus_prob_excluding_book()
   e. Compute EV, edge, shrinkage, edge_z
   f. Quality scoring (edge + agreement + coverage + freshness)
   g. Kelly sizing
   h. Build BetRecommendation objects
5. Tiering           → assign_tiers(recs, method)            [tiering.py]
6. Classification    → classify_rec(entry)                   [slate.py]
   a. Hard disqualifiers → Stay Away
   b. Hold gate → Stay Away (>=8%)
   c. Negative edge → Stay Away
   d. Tier 1 (Core Value): edge_ev_shrunk > 0, edge_z >= 1.75,
      quality_score >= 70, consensus_prob >= 0.30, books >= 5,
      hold <= 7.5%, edge >= dynamic_floor
   e. Tier 2 (High Variance): edge_ev_shrunk > 0, edge_z >= 1.75,
      quality_score >= 65, consensus_prob < 0.30, books >= 4, hold <= 7.5%
   f. Tier 3 (Moderate Edge): edge_ev_shrunk > 0,
      1.15 <= edge_z < 1.75, quality_score >= 60, books >= 4, ev_100 >= 0.50
   g. Tier 3 catch-all: positive edge, no hard flags
7. Alpha scoring     → alpha_score(entry)                    [alpha.py]
8. Hybrid ranking    → compute_hybrid_score(entry)           [scoring.py]
9. Slate assembly    → build_daily_slate()                   [slate.py]
10. Display          → Streamlit dashboard (not in repo)
11. CLV snapshot     → build_snapshot_rows() + log           [rec_snapshot_service.py]
12. Closing capture  → capture_closing_lines()               [rec_snapshot_service.py]
```

## Predictive Model

### No Predictive Model Exists
There is **no machine learning model** in this repo. The system is entirely rule-based.

**Alpha scoring** (`alpha.py`) uses smoothstep curves over market structure signals — it is *not* ML-based despite its name.

### Where a Model Would Plug In
1. **Consensus probability override** — replace or augment `_consensus_prob()` in `best_bets.py` with a model-predicted probability.
2. **Alpha scoring** — replace `alpha.py:alpha_score()` with a trained classifier. The smoothstep rule-based approach is explicitly described as "No ML, no external APIs" in its docstring.
3. **Tier classification** — `slate.py:classify_rec()` could consume model-predicted tier scores instead of fixed thresholds.
4. **Calibration** — `calibration.py:grid_search_thresholds()` is a brute-force grid search that could be replaced with Bayesian optimization or a learned threshold model.

### Placeholders/TODOs
- `ALPHA_GATE_ENABLED = False` — alpha gating is in shadow mode (scores computed but not used for gating).
- `calibration.py` already has the infrastructure for rolling CLV-based threshold adjustment — a model could drive this.
- The `tiering.py` quantile method already uses a weighted composite `tier_score()` — this function's weights could be learned.

## Testing

- **Framework:** pytest (9.0.2)
- **Total tests:** 2,187 across 49 test files
- **Linter:** ruff (0.15.4)
- **No CI/CD configuration found** (no `.github/workflows/`, no `Makefile`, no `tox.ini`)

### Test Coverage by Area
| Area | Test File(s) | Lines |
|---|---|---|
| Consensus/EV/Quality | `test_best_bets.py` | 2189 |
| Slate/Tiering | `test_slate.py`, `test_tiering.py` | 2848 |
| Scoring/Alpha | `test_scoring.py`, `test_alpha.py` | 848+ |
| CLV/Performance | `test_clv.py`, `test_performance.py` | — |
| Betting math | `test_validate_betting_math.py` (fuzz tests) | — |
| Storage/Migrations | `test_storage.py`, `test_migrations.py` | — |
| Services | `test_services_*.py` | — |
| Market structure | `test_market_structure.py` | — |

## Risks / Gaps

1. **No dashboard.py in repo** — the Streamlit UI is referenced everywhere but the file is missing. This means the product cannot be run as a user-facing application.
2. **No automated ingestion** — no scheduler, cron, or background job. Odds must be fetched manually. CLV closing capture is also manual.
3. **No CI/CD** — no GitHub Actions, no pre-commit hooks, no automated test runs.
4. **No predictive model** — pure rule-based system. Alpha gating is disabled (`ALPHA_GATE_ENABLED = False`).
5. **SQLite single-writer** — WAL mode helps, but production deployment with concurrent writes will hit contention.
6. **No parlay generation** — only parlay payout math exists. No automated parlay recommendation logic.
7. **No result tracking** — bets can be settled manually, but there's no automated result fetching (game scores, outcomes).
8. **API key in environment** — no secrets manager integration beyond Streamlit secrets.
9. **No rate limiting on API calls** — `OddsClient` has no retry logic or rate limiting for The Odds API quota management.
10. **Stale line detection is time-based only** — `oldest_update_age_min > 120` is the only staleness check; no comparison against game start time.

## Next Steps (Top 10)

### 1. Restore or rebuild `dashboard.py`
**Why:** The Streamlit UI is the primary user interface — without it the product is CLI-only.
**Files:** Create `src/line_tracker/dashboard.py` (or `dashboard.py` at root)
**Acceptance:** `streamlit run dashboard.py` renders slate, bet slip, performance pages; all service-layer functions exercised.

### 2. Add automated ingestion scheduler
**Why:** Manual `fetch` means stale data and missed closing-line windows.
**Files:** New `src/line_tracker/scheduler.py`; update `__main__.py` with `run` command
**Acceptance:** Background process fetches odds every N minutes for configured sports; closing-line capture runs automatically before game time.

### 3. Add CI/CD pipeline
**Why:** 2187 tests exist but aren't run automatically — regressions can ship silently.
**Files:** `.github/workflows/ci.yml`
**Acceptance:** Push to any branch runs `pytest` + `ruff check`; PRs blocked on failure.

### 4. Enable alpha gating and validate with CLV
**Why:** Alpha scoring exists but `ALPHA_GATE_ENABLED = False` — the system doesn't use its own robustness overlay.
**Files:** `alpha.py` (flip flag), `performance.py` (add alpha-gated CLV comparison)
**Acceptance:** A/B comparison of CLV metrics with and without alpha gating; measurable improvement in CLV beat-rate for Tier 1.

### 5. Add automated result fetching
**Why:** Bet settlement is manual. Need game results to compute win/loss automatically.
**Files:** New `src/line_tracker/results.py`; update `bet_history.py`
**Acceptance:** After game completion, bets auto-settle based on scores API; P&L dashboard updates without manual intervention.

### 6. Implement parlay recommendation engine
**Why:** Parlay payout math exists but no recommendation logic. Parlays are a key user feature.
**Files:** New `src/line_tracker/parlay_builder.py`; extend `slate.py`
**Acceptance:** System suggests 2-3 leg parlays from Tier 1/2 picks with positive expected value; correlated legs excluded.

### 7. Add API rate limiting and retry logic
**Why:** The Odds API has a monthly quota. No retry on network errors.
**Files:** `scraper.py:OddsClient`
**Acceptance:** Exponential backoff on 429/5xx; remaining quota tracked and surfaced; requests stop when quota exhausted.

### 8. Build a trained probability model
**Why:** Consensus probability is purely market-derived. A model trained on historical outcomes could improve edge detection.
**Files:** New `src/line_tracker/model/`; integrate into `best_bets.py:_consensus_prob()`
**Acceptance:** Model-predicted probabilities show higher CLV beat-rate than market consensus alone on held-out test set.

### 9. Migrate to PostgreSQL for production
**Why:** SQLite single-writer limits concurrent dashboard + scheduler access.
**Files:** `storage.py`, `db/repos/*.py` (adapt SQL dialect), `config.py` (use `DATABASE_URL`)
**Acceptance:** All tests pass against PostgreSQL; concurrent reads/writes work without `SQLITE_BUSY`.

### 10. Add observability and monitoring
**Why:** `core/logging.py` exists but there's no structured monitoring, no error tracking, no health checks.
**Files:** Extend `core/logging.py`; add `src/line_tracker/health.py`; integrate with external monitoring (e.g., Sentry)
**Acceptance:** Structured JSON logs; error rates tracked; alerts on ingestion failures or CLV degradation; health endpoint for uptime monitoring.
