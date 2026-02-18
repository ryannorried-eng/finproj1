# Line Tracker — Technical Brief

## 1. Project Overview

**Line Tracker** is a sports-betting analytics platform that ingests live odds
from multiple sportsbooks, computes consensus probabilities via robust
statistical methods, surfaces tiered bet recommendations with full
explainability, tracks closing-line value (CLV), and presents everything
through an interactive Streamlit dashboard.

| Attribute | Value |
|-----------|-------|
| Language | Python 3.10+ |
| Package layout | `src/line_tracker` (PEP 621 src-layout) |
| Web UI | Streamlit |
| Persistence | SQLite (WAL mode) |
| HTTP client | httpx |
| Data analysis | pandas, matplotlib |
| Test framework | pytest (35 test modules) |
| Linter | ruff |

---

## 2. Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐
│  The Odds API │────▶│  OddsClient  │────▶│  IngestionService        │
│  (external)   │     │  scraper.py  │     │  ingestion_service.py    │
└──────────────┘     └──────────────┘     └────────────┬─────────────┘
                                                       │ store lines
                                                       ▼
                                          ┌──────────────────────────┐
                                          │  LineStore / Repos       │
                                          │  storage.py, db/repos/*  │
                                          │  SQLite (WAL, FK, busy)  │
                                          └────────────┬─────────────┘
                                                       │ read lines
                      ┌────────────────────────────────┼────────────────────┐
                      ▼                                ▼                    ▼
          ┌──────────────────┐          ┌──────────────────┐    ┌──────────────────┐
          │  best_bets.py    │          │  arbitrage.py    │    │  movements.py    │
          │  Consensus/Edge  │          │  Arb detection   │    │  Line moves      │
          └───────┬──────────┘          └──────────────────┘    └──────────────────┘
                  │ recommendations
                  ▼
          ┌──────────────────┐     ┌──────────────────┐
          │  tiering.py      │────▶│  slate.py        │
          │  Tier assignment │     │  Daily aggregator │
          └──────────────────┘     └───────┬──────────┘
                                           │
                                           ▼
          ┌──────────────────┐     ┌──────────────────┐
          │  performance.py  │◀────│  dashboard.py    │
          │  CLV analytics   │     │  Streamlit app   │
          └──────────────────┘     └──────────────────┘
```

### Layer summary

| Layer | Modules | Responsibility |
|-------|---------|----------------|
| **Ingestion** | `scraper.py`, `ingestion_service.py` | Fetch odds from The Odds API, normalize to `BettingLine`, persist |
| **Persistence** | `storage.py`, `db/repos/*`, `db/migrate.py` | SQLite CRUD, schema versioning, WAL journaling |
| **Analysis** | `best_bets.py`, `market_structure.py`, `arbitrage.py`, `movements.py` | Consensus probs, edge scoring, arb detection, line moves |
| **Recommendation** | `tiering.py`, `slate.py`, `calibration.py` | Tier assignment, daily slate, CLV-driven threshold tuning |
| **Services** | `services/*` | Orchestration layer bridging analysis ↔ persistence ↔ UI |
| **Presentation** | `dashboard.py`, `ui/components/*` | Streamlit pages, explainability rendering |
| **Shared math** | `core/math.py`, `core/clv.py` | Odds conversion, Kelly criterion, EV, CLV metrics |

---

## 3. Data Models

### 3.1 `BettingLine` (models.py)

Core domain object representing a single sportsbook price snapshot.

| Field | Type | Description |
|-------|------|-------------|
| `sportsbook` | `str` | Source book name |
| `sport` | `str` | Sport key (e.g. `americanfootball_nfl`) |
| `event` | `str` | Match-up description |
| `bet_type` | `BetType` | MONEYLINE, SPREAD, or TOTAL |
| `home_team` / `away_team` | `str` | Team names |
| `home_value` / `away_value` | `float` | Spread or total value (0 for ML) |
| `home_price` / `away_price` | `int` | American odds |
| `timestamp` | `datetime` | When the odds were captured |
| `commence_time` | `datetime` | Scheduled event start |

### 3.2 `BestBetResult` (models.py)

Simplified recommendation output with an explainability payload.

| Field | Type | Description |
|-------|------|-------------|
| `event` | `str` | Match-up |
| `bet_type` | `BetType` | Market type |
| `side` | `str` | Recommended selection |
| `sportsbook` | `str` | Best-price book |
| `odds` | `int` | Best American odds |
| `edge` | `float` | EV/$100 |
| `confidence` | `str` | High / Medium / Low |
| `quality_tier` | `str` | Elite / Strong / Moderate / Thin |
| `explanation` | `dict` | Full diagnostic payload for UI |

### 3.3 Database schema (4 migrations)

| Table | Purpose |
|-------|---------|
| `lines` | Historical line snapshots |
| `bets` | Placed bets with settlement status |
| `bet_legs` | Individual legs (parlay support) |
| `bet_clv` | Pick-time vs close-time metrics per leg |
| `calibration` | JSON-serialized tier threshold configs |
| `schema_version` | Migration tracking |

---

## 4. Core Algorithms

### 4.1 Consensus probability

1. **Vig removal** — For each book, convert American odds to implied
   probability and divide by the overround to get a fair (no-vig) probability.
2. **Book weighting** — Sharp books (Pinnacle, Circa, Bookmaker) receive
   1.5–3.0× weight; retail books receive 0.8–1.0×.
3. **Recency multiplier** — `exp(−age_minutes / 60)` gives a 60-minute
   half-life, upweighting fresher lines.
4. **Weight capping** — No single book may exceed 40 % of the total weight.
5. **Robust aggregation** — Trimmed weighted mean (20 % trim) or weighted
   median, depending on sample size.

### 4.2 Edge & expected value

| Metric | Formula |
|--------|---------|
| Breakeven prob | `1 / decimal_odds` |
| Edge (prob-point) | `consensus_prob − breakeven_prob` |
| Edge EV | `consensus_prob × decimal_odds − 1` |
| EV/$100 | `100 × edge_ev` |
| Shrunk EV | `edge_ev × n_eff / (n_eff + 5)` — Bayesian shrinkage for sample size |
| Edge Z-score | `shrunk_edge / max(volatility_sigma × odds, floor)` |

### 4.3 Quality scoring (0–100)

Four sub-scores are combined:

| Component | Measures |
|-----------|----------|
| Edge score | Magnitude of the expected-value advantage |
| Agreement score | Inverse of cross-book volatility sigma |
| Coverage score | Number of books offering the market |
| Freshness score | Age of the most recent price |

Overall quality tier: **Elite** (>90), **Strong** (70–90), **Moderate** (50–70),
**Thin** (<50).

### 4.4 Kelly criterion

```
base_kelly = (p × d − 1) / (d − 1)      # capped at 25 %
adjusted_kelly = base_kelly × conf_mult  # High=1.0, Med=0.5, Low=0.25
```

### 4.5 Tier assignment

Four pluggable strategies (`tiering.py`):

| Method | Logic |
|--------|-------|
| **Absolute** | Fixed thresholds on quality tier + confidence + edge |
| **Percentile** | Rank by edge-Z within the day's slate |
| **Hybrid** | `max(absolute floor, slate percentile)` |
| **Composite** | EV-weighted composite score |

Hard "stay-away" gates: hold ≥ 8 %, < 4 books, stale data, or statistical
outlier.

### 4.6 Arbitrage detection

Moneyline: compare best home odds across all books against best away odds;
if combined implied probability < 100 %, an arb exists.
Spread: detect line-value gaps between books on opposite sides.

### 4.7 CLV (Closing Line Value)

| Metric | Formula |
|--------|---------|
| CLV decimal | `pick_decimal − close_decimal` |
| CLV prob | `close_prob − pick_prob` |

Positive values mean the bettor captured better-than-closing odds — the
gold-standard measure of long-term profitability.

---

## 5. External API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v4/sports/` | GET | List available sports |
| `/v4/sports/{sport}/odds/` | GET | Live odds for a sport |

Provider: **The Odds API** (`https://the-odds-api.com`).
Supported sports: NFL, NBA, MLB, NHL, NCAAF, NCAAB, MMA, Soccer (EPL).
Markets: h2h (moneyline), spreads, totals.  Region: US.  Format: American.

---

## 6. Persistence

SQLite with defensive settings:

- **WAL journaling** — concurrent readers without blocking writers.
- **Foreign keys** enforced.
- **Busy timeout** — retries on lock contention.
- **Savepoint-based transactions** — nested transaction support.
- **Schema versioning** via `db/migrate.py` applying numbered SQL migrations.

Repository pattern (`db/repos/`): `LinesRepo`, `BetsRepo`, `ClvRepo`,
`CalibrationRepo` encapsulate all SQL and expose typed Python methods.

---

## 7. Dashboard (Streamlit)

`dashboard.py` (114 KB) provides seven pages:

| Page | Function |
|------|----------|
| **Slate** | Ranked daily recommendations across all events |
| **Shopping** | Line shopping for a single event |
| **Bet History** | Placed bets, settlement tracking |
| **Performance** | CLV KPIs, rolling trends, distribution charts |
| **Arbitrage** | Real-time arb scanning |
| **Debug** | Recommendation explanations, market structure |
| **Pro** | Calibration grid-search, tier analysis |

State management uses Streamlit session state for placed bets, recalculation
triggers, and filters.

---

## 8. Test Suite

35 test modules under `tests/` covering:

| Area | Key files |
|------|-----------|
| Betting math | `test_ev_math.py`, `test_kelly.py`, `test_odds_conversion.py` |
| Core engine | `test_best_bets.py` (83 KB), `test_consensus.py` |
| Slate | `test_slate.py` (59 KB), `test_slate_ranking.py` |
| CLV & performance | `test_clv.py`, `test_clv_analytics.py` |
| Tiering | `test_tiering.py`, `test_calibration.py` |
| Persistence | `test_bet_persistence.py`, `test_bet_service.py` |
| Market analysis | `test_market_structure.py`, `test_arbitrage.py` |
| Integration | `test_pro_mode.py` (29 KB), `test_phase3_best_bet_result.py` (30 KB) |
| UI smoke | `test_streamlit_smoke.py` |

All tests run via `pytest -q` from the repo root.

---

## 9. Development Workflow

```bash
# Install editable
python -m pip install -e .

# Run dashboard
streamlit run src/line_tracker/dashboard.py

# Run tests
pytest -q

# Lint
ruff check src/ tests/
```

---

## 10. Design Patterns & Conventions

| Pattern | Where |
|---------|-------|
| **Repository** | `db/repos/*` — data-access abstraction over SQLite |
| **Service layer** | `services/*` — orchestration between repos, analysis, and UI |
| **Context manager** | `LineStore`, `OddsClient` — resource lifecycle |
| **Pluggable strategy** | `tiering.py` — interchangeable tier-assignment methods |
| **Explainability payload** | `BestBetResult.explanation` dict — every recommendation carries its full diagnostic breakdown |
| **Defensive persistence** | WAL + FK + busy timeout + savepoints |
| **Src-layout** | Standard PEP 621 project structure for clean imports |
