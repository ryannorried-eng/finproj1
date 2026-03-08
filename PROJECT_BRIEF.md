# COMPLETE PROJECT BRIEF — Line Tracker

---

## 1. Executive Summary

Line Tracker is a **sports betting analytics engine** that ingests real-time odds from multiple US sportsbooks via The Odds API, computes vig-free consensus probabilities, identifies +EV (positive expected value) edges, and produces a tiered daily "slate" of ranked bet recommendations. It includes a Streamlit dashboard, a CLI, CLV (Closing Line Value) tracking, a pick pruning pipeline, bet tracking/settlement, and a self-calibrating tier system.

**Current state**: Functionally complete prototype / early MVP. All 2,387 tests pass. The core math is sound and well-tested. The architecture is coherent but accumulated incremental complexity — there are 5 different tiering methods, 3 ranking modes, multiple overlapping scoring systems (quality_score, alpha_score, hybrid_score, tier_score, composite_score), and a 3,730-line Streamlit dashboard that has become a monolith. The project works end-to-end but needs consolidation, simplification, and operational hardening.

**Top 3 risks**:
1. **Scoring system proliferation** — Too many overlapping scoring/ranking/tiering systems making it hard to know which one "is the answer."
2. **Dashboard monolith** — 3,730 lines of Streamlit in a single file; fragile, hard to maintain, hard to test.
3. **CLV closing-line capture uses heuristic matching** — The `_find_closing_odds` function uses fragile string matching (`selection.lower() in home_team.lower()`) which will silently produce wrong CLV data.

---

## 2. Project Identity

- **Project name**: Line Tracker (`line-tracker`)
- **One-sentence description**: A quantitative sports betting tool that scrapes odds from US sportsbooks, computes consensus probabilities with vig removal, and surfaces tiered +EV bet recommendations with CLV tracking.
- **Problem it solves**: Automates the process of comparing odds across books, computing true probabilities, finding value edges, and tracking whether those edges were real (via CLV).
- **Primary user**: A sports bettor (likely the developer) who wants systematic, data-driven bet selection rather than gut picks.
- **Stage**: Working prototype / early MVP. Core pipeline works. Tests pass. Dashboard runs. Not production-hardened.
- **Long-term vision**: A fully automated betting recommendation engine that learns from its own CLV history, self-calibrates thresholds, and produces daily actionable slates — possibly for multiple users.
- **Does implementation match vision?**: Partially. The core math and pipeline are solid. But the system is over-engineered in some areas (5 tiering methods, 3 ranking modes, 6+ scoring subsystems) and under-engineered in others (no scheduler, manual fetch, fragile CLV matching, monolithic dashboard).

**Core value proposition**: Turn raw odds data into probabilistically-grounded, edge-aware bet recommendations with confidence levels and suggested sizing.

**Blunt assessment**: The architecture is coherent (clear separation of scraper → best_bets → slate → tiering → pruning → ranking → snapshot) but has accumulated significant complexity from iterative development. Multiple "directions" (absolute tiering vs quantile tiering, quality_tier vs alpha_label vs confidence_label, hybrid_score vs tier_score) coexist without clear resolution. The system needs a decision about which scoring path is canonical.

---

## 3. Repo / Architecture Overview

### Top-Level Structure

```
finproj1/
├── src/line_tracker/           # Main package
│   ├── __main__.py             # CLI entry point
│   ├── config.py               # Centralized config (env/secrets)
│   ├── dashboard.py            # Streamlit dashboard (3,730 lines!)
│   ├── models.py               # Data models (BettingLine, BestBetResult, BetType)
│   ├── scraper.py              # The Odds API client
│   ├── storage.py              # SQLite persistence (LineStore)
│   ├── best_bets.py            # Core consensus/EV/edge computation engine
│   ├── slate.py                # Daily slate builder (tier classification)
│   ├── scoring.py              # Alpha, hybrid, market-weight, Kelly scoring
│   ├── alpha.py                # Alpha v1.1 robustness overlay (0-100)
│   ├── tiering.py              # 5 pluggable tiering methods
│   ├── calibration.py          # Self-calibrating tier thresholds from CLV
│   ├── bet_slip.py             # Odds conversion wrappers
│   ├── bet_history.py          # Bet state management
│   ├── movements.py            # Line movement detection
│   ├── arbitrage.py            # Arbitrage finder
│   ├── alerts.py               # Alert manager
│   ├── market_structure.py     # Market analysis (sharp/retail divergence)
│   ├── performance.py          # CLV performance analytics
│   ├── core/
│   │   ├── math.py             # Canonical math functions
│   │   ├── clv.py              # CLV metric computations
│   │   └── logging.py          # Structured logging
│   ├── db/
│   │   ├── migrate.py          # Migration runner
│   │   ├── migrations/         # 13 SQL migrations (0000-0012)
│   │   └── repos/              # Repository pattern (10 repos)
│   ├── services/
│   │   ├── automation_service.py   # Cycle orchestrator
│   │   ├── slate_service.py        # Slate builder wrapper
│   │   ├── rec_snapshot_service.py  # Snapshot logging + CLV capture
│   │   ├── pick_pruning_service.py  # Gate cascade pruning
│   │   ├── ranking_service.py      # Top-pick selection
│   │   ├── clv_selection_service.py # CLV-driven filter profiles
│   │   ├── clv_model_service.py    # CLV prediction model
│   │   ├── ingestion_service.py    # Fetch + persist
│   │   ├── bet_service.py          # Bet persistence
│   │   ├── outcome_service.py      # Settlement import
│   │   ├── performance_service.py  # Performance loading
│   │   ├── publish_service.py      # Slate publication
│   │   └── explanation_service.py  # Pick explainability
│   └── ui/
│       └── components/
│           ├── diagnostics.py   # Debug diagnostics UI
│           └── explainability.py # "Why this pick?" UI
├── tests/                      # 2,387 tests (all pass)
├── scripts/
│   ├── audit_math.py           # Math audit script
│   └── tiering_analysis.py     # Tiering analysis script
├── docs/                       # 3 markdown docs
├── pyproject.toml              # Python project config
└── requirements.txt            # Pinned dependencies
```

### Important Entry Points

| Entry Point | Location | Purpose |
|---|---|---|
| CLI | `src/line_tracker/__main__.py` | 12 subcommands (fetch, slate, cycle, etc.) |
| Dashboard | `src/line_tracker/dashboard.py` | Streamlit web app |
| Automation cycle | `services/automation_service.py:run_cycle()` | Full pipeline in one call |

### Central vs Peripheral Files

**Central (must-understand)**:
- `best_bets.py` — The engine. Computes consensus probabilities, EV, edges, quality scores, Kelly fractions. ~1,100 lines.
- `slate.py` — Orchestrates per-event analysis into a daily slate with tier1a/1b/2/3/stay-away classification.
- `scoring.py` — Alpha fields, hybrid scores, market weights, confidence labels, Kelly effective sizing.
- `storage.py` — All DB operations via repository delegation.
- `config.py` — All configurable thresholds (30+ parameters).

**Peripheral (secondary)**:
- `arbitrage.py`, `movements.py`, `alerts.py` — Useful features but not core to the recommendation pipeline.
- `bet_slip.py` — Pure wrapper around `core/math.py`. Could be eliminated.
- `market_structure.py` — Sharp/retail divergence analysis, used in stay-away logic.

### Code Quality Issues

1. **`bet_slip.py` is entirely wrapper functions** — every function just delegates to `core/math.py`. This is dead indirection.
2. **Kelly cap is defined in 3 places**: `core/math.py` (`_KELLY_CAP = 0.25`), `best_bets.py` (`_KELLY_CAP = 0.25`), `scoring.py` (`_KELLY_CAP = 0.05` — different value!). The `scoring.py` one is for normalization, not capping, but the naming is misleading.
3. **`confidence` has two different meanings**: In `best_bets.py` it's edge_z based (z >= 2.5 → High). In `scoring.py:compute_confidence_label()` it's a quality/prob/hold/alpha composite. Both get called "confidence" in different contexts, creating semantic confusion.
4. **Dashboard monolith**: `dashboard.py` is 3,730 lines handling 7+ pages in one file. No separation of concerns.

---

## 4. Current Workflow

### A. High-Level Workflow

```
[The Odds API] → Fetch → SQLite → Best Bets Analysis → Slate → Tier → Prune → Rank → Display/Snapshot
                                                                                          ↓
                                                                              CLV Closing Capture ← (later)
```

### B. Detailed Step-by-Step

1. **Fetch** (`scraper.py:OddsClient.get_odds()`): Calls The Odds API for a sport, returns `BettingLine` objects.
2. **Persist** (`storage.py:LineStore.save_lines()`): Upserts lines + events into SQLite.
3. **Best Bets Analysis** (`best_bets.py:recommend_best_bets()`): For each event:
   - Group lines by bet type (moneyline/spread/total)
   - For spread/total: find the best line group (most books on same number)
   - Remove vig via proportional normalization: `p_no_vig = p_implied / (p_home + p_away)`
   - Filter outliers (>15% relative deviation from median)
   - Compute consensus probability via trimmed mean (>=5 books) or median (<5)
   - Compute weighted robust consensus using inverse-hold × inverse-deviation weights
   - Find best available odds across books
   - Compute EV metrics: `ev_roi = p * d - 1`, `ev_100 = 100 * ev_roi`
   - Apply Bayesian shrinkage: `edge_ev_shrunk = edge_ev * n_eff / (n_eff + 5.0)`
   - Compute edge_z: `edge_ev_shrunk / max(ev_sigma, 0.02)`
   - Compute quality subscores (edge 45%, agreement 25%, coverage 20%, freshness 10%)
   - Compute Kelly fractions, confidence labels, sizing notes
   - Return `BetRecommendation` objects with `BestBetResult` attached
4. **Slate Building** (`slate.py:build_daily_slate()`):
   - Runs `recommend_best_bets()` for each event
   - Applies tiering (default: hybrid method using quantile for assignments, then cascade thresholds for 1a/1b split)
   - Enriches entries with alpha_score, hybrid_score, confidence_label, kelly_effective
   - Classifies into tier1a (institutional), tier1b (standard), tier2, tier3, stay_away
   - Applies stay-away filters (negative EV, high hold, high noise, stale data)
5. **Pruning** (`pick_pruning_service.py`): 8-gate cascade:
   - tier ∈ {tier1a, tier1b, tier2}
   - alpha_label ∈ {Strong, Neutral}
   - edge_z >= 1.75
   - edge_ev_shrunk > 0
   - quality_score >= 65
   - books_used >= 5
   - market_hold_median <= 7.0
   - CLV filter (if historical data exists)
6. **Ranking** (`scoring.py:rank_candidates()`): 3 modes:
   - `hybrid`: weighted(alpha_score, kelly, prob) × market_weight
   - `hit`: probability-first (for win-rate optimization)
   - `value`: EV-first (for profit optimization)
7. **Snapshot** (`rec_snapshot_service.py`): Logs all displayed candidates to `rec_snapshots` table.
8. **CLV Closing** (`rec_snapshot_service.py:capture_closing_lines()`): Finds unclosed snapshots, looks up latest odds, computes CLV delta.

### C. Typical User Workflow

1. Open Streamlit dashboard
2. Select sport (NBA, NFL, etc.)
3. Click "Fetch Live Odds" to pull from API
4. View "Daily Slate" page — see tier1a/1b/2/3 picks
5. Optionally add picks to bet slip
6. View "Performance" page for CLV tracking
7. Alternatively: run `python -m line_tracker cycle --sport nba` for automated pipeline

### D. Current Pain Points

1. **Manual fetching** — No scheduler. User must manually trigger fetches.
2. **CLV closing is fragile** — `_find_closing_odds` uses substring matching on team names.
3. **Too many scoring dimensions** — A user sees quality_score, alpha_score, hybrid_score, edge_z, confidence, confidence_label, quality_tier, tier... overwhelming.
4. **No outcome-tracking automation** — Outcomes must be manually imported via CSV.

---

## 5. Product / Business Logic

### Decision System

The system makes recommendations via this logic chain:

1. **Edge Detection**: Is the consensus probability higher than the breakeven probability? (`consensus_prob > 1/decimal_odds`)
2. **Edge Significance**: Is the edge statistically significant? (`edge_z >= threshold`)
3. **Market Quality**: Is the market deep enough to trust? (`books_used >= 4`, `quality_score >= threshold`)
4. **Robustness**: Does the alpha overlay confirm the edge is robust? (`alpha_label` ∈ {Strong, Neutral})
5. **Sizing**: How much to bet? (Kelly fraction × confidence multiplier)

### What "Good" Means

- **Tier 1a** (Institutional): 6+ books, hold <= 6%, high edge_z, Strong alpha → highest confidence picks
- **Tier 1b** (Standard): 5+ books, hold <= 7.5%, moderate edge_z → reliable picks
- **Tier 2**: Positive EV, reasonable quality → secondary picks
- **Tier 3**: Marginal edge, lower quality → speculative
- **Stay Away**: Negative EV, high hold, stale data, noisy markets

### Contradictions / Tensions

1. **EV logic vs hit-rate logic**: The system has both `value` mode (EV-first) and `hit` mode (prob-first). These optimize different things. The `hybrid` mode tries to blend them, but the user needs to decide what they actually want.

2. **Too many confidence concepts**:
   - `confidence` (in `BetRecommendation`): based on edge_z thresholds (z >= 2.5 = High, z >= 1.5 = Medium)
   - `confidence_label` (in `scoring.py`): based on quality_score + consensus_prob + hold + alpha
   - Both are called "confidence" in different places. They can disagree (a pick can be "High" confidence by edge_z but "Low" confidence_label by quality).

3. **Alpha gating is disabled**: `ALPHA_GATE_ENABLED = False` in `alpha.py`. Alpha scores are computed but never used for actual gating. The system computes them in shadow mode.

4. **5 tiering methods exist but only 1-2 are used**: The `tiering.py` module implements absolute, percentile, hybrid, composite, and quantile methods. In practice, `hybrid` is the default for `assign_tiers()` and `quantile` is used within `slate.py`. The others are dead code paths (except for tests).

---

## 6. Math / Quant Engine

### 6.1 Vig Removal (`best_bets.py:_remove_vig`)
- **Formula**: `p_no_vig = p_implied / (p_home + p_away)` where `p_implied = |odds| / (|odds| + 100)`
- **Assessment**: Correct. Standard proportional normalization. Assumes no draw (guarded by `_is_three_way_market` for soccer).

### 6.2 Consensus Probability (`best_bets.py:_consensus_prob`)
- **Formula**: Trimmed mean (trim 20% from each tail) when >= 5 books; plain median when < 5.
- **Assessment**: Sound. Trimmed mean is a reasonable robust estimator. Could be improved with weighted approaches (which exist separately as `weighted_robust_consensus`).

### 6.3 Weighted Robust Consensus (`best_bets.py:weighted_robust_consensus`)
- **Formula**: `w_i = 1/(eps + hold_i) * 1/(eps + |p_i - median|)` then weighted average.
- **Assessment**: Clever. Weights by sharpness (low hold) and agreement (close to median). The double-inverse weighting is defensible. Note: this is a separate signal from `_consensus_prob` — the system computes both.

### 6.4 Expected Value (`best_bets.py:_compute_ev`)
- **Formula**: `EV = p * (d - 1) - (1 - p)` where d = decimal odds
- **Assessment**: Correct. Standard EV formula for a binary bet.

### 6.5 Bayesian Shrinkage (`best_bets.py:compute_shrinkage`)
- **Formula**: `edge_ev_shrunk = edge_ev * n_eff / (n_eff + k)` where `k = 5.0`
- **Assessment**: Correct. Standard empirical Bayes shrinkage toward zero. The shrinkage constant k=5.0 means you need ~5 effective books for the edge to be half-strength. Reasonable for this domain.

### 6.6 Effective Sample Size
- **Formula**: `n_eff = books_used * (1 - outlier_rate)` where `outlier_rate = outliers_removed / total_books`
- **Assessment**: Simple but sensible. Penalizes markets where many books were filtered.

### 6.7 Edge Z-Score (`best_bets.py:compute_edge_z`)
- **Formula**: `edge_z = edge_ev_shrunk / max(ev_sigma, 0.02)`
- **Assessment**: Sound. Uses robust sigma (`IQR / 1.349`) scaled by decimal odds as the denominator. The floor of 0.02 (configurable via `EDGE_Z_SIGMA_MIN`) prevents division artifacts when volatility is very low.

### 6.8 Kelly Criterion (`core/math.py:kelly_fraction`)
- **Formula**: `f = (p * d - 1) / (d - 1)`, capped at 25%
- **Assessment**: Correct. Standard Kelly formula. The 25% cap is conservative (full Kelly is aggressive). The confidence multiplier (High=1.0, Medium=0.5, Low=0.25) further reduces sizing. This is effectively quarter-Kelly or less for most recommendations.

### 6.9 Alpha Score (`alpha.py:alpha_score`)
- **Formula**: 0-100 composite:
  - Market robustness (0-30): `18 * smoothstep(4, 10, books) + 12 * (1 - smoothstep(4.5, 8.0, hold))`
  - Edge robustness (0-40): `20 * smoothstep(0.75, 2.25, edge_z) + 20 * smoothstep(0.5, 3.0, shrunk_pct)`
  - Agreement/stability (0-20): `12 * smoothstep(70, 95, agreement) + 8 * (1 - smoothstep(0.006, 0.016, sigma))`
  - Consensus longshot penalty (0 to -18)
  - Implied longshot penalty (0 to -12)
  - ML market penalty (0 to -7)
- **Assessment**: Well-designed. Uses Hermite smoothstep for smooth transitions (avoids cliff effects). Weights are reasonable. Longshot penalties are appropriate. The total theoretical range is [-37, 90] before clamping, which gets clamped to [0, 100].

### 6.10 Hybrid Score (`scoring.py:compute_hybrid_fields`)
- **Formula**: `hybrid = (0.40 * alpha/100 + 0.40 * kelly/0.05 + 0.20 * prob) * market_weight`
- **Assessment**: Reasonable weighted composite. The Kelly normalization cap of 0.05 (5%) is lower than the actual Kelly cap (25%), which effectively clips any Kelly fraction above 5% to full score.

### 6.11 Tier Score (`tiering.py:tier_score`)
- **Formula**: `0.35 * clip(edge_z/3) + 0.25 * clip(ev_shrunk_pct/5) + 0.15 * clip(agreement/100) + 0.15 * clip(books/8) + 0.10 * clip(quality/100) - 0.15 * clip(hold/0.10)` with longshot penalty of -0.10.
- **Assessment**: Sound. Weights are defensible. The hold penalty acts as a quality gate. Could benefit from calibration against actual outcomes.

### 6.12 Quality Score (`best_bets.py:_quality_score`)
- **Formula**: `0.45 * edge_score + 0.25 * agreement_score + 0.20 * coverage_score + 0.10 * freshness_score`
- **Assessment**: Reasonable composite. Edge dominates (45%), which aligns with the product goal. Coverage and freshness are appropriate secondary signals.

### 6.13 CLV (Closing Line Value) (`rec_snapshot_service.py:compute_clv_for_snapshot`)
- **Formula**: `clv_implied = close_implied_prob - open_implied_prob` (positive = you beat the close)
- **Assessment**: Standard CLV formulation. Implementation is mathematically correct. The data-collection side (`_find_closing_odds`) is fragile (see Section 11).

### What Math is Missing

1. **No probability calibration against outcomes** — The system computes consensus probabilities but never validates them against actual results (win/loss/push) to see if they're well-calibrated. The `calibration.py` module calibrates *thresholds* (tier cutoffs) against CLV, not the probabilities themselves.
2. **No correlation handling** — When building multi-leg parlays, there's no adjustment for correlated legs.
3. **No bankroll tracking** — Kelly fractions are computed but there's no actual bankroll state to apply them to.
4. **No market-specific probability adjustments** — Spreads and totals use the same vig-removal formula, but in practice the vig structure differs by market type.

---

## 7. Data Model / Database

### Database Type
SQLite (WAL mode, foreign keys ON, busy timeout 5s). File at `~/.line_tracker/lines.db` by default.

### Schema (13 migrations: 0000-0012)

| Table | Purpose | Key Fields | Rows (est.) |
|---|---|---|---|
| `lines` | Historical odds snapshots | sportsbook, sport, event, bet_type, home_value, away_value, prices, timestamp, api_event_id | High volume |
| `events` | Canonical event registry | api_event_id (PK), sport, commence_time, home/away_team | Medium |
| `bets` | User-placed bets | bet_id (PK), sportsbook, stake, odds, status, outcome | Low |
| `bet_legs` | Individual legs of bets | leg_id (PK), bet_id (FK), market, event, selection, odds | Low |
| `bet_clv` | CLV tracking per bet leg | bet_id+leg_index (unique), pick/close odds and probs | Low |
| `rec_snapshots` | Recommendation snapshots per slate run | snapshot_id, event_id, market, selection, odds, consensus_prob, tier, CLV fields | High volume |
| `outcomes` | Settled game outcomes (CSV import) | event_id, market, selection, result (win/loss/push) | Medium |
| `clv_model_scores` | Per-market CLV prediction model | market, selection_type, predicted_clv_positive_prob | Low |
| `calibration_thresholds` | Saved calibration configs | key (PK), json | Low |
| `published_slates` | Published slate metadata | (from migration 0005) | Low |
| `slate_picks` | Individual picks in published slates | (from migration 0006) | Low |

### Relationships
- `bet_legs.bet_id` → `bets.bet_id` (FK)
- `rec_snapshots` linked to `outcomes` via join on `event_id + market + selection` (no FK)
- `rec_snapshots` linked to `events` via `event_id` (no FK — could be fragile)

### Data Integrity Concerns
1. **No FK from `rec_snapshots` to `events`** — snapshots reference events by api_event_id but there's no foreign key constraint.
2. **`lines` table has no api_event_id index** — added in migration 0004 as a dedup mechanism but querying lines by api_event_id may be slow without a dedicated index.
3. **`rec_snapshots` dedup relies on composite unique** — `(event_id, market, selection, book, odds_american, created_at)` which could miss near-simultaneous updates.

### Repository Pattern
Clean separation: 10 repo classes in `db/repos/` each handling one table/domain. `LineStore` composes all repos and manages transactions. Well-structured.

---

## 8. APIs / Commands / Interfaces

### CLI Commands (`__main__.py`)

| Command | Purpose | Input | Depends On |
|---|---|---|---|
| `fetch --sport nfl` | Fetch live odds, save to DB | ODDS_API_KEY | API access |
| `lines --event "..." --type spread` | Query stored lines | DB | Prior fetch |
| `arbs --sport nfl` | Scan for arbitrage | ODDS_API_KEY | Live API |
| `moves --event "..."` | Detect line movements | DB | 2+ fetches |
| `events` | List tracked events | DB | Prior fetch |
| `slate --sport nba --mode Standard` | Build daily slate | DB | Prior fetch |
| `snapshot --sport nba` | Log recommendation snapshots | DB | Prior fetch |
| `cycle --sport nba --mode Standard` | Full automation cycle | DB | Prior fetch |
| `import-outcomes <csv>` | Import game outcomes | CSV file | CSV format |
| `roi-report` | Show ROI report | DB | Outcomes imported |
| `kpi --last-hours 24` | Show KPI summary | DB | Cycle runs |
| `train-model` | Train CLV prediction model | DB | Closed snapshots |
| `sports` | List available sports | ODDS_API_KEY | API access |

### Correct Order of Operations

1. `fetch` (get data from API)
2. `slate` or `cycle` (analyze and produce recommendations)
3. Wait for games to start → `fetch` again → closing-line capture happens in `cycle`
4. After games finish → `import-outcomes` (manual CSV)
5. `roi-report` or `kpi` (analyze performance)
6. `train-model` (optional: refine CLV predictions)

### Streamlit Dashboard Pages (from `dashboard.py`)

Based on the imports and the 3,730-line file, the dashboard includes:
- **Live Odds** — Fetch and display current odds
- **Daily Slate** — Tiered recommendations with bet slip
- **Best Lines** — Line shopping across books
- **Performance** — CLV analytics and charts
- **Bet History** — Track placed bets
- **Calibration** — Self-calibrating tier thresholds
- **Market Structure** — Sharp/retail divergence analysis
- **Diagnostics** — Debug mode for pipeline inspection

---

## 9. Frontend / UX

### What the User Sees
A Streamlit web app with sidebar navigation. Primary value is on the Daily Slate page showing tiered recommendations.

### Key UX Issues

1. **Information overload** — Each recommendation shows too many metrics (quality_score, alpha_score, hybrid_score, edge_z, edge_pct, ev_100, confidence, confidence_label, tier, quality_tier, books_used, hold, sigma, Kelly...). A regular user cannot process this.

2. **Dual-confidence confusion** — The user sees both a "confidence" label and a "confidence_label" which can conflict.

3. **Dashboard is a monolith** — 3,730 lines in one file. No component reuse. Adding features means editing a massive file.

4. **Debug mode is useful but user-facing** — The explainability ("Why this pick?") feature is well-built but exposes implementation details that may confuse non-technical users.

5. **Bet slip exists but is session-only** — Bet slip state lives in `st.session_state` unless explicitly persisted. Refreshing the page loses the slip.

### What's Actually Useful
- Tier 1a/1b picks with edge_z, EV/$100, and Kelly sizing
- CLV performance tracking over time
- Line comparison across books
- The pruning/ranking pipeline produces genuinely useful filtered output

### What's Cosmetic
- Multiple tiering method comparisons (user doesn't need 5 methods)
- Market-weight display (internal scoring detail)
- Alpha component breakdowns (useful for development, not for betting)

---

## 10. Setup / Environment / Operations

### Install
```bash
pip install -e ".[dev]"
```

### Required Environment Variables
| Variable | Required? | Purpose |
|---|---|---|
| `ODDS_API_KEY` | Yes (for fetch) | The Odds API key |
| `DB_PATH` | No | Custom SQLite path (default: `~/.line_tracker/lines.db`) |
| `DISPLAY_TZ` | No | Timezone for display (default: America/Chicago) |
| `TIER_METHOD` | No | Tiering method (default: quantile) |
| 25+ tuning knobs | No | See `config.py` for all configurable parameters |

### Run Commands
```bash
# CLI
python -m line_tracker fetch --sport nba
python -m line_tracker cycle --sport nba

# Dashboard
PYTHONPATH=src streamlit run src/line_tracker/dashboard.py

# Tests
python -m pytest tests/ -q
```

### Dependencies
- Python >= 3.10
- httpx (HTTP client for API)
- streamlit (dashboard)
- pandas (data manipulation)
- matplotlib (charts)
- SQLite (built-in, no external DB)

### What's Missing from Setup
1. **No Docker/Compose** — No containerization.
2. **No scheduler** — No cron job, Celery, or APScheduler for automated cycling.
3. **No `.env.example`** — User must figure out env vars from reading `config.py`.
4. **No seed data** — Fresh installs start with empty DB.

---

## 11. What Is Broken / Unclear / Risky

### A. Probably Broken

1. **CLV closing-line matching** (`rec_snapshot_service.py:_find_closing_odds` lines 234-286):
   ```python
   if selection.lower() in (ln.home_team or "").lower():
       candidate = ln.home_value
   ```
   This substring match is fragile. "Phoenix" would match "Phoenix Suns" but also any other team containing "Phoenix". For totals (Over/Under), it falls through to home_value as a fallback, which is wrong. This will produce silent bad CLV data.

2. **`BetRecommendation.home_value` ambiguity**: For moneylines, `home_value` and `away_value` are odds. For spreads, they're point spreads. For totals, they're the over/under numbers. The closing-line lookup doesn't account for this difference.

### B. Probably Incomplete

1. **CLV model training** (`clv_model_service.py`): The `train-model` command exists but the model is simple feature-based lookup, not an actual ML model. The "predicted_clv_positive_prob" is computed from historical bucket stats, not a trained model.

2. **Outcome import** is manual CSV-only. No API integration for automated result fetching.

3. **Published slates** (migrations 0005/0006): Tables exist for `published_slates` and `slate_picks` but usage in `publish_service.py` is basic — this appears to be infrastructure for a future feature (publishing picks to subscribers/API consumers).

4. **Alpha gating** (`alpha.py:ALPHA_GATE_ENABLED = False`): Built but never turned on. The system computes alpha in shadow mode. Need to decide whether to enable it.

5. **Three-way market support** (`best_bets.py:_is_three_way_market`): Soccer ML markets are detected and skipped. No actual three-way probability handling is implemented.

### C. Likely Technical Debt

1. **`bet_slip.py` is pure indirection**: Every function is a 1-line wrapper around `core/math.py`. Should be eliminated; callers should import `core/math` directly.

2. **5 tiering methods in `tiering.py`**: Only `hybrid` and `quantile` are actively used. `absolute`, `percentile`, and `composite` are dead code paths (tested but unused in production flows).

3. **Dual confidence concepts**: `confidence` (edge_z based) vs `confidence_label` (composite quality). These should be unified or clearly distinguished in the UI.

4. **Hardcoded constants scattered**: `_SHRINKAGE_K = 5.0`, `_OUTLIER_REL_THRESHOLD = 0.15`, `RECENCY_HALF_LIFE_MIN = 60.0` etc. are embedded in `best_bets.py`. Some are in `config.py` but many aren't, making tuning inconsistent.

5. **Dashboard at 3,730 lines**: Needs decomposition into page-per-file structure.

6. **BOOK_WEIGHTS are hardcoded** (`best_bets.py`): Sharp book weights (Pinnacle=3.0, Circa=2.5, etc.) are hardcoded. Should be configurable. Also, Pinnacle and Circa aren't available via The Odds API for US markets, making their weights potentially irrelevant unless the user has access to non-standard data sources.

### D. Major Ambiguity Needing Human Clarification

1. **Which scoring system is canonical?** The system computes quality_score, alpha_score, hybrid_score, tier_score, composite_score. Which one should the user actually look at? Which one drives the final ranking?

2. **What is the target user?** Is this for the developer only, or intended for other users? This affects whether the debug/explainability features should be prominent or hidden.

3. **What is the intended deployment?** Streamlit Cloud? Railway? Local only? This affects DB persistence, scheduling, and operational concerns.

4. **Should alpha gating be enabled?** `ALPHA_GATE_ENABLED = False`. What data should inform this decision?

5. **What ranking mode should be default?** `hybrid`, `hit`, or `value`? The system defaults to `hybrid` but gives users the choice, which may be confusing.

---

## 12. Recommended Product Direction

### What the Clean Version Should Be

A focused pipeline:

```
Fetch → Consensus Probability → Edge Detection → Quality Filtering → Tiered Ranking → Display
                                                                                        ↓
                                                                             CLV Tracking (automated)
```

### What the Core Engine Should Own
1. **One consensus method** (weighted robust is best)
2. **One scoring system** (alpha_score is the most principled; deprecate quality_tier, composite_score)
3. **One tiering method** (quantile with market weighting is the most adaptive)
4. **One ranking mode** (hybrid as default, with hit/value as advanced toggles)
5. **One confidence concept** (merge into a single label)

### What Should Be Simplified or Removed
- Eliminate `bet_slip.py` (pure wrapper)
- Remove `absolute`, `percentile`, `composite` tiering methods (keep `quantile` and `hybrid`)
- Merge the two confidence concepts into one
- Move hardcoded constants from `best_bets.py` into `config.py`
- Break `dashboard.py` into per-page modules

### What Should Be Deferred
- ML-based CLV prediction model (current bucket-stats approach is fine for now)
- Parlay correlation adjustments
- Multi-user support
- External publication API

### The Real MVP
- Fetch odds → compute consensus → identify edges → produce 1-5 top picks per sport → track CLV
- That's it. Everything else is secondary.

### What Must Be Reliable First
1. CLV closing-line capture (currently fragile)
2. Automated fetching on a schedule
3. Outcome import (at least semi-automated)

---

## 13. Prioritized Next Steps

### 1. Immediate Cleanup (1-2 days)
- [ ] **Delete `bet_slip.py`** and update all imports to use `core/math.py` directly
  - Files affected: `best_bets.py`, `dashboard.py`, `bet_history.py`
  - Why: Pure dead indirection
- [ ] **Unify confidence concepts**: Rename `confidence` → `edge_confidence` and `confidence_label` → `quality_confidence` (or merge them)
  - Files: `best_bets.py:_build_rec`, `scoring.py:compute_confidence_label`
  - Why: Semantic confusion between two different "confidence" fields
- [ ] **Move BOOK_WEIGHTS to config.py** (or at least to a central constants file)
  - File: `best_bets.py`
  - Why: Hardcoded values that users will want to tune

### 2. Stabilization (2-3 days)
- [ ] **Fix CLV closing-line matching**: Use `api_event_id` + market + explicit side matching instead of substring matching
  - File: `rec_snapshot_service.py:_find_closing_odds`
  - Why: Currently produces silently wrong CLV data
  - This is the highest priority bug
- [ ] **Add `.env.example`** with all configurable variables
  - Why: New users/collaborators can't set up without reading source
- [ ] **Break `dashboard.py` into page modules**: Create `ui/pages/` with one file per page
  - Why: 3,730 lines is unmaintainable

### 3. Core Logic Fixes (1-2 days)
- [ ] **Decide on canonical scoring path**: Document which scores are primary (recommendation: alpha_score for quality, edge_z for significance, hybrid_score for ranking)
  - Files: `scoring.py`, `tiering.py`, `slate.py`
  - Why: 6+ overlapping scores create confusion
- [ ] **Remove unused tiering methods**: Keep `quantile` and `hybrid`, remove `absolute`, `percentile`, `composite`
  - File: `tiering.py`
  - Why: Dead code paths that add confusion
- [ ] **Evaluate alpha gating**: Turn on `ALPHA_GATE_ENABLED` with a feature flag in config
  - File: `alpha.py`, `config.py`
  - Why: Shadow mode has run long enough; time to decide

### 4. Workflow Fixes (2-3 days)
- [ ] **Add automated scheduling**: Use APScheduler or a simple cron wrapper for periodic `cycle` runs
  - New file: `services/scheduler_service.py`
  - Why: Manual fetching defeats the purpose of automation
- [ ] **Automate outcome collection**: Use The Odds API scores endpoint or another source
  - File: `services/outcome_service.py`
  - Why: Manual CSV import is the weakest link in the feedback loop

### 5. Product Clarity Decisions (needs human input)
- [ ] Decide: single-user tool or multi-user platform?
- [ ] Decide: deploy to Streamlit Cloud, Railway, or local-only?
- [ ] Decide: which ranking mode is default? (hybrid recommended)
- [ ] Decide: what information should the primary card show? (reduce to 4-5 key metrics)

### 6. Nice-to-Have Later
- [ ] Probability calibration against actual outcomes
- [ ] Parlay builder with correlation awareness
- [ ] Bankroll tracker with Kelly-based position sizing
- [ ] CLV prediction model (real ML, not bucket stats)
- [ ] API endpoint for programmatic access to picks
- [ ] Notification system (email/SMS/Discord for tier1a picks)

### Quickest Path to...

**Usable version**: Fix CLV closing-line matching + add `.env.example` + document the "fetch → cycle → check" workflow. (1-2 days)

**Trustworthy version**: Above + validate consensus probabilities against outcomes + enable alpha gating + add scheduling. (1-2 weeks)

**Demo-ready version**: Above + break dashboard into pages + simplify the card display to 5 key metrics + add a "Getting Started" onboarding flow. (2-3 weeks)

---

## 14. ChatGPT Handoff Brief

```
CHATGPT HANDOFF BRIEF — Line Tracker

PROJECT:
Line Tracker is a Python sports betting analytics engine. It scrapes odds from
8+ US sportsbooks via The Odds API, computes vig-free consensus probabilities,
identifies +EV edges with Bayesian shrinkage and z-score significance testing,
and produces tiered daily bet recommendations. It tracks CLV (Closing Line Value)
to validate whether recommendations beat the closing line. Built with Python 3.10+,
SQLite, Streamlit, and httpx.

ARCHITECTURE:
- Scraper (The Odds API) → SQLite (lines, events, rec_snapshots, outcomes, bets)
- Analysis engine: best_bets.py (consensus + EV + quality) → slate.py (daily slate)
  → tiering.py (tier assignment) → scoring.py (alpha/hybrid/Kelly enrichment)
  → pick_pruning_service.py (8-gate cascade) → ranking_service.py (top picks)
- Frontend: Streamlit dashboard (3,730 lines, single file — needs decomposition)
- CLI: 12 subcommands including fetch, slate, cycle, import-outcomes, kpi
- All 2,387 tests pass.

CURRENT WORKFLOW:
1. Fetch odds from API → save to SQLite
2. Build recommendations per event (vig removal, consensus prob, EV, shrinkage, z-score)
3. Assign tiers (1a/1b/2/3/stay-away) via quantile bucketing with market weights
4. Enrich with alpha score, hybrid score, Kelly sizing, confidence labels
5. Prune via 8-gate cascade (tier, alpha, edge_z, ev_shrunk, quality, books, hold, CLV)
6. Rank via hybrid/hit/value mode
7. Log snapshots; later capture closing lines for CLV tracking

CURRENT BLOCKERS:
1. CLV closing-line capture uses fragile substring matching on team names
   (rec_snapshot_service.py:_find_closing_odds). Produces wrong data silently.
2. No scheduler — user must manually run fetch/cycle commands.
3. Too many overlapping scoring systems (quality_score, alpha_score, hybrid_score,
   tier_score, composite_score, confidence, confidence_label). Needs consolidation.
4. Dashboard is a 3,730-line monolith. Needs decomposition.
5. Alpha gating is computed but disabled (ALPHA_GATE_ENABLED = False).

MATH SUMMARY:
- Vig removal: proportional normalization (correct)
- Consensus: trimmed mean (≥5 books) or median (<5); also weighted robust consensus
- EV: p * d - 1 (standard)
- Shrinkage: edge_ev * n_eff / (n_eff + 5.0) (empirical Bayes)
- Edge Z: edge_ev_shrunk / max(robust_sigma * decimal_odds, 0.02)
- Alpha: 0-100 smoothstep composite (market robustness + edge robustness + agreement - longshot penalties)
- Kelly: (p*d - 1)/(d - 1), capped at 25%, then × confidence multiplier
- CLV: close_implied_prob - open_implied_prob (positive = beat the close)

IMPORTANT FILES:
- best_bets.py — Core engine (1100 lines). Consensus, EV, quality, Kelly.
- slate.py — Daily slate builder with tier cascade thresholds.
- scoring.py — Alpha, hybrid, market-weight, confidence-label, Kelly effective.
- alpha.py — Robustness overlay (0-100 smoothstep scoring).
- tiering.py — 5 tiering methods (quantile is primary).
- config.py — 30+ configurable parameters via env/secrets.
- storage.py — SQLite wrapper composing 10 repository classes.
- pick_pruning_service.py — 8-gate cascade for pick quality control.
- automation_service.py — Full cycle orchestrator.
- rec_snapshot_service.py — Snapshot logging + CLV capture (has the fragile matching bug).
- dashboard.py — 3,730-line Streamlit app (needs decomposition).

QUESTIONS TO ASK CHATGPT NEXT:
1. "Here is my CLV closing-line matching code [paste _find_closing_odds]. How should
   I fix it to use proper event_id + market + side matching instead of substring matching?"
2. "I have 6 scoring systems that overlap. Here are their formulas [paste from brief].
   Help me decide which to keep as canonical and which to deprecate."
3. "My dashboard.py is 3,730 lines. Help me plan a decomposition into per-page modules
   while keeping Streamlit session state working correctly."
4. "I want to add APScheduler for periodic fetch+cycle runs. Here's my current
   automation_service.py [paste]. How should I integrate scheduling?"
5. "My alpha gating is disabled (ALPHA_GATE_ENABLED = False). Here's the alpha scoring
   logic [paste alpha.py]. What data analysis should I do to decide whether to enable it?"

RECOMMENDED PRIORITIES:
1. Fix CLV closing-line matching (highest impact bug)
2. Consolidate scoring systems (reduce cognitive load)
3. Add scheduling (enable true automation)
4. Decompose dashboard (enable maintainability)
5. Enable alpha gating (improve pick quality)
6. Automate outcome import (close the feedback loop)

ANYTHING ELSE CHATGPT SHOULD KNOW:
- The codebase is well-tested (2,387 tests, all passing). Any refactoring should
  maintain test coverage.
- config.py uses Streamlit secrets fallback → env vars. All 30+ parameters are
  tunable without code changes.
- The system has a "Pro" mode (stricter thresholds) and "Standard" mode (broader).
  "Auto" mode uses self-calibrated thresholds from CLV history.
- Book weights (Pinnacle=3.0, Circa=2.5, etc.) are hardcoded but Pinnacle/Circa
  aren't typically available via The Odds API for US markets.
- The project uses a clean repository pattern for DB access (10 repos in db/repos/).
```

---

## 15. File-by-File Appendix

### Core Engine Files

| File | Lines | Role | Key Functions | Stability | Priority |
|---|---|---|---|---|---|
| `best_bets.py` | ~1100 | Core consensus + EV engine | `recommend_best_bets()`, `_remove_vig()`, `weighted_robust_consensus()`, `compute_shrinkage()`, `compute_edge_z()`, `_build_rec()` | Stable, well-tested | Critical |
| `slate.py` | ~700 | Daily slate builder | `build_daily_slate()`, `_classify_candidate()`, tier cascade logic | Stable | Critical |
| `scoring.py` | ~490 | Alpha/hybrid/Kelly scoring | `enrich_entry()`, `compute_alpha_fields()`, `compute_hybrid_fields()`, `rank_candidates()`, `compute_confidence_label()`, `compute_kelly_effective()` | Stable | Critical |
| `alpha.py` | ~230 | Alpha robustness overlay | `alpha_score()`, `alpha_label()`, `smoothstep()` | Stable | High |
| `tiering.py` | ~780 | Pluggable tiering system | `assign_tiers()`, `assign_tiers_quantile()`, `tier_score()` | Stable but over-engineered | High |
| `models.py` | ~95 | Data models | `BettingLine`, `BestBetResult`, `BetType` | Stable | Critical |
| `config.py` | ~320 | Centralized configuration | 30+ getter functions | Stable | High |
| `storage.py` | ~510 | SQLite persistence | `LineStore` class with 10 repos | Stable | Critical |

### Service Files

| File | Lines | Role | Key Functions | Stability | Priority |
|---|---|---|---|---|---|
| `automation_service.py` | ~240 | Cycle orchestrator | `run_cycle()`, `kpi_report()` | Stable | Critical |
| `rec_snapshot_service.py` | ~290 | Snapshot + CLV capture | `build_snapshot_rows()`, `capture_closing_lines()`, `compute_clv_for_snapshot()` | **Has bug in `_find_closing_odds()`** | Critical — fix first |
| `pick_pruning_service.py` | ~190 | 8-gate cascade | `prune_picks()`, `prune_picks_with_reasons()` | Stable | High |
| `slate_service.py` | ~45 | Slate wrapper | `build_daily_slate_service()` | Stable, thin | Medium |
| `clv_selection_service.py` | ~135 | CLV filter profiles | `build_clv_filter_profile()`, `passes_clv_filter()` | Stable | Medium |
| `ranking_service.py` | ~? | Top-pick selection | `select_top_picks()` | Stable | Medium |

### Infrastructure Files

| File | Lines | Role | Stability | Priority |
|---|---|---|---|---|
| `scraper.py` | ~230 | The Odds API client | Stable | High |
| `core/math.py` | ~130 | Canonical math functions | Stable, well-tested | Critical |
| `core/clv.py` | ~? | CLV metrics | Stable | Medium |
| `db/migrate.py` | ~? | Migration runner | Stable | Low |
| `bet_slip.py` | ~200 | Pure wrappers — **candidate for deletion** | Stable but unnecessary | Low |
| `bet_history.py` | ~? | Bet state management | Stable | Low |
| `dashboard.py` | 3730 | Streamlit monolith — **needs decomposition** | Works but unmaintainable | High |

### Files to Inspect First (in order)

1. **`rec_snapshot_service.py:_find_closing_odds`** — Fix the CLV matching bug
2. **`scoring.py`** — Understand all scoring paths, decide which to keep
3. **`best_bets.py:_build_rec`** — The heart of the recommendation builder
4. **`config.py`** — All tunable parameters
5. **`slate.py:build_daily_slate`** — How tiers are actually assigned
6. **`automation_service.py:run_cycle`** — The complete pipeline in one function
7. **`dashboard.py`** — Plan decomposition
