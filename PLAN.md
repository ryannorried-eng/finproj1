# Implementation Plan: Steps 1–7

## Overview

Seven features, implemented as **new modules** under `services/` and `core/`, a new
migration, and new CLI commands.  No changes to the Streamlit dashboard.

---

## File-by-File Changes

### 1. New migration: `src/line_tracker/db/migrations/0010_outcomes_and_clv_model.sql`

Creates two new tables:

```sql
-- Stores settled game outcomes imported from CSV.
CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id       TEXT NOT NULL,
    market         TEXT NOT NULL,
    selection      TEXT NOT NULL,
    result         TEXT NOT NULL,  -- 'win' | 'loss' | 'push'
    settled_at     TEXT,
    created_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(event_id, market, selection)
);

-- Stores per-pick ROI after matching outcomes to rec_snapshots.
-- Also extends rec_snapshots with an outcome_result + actual_roi column.
ALTER TABLE rec_snapshots ADD COLUMN outcome_result TEXT;
ALTER TABLE rec_snapshots ADD COLUMN actual_roi REAL;

-- CLV model feature table: stores features + predicted CLV-positive probability
-- per (market, behavior-signature) pair, learned from historical rec_snapshots.
CREATE TABLE IF NOT EXISTS clv_model_scores (
    score_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    market         TEXT NOT NULL,
    selection_type TEXT,  -- e.g. 'home_fav', 'away_dog', 'over', 'under'
    feature_json   TEXT NOT NULL,
    predicted_clv_positive_prob REAL NOT NULL,
    model_version  TEXT NOT NULL DEFAULT 'v1',
    UNIQUE(market, selection_type, model_version)
);
```

### 2. New repo: `src/line_tracker/db/repos/outcomes_repo.py`

- `OutcomesRepo(conn)` with methods:
  - `import_from_rows(rows: list[dict]) -> int` — bulk upsert from CSV dicts
  - `get_for_event(event_id, market, selection) -> dict | None`
  - `get_all() -> list[dict]`

### 3. New repo: `src/line_tracker/db/repos/clv_model_repo.py`

- `ClvModelRepo(conn)` with methods:
  - `save_scores(scores: list[dict]) -> int`
  - `get_score(market, selection_type) -> dict | None`
  - `get_all_scores() -> list[dict]`

### 4. Update `src/line_tracker/db/repos/__init__.py`

Add exports for `OutcomesRepo`, `ClvModelRepo`.

### 5. Update `src/line_tracker/storage.py`

- Add `outcomes_repo` and `clv_model_repo` to `LineStore.__init__`
- Add convenience methods:
  - `import_outcomes(rows) -> int`
  - `get_outcome(event_id, market, selection) -> dict | None`
  - `link_outcomes_to_snapshots() -> int` — joins outcomes to rec_snapshots and sets `outcome_result` + `actual_roi`
  - `get_clv_model_score(market, selection_type) -> dict | None`

### 6. New service: `src/line_tracker/services/clv_selection_service.py` (Step 1 — CLV-driven selection)

**Purpose**: Query historical CLV data from `rec_snapshots` to learn which (market, tier, alpha_label, edge_z range, books range, hold range) combinations historically produce CLV > 0.  Produce a "CLV filter profile" dict.

- `build_clv_filter_profile(store) -> dict` — Queries `get_closed_snapshots()`, groups by market × tier × alpha_label, computes `pct_positive_clv` and `avg_clv_implied` for each group, returns a nested dict.
- `passes_clv_filter(entry: dict, profile: dict) -> bool` — Given a slate entry and the filter profile, returns True if the entry's group historically has `pct_positive >= 55%` (configurable threshold).

### 7. New service: `src/line_tracker/services/pick_pruning_service.py` (Step 2 — Aggressive pick pruning)

**Purpose**: Tight gates on top of existing tiering. Only entries that survive ALL gates are promoted to "actionable picks."

- `prune_picks(entries: list[dict], *, clv_profile: dict | None = None) -> list[dict]` — Applies a strict gate cascade:
  1. `tier` must be in `("tier1b", "tier2")` — no tier3
  2. `alpha_label` must be `"Strong"` or `"Neutral"`
  3. `edge_z >= 1.75`
  4. `edge_ev_shrunk > 0`
  5. `quality_score >= 65`
  6. `books_used >= 5`
  7. `market_hold_median <= 7.0`
  8. If `clv_profile` provided: `passes_clv_filter(entry, profile)` must be True

  Returns the filtered list.

### 8. New service: `src/line_tracker/services/automation_service.py` (Step 3 — Automation scheduler)

**Purpose**: A single `run_cycle()` function that performs the full fetch→slate→snapshot loop.

```python
def run_cycle(
    store, api_key: str, *, sport: str,
    clv_profile: dict | None = None,
    dry_run: bool = False,
) -> dict:
    """Execute one automation cycle: fetch → slate → prune → snapshot."""
```

Steps:
1. `ingestion_service.fetch_and_persist_snapshot(store, api_key, sport=sport)`
2. Group lines by event from `store.get_latest_for_api_event()`
3. `slate_service.build_daily_slate_service(lines_by_event, ...)`
4. `pick_pruning_service.prune_picks(all_entries, clv_profile=clv_profile)`
5. `rec_snapshot_service.build_snapshot_rows(slate)` → `store.log_rec_snapshots()`
6. `rec_snapshot_service.capture_closing_lines(store)` (for previously-open snapshots)
7. Return summary dict.

### 9. New service: `src/line_tracker/services/ranking_service.py` (Step 4 — Ranking-first betting, top tail only)

**Purpose**: After pruning, rank by `hybrid_score` descending and take only the top N (default 3).

- `select_top_picks(pruned: list[dict], *, top_n: int = 3) -> list[dict]`
  Sorts by `hybrid_score` desc, returns `pruned[:top_n]`.
- `format_picks_report(picks: list[dict]) -> str`
  Human-readable CLI report.

### 10. Update `src/line_tracker/services/rec_snapshot_service.py` (Step 5 — Volume via frequent snapshots)

Add a convenience wrapper:

- `log_snapshot_for_picks(store, picks: list[dict], *, sport: str) -> int`
  Converts a list of pick dicts into snapshot rows and inserts them.  This is called by the automation loop at each cycle; volume comes from running the cycle frequently (e.g. every 15 min), not from relaxing thresholds.

### 11. New service: `src/line_tracker/services/outcome_service.py` (Step 6 — Outcome feedback loop)

**Purpose**: Import CSV outcomes and compute ROI.

- `import_outcomes_csv(store, csv_path: str) -> int` — Reads a CSV with columns `event_id, market, selection, result`, upserts into `outcomes` table.
- `link_outcomes(store) -> int` — Joins `outcomes` to `rec_snapshots` on `(event_id, market, selection)`, sets `outcome_result` and computes `actual_roi = (odds_decimal - 1) * result_multiplier` where win=1, loss=-1, push=0.
- `roi_report(store) -> dict` — Queries `rec_snapshots` where `outcome_result IS NOT NULL`, aggregates by tier, alpha_label, and overall: `total_picks, wins, losses, pushes, total_staked_units, net_roi_units, roi_pct`.

### 12. New service: `src/line_tracker/services/clv_model_service.py` (Step 7 — "Market + behavior" model to predict CLV > 0)

**Purpose**: Train a simple logistic-regression-style model (no external ML libs — pure Python) from closed `rec_snapshots` features.

- `extract_features(snap: dict) -> dict` — Pulls: `edge_z`, `edge_ev_shrunk`, `quality_score`, `books_used`, `market_hold_median`, `alpha_score`, `consensus_prob`, and derives `market_type` (moneyline/spread/total) and `selection_type` (home_fav/away_dog/over/under heuristic).
- `train_clv_model(store) -> dict` — Loads all closed snapshots, extracts features, computes per-(market, selection_type) logistic scores via simple weighted average of z-scored features, stores results in `clv_model_scores`.
- `predict_clv_positive(entry: dict, store) -> float` — Looks up the model score for the entry's market/selection_type, returns the predicted probability of CLV > 0.
- `refresh_model(store) -> dict` — Retrains and returns summary stats.

### 13. Update `src/line_tracker/__main__.py` (CLI commands)

Add 6 new subcommands:

| Command | Description | Calls |
|---------|-------------|-------|
| `slate` | Build and display daily slate | `slate_service` → print summary |
| `snapshot` | Log recommendation snapshots | `rec_snapshot_service` |
| `cycle` | Run one automation cycle | `automation_service.run_cycle()` |
| `import-outcomes` | Import outcomes from CSV | `outcome_service.import_outcomes_csv()` |
| `roi-report` | Show ROI report | `outcome_service.roi_report()` |
| `train-model` | Train/refresh CLV model | `clv_model_service.refresh_model()` |

Each command takes `--db` and `--sport` where applicable; `cycle` also takes `--api-key` and `--dry-run`.

### 14. New tests

| Test File | Covers |
|-----------|--------|
| `tests/test_clv_selection.py` | `clv_selection_service`: profile building, filter pass/fail |
| `tests/test_pick_pruning.py` | `pick_pruning_service`: gate cascade, edge cases |
| `tests/test_automation.py` | `automation_service.run_cycle()` with mocked fetch |
| `tests/test_ranking_service.py` | `ranking_service`: top-N selection, sort order |
| `tests/test_outcome_service.py` | CSV import, outcome linking, ROI report |
| `tests/test_clv_model.py` | Feature extraction, model training, prediction |
| `tests/test_cli_new_commands.py` | CLI integration for all 6 new commands |

---

## Dependency Graph

```
Step 1 (CLV selection)  ──┐
                          ├──▶ Step 2 (Pick pruning) ──▶ Step 4 (Ranking) ──┐
Step 7 (CLV model)    ──┘                                                   │
                                                                            ▼
Step 5 (Snapshot volume) ◀──────────────────── Step 3 (Automation) ◀────────┘
                                                        │
Step 6 (Outcome feedback) ◀─────────────────────────────┘ (uses outcomes for ROI)
```

Steps 1 and 7 can be built in parallel (both read from `rec_snapshots`).
Step 2 depends on Step 1.
Steps 3, 4, 5 are the integration layer.
Step 6 is independent (CSV import + ROI reporting).

---

## Design Principles

1. **No new external APIs** — outcomes from CSV, model is pure-Python math.
2. **Volume via frequency, not relaxed thresholds** — `prune_picks()` is strict; run `cycle` every 15 min for volume.
3. **Layered architecture** — new modules follow existing patterns: repo → storage → service → CLI.
4. **All new code unit-tested** — each service gets its own test file.
5. **Migration is additive** — new tables/columns, no breaking changes.
6. **CLV model predicts CLV > 0, NOT game outcomes** — features are market-structure and behavior signals.
