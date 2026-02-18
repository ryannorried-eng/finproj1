# Line Tracker

A sports-betting analytics platform that ingests live odds from multiple sportsbooks, computes consensus probabilities, surfaces tiered bet recommendations with full explainability, tracks closing-line value (CLV), and presents everything through an interactive Streamlit dashboard.

## Features

- **Live odds ingestion** from The Odds API across NFL, NBA, MLB, NHL, NCAAF, NCAAB, MMA, and Soccer (EPL)
- **Consensus probability engine** using vig removal, sharp-book weighting, recency decay, and robust aggregation (trimmed mean / weighted median)
- **Edge & EV scoring** with Bayesian shrinkage and Kelly criterion sizing
- **Tiered recommendations** via pluggable strategies (absolute, percentile, hybrid, composite) with hard "stay-away" gates
- **Arbitrage detection** across moneyline and spread markets
- **Closing-line value tracking** to measure long-term betting profitability
- **Interactive Streamlit dashboard** with seven pages: Slate, Shopping, Bet History, Performance, Arbitrage, Debug, and Pro (calibration)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Web UI | Streamlit |
| Persistence | SQLite (WAL mode) |
| HTTP client | httpx |
| Data analysis | pandas, matplotlib |
| Testing | pytest (35 test modules) |
| Linter | ruff |

## Project Layout

```
src/line_tracker/
  scraper.py              # Odds API client
  ingestion_service.py    # Fetch, normalize, persist lines
  storage.py              # SQLite line store
  db/repos/               # Repository pattern (lines, bets, CLV, calibration)
  db/migrate.py           # Schema versioning via numbered SQL migrations
  best_bets.py            # Consensus probabilities & edge scoring
  tiering.py              # Pluggable tier-assignment strategies
  slate.py                # Daily slate aggregator
  arbitrage.py            # Arb detection
  movements.py            # Line movement tracking
  calibration.py          # CLV-driven threshold tuning
  performance.py          # CLV analytics
  dashboard.py            # Streamlit app (7 pages)
  services/               # Orchestration layer
  ui/components/          # Reusable UI components
  core/math.py            # Odds conversion, Kelly, EV helpers
  core/clv.py             # CLV metric calculations
```

## Quick Start

```bash
# Install in editable mode
python -m pip install -e .

# Run the dashboard
streamlit run src/line_tracker/dashboard.py

# Run tests
pytest -q

# Lint
ruff check src/ tests/
```

## Documentation

- [Technical Brief](docs/technical_brief.md) — detailed architecture, data models, algorithms, and design patterns
- [Streamlit Debugging Guide](docs/debug_streamlit.md) — local setup, import diagnostics, and known gotchas
