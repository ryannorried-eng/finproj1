# Line Tracker

Track sports betting lines across sportsbooks. Compare odds, detect line movements, and find arbitrage opportunities.

## Getting Started

### 1. Clone or Pull the Repository

**Clone (first time):**

```bash
git clone https://github.com/ryannorried-eng/finproj1.git
cd finproj1
```

**Pull latest changes (existing clone):**

```bash
cd finproj1
git pull origin master
```

### 2. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 3. Run the Streamlit App

```bash
streamlit run src/line_tracker/dashboard.py
```

The app will start and print a URL in your terminal (typically `http://localhost:8501`). Open that URL in a browser to use the dashboard.

To specify a custom port:

```bash
streamlit run src/line_tracker/dashboard.py --server.port 8080
```

### 4. CLI Usage

You can also use the command-line interface:

```bash
# List available sports
python -m line_tracker sports

# Fetch live odds
python -m line_tracker fetch --sport nba

# Show stored lines
python -m line_tracker lines

# Scan for arbitrage opportunities
python -m line_tracker arbs --sport nba

# Detect line movements
python -m line_tracker moves

# List tracked events
python -m line_tracker events
```

## API Key

The app uses [The Odds API](https://the-odds-api.com/) for live data. You can enter your API key in the sidebar of the Streamlit dashboard, or set it as an environment variable:

```bash
export ODDS_API_KEY="your_key_here"
```

## Supported Sports

NFL, NBA, MLB, NHL, NCAAF, NCAAB, MMA, Soccer (EPL)
