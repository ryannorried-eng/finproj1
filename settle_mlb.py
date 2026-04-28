#!/usr/bin/env python3
"""
settle_mlb.py — settle yesterday's MLB model predictions against actual results.

Usage:
    python3 settle_mlb.py              # settles yesterday
    python3 settle_mlb.py 2026-04-25   # settles a specific date
"""

import sqlite3
import sys
import requests
from datetime import date, timedelta

DB_PATH = "lines.db"
MLB_API = "https://statsapi.mlb.com/api/v1"

TEAM_NAME_TO_BR: dict[str, str] = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Athletics": "OAK", "Sacramento Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP", "Seattle Mariners": "SEA",
    "San Francisco Giants": "SFG", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN",
}

CANONICAL: dict[str, str] = {
    "AZ": "ARI", "SF": "SFG", "SD": "SDP", "TB": "TBR",
    "KC": "KCR", "WSH": "WSN", "CWS": "CHW", "ATH": "OAK",
}

def canonical(abbr: str) -> str:
    return CANONICAL.get(abbr, abbr)

def fetch_scores(game_date: str) -> dict:
    url = f"{MLB_API}/schedule?sportId=1&date={game_date}&gameType=R&hydrate=team,linescore"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"API error: {e}")
        return {}

    results = {}
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("status", {}).get("abstractGameState", "") != "Final":
                continue
            try:
                home_abbr = canonical(game["teams"]["home"]["team"]["abbreviation"])
                away_abbr = canonical(game["teams"]["away"]["team"]["abbreviation"])
                home_score = int(game["teams"]["home"]["score"])
                away_score = int(game["teams"]["away"]["score"])
                results[(home_abbr, away_abbr)] = {
                    "home_score": home_score,
                    "away_score": away_score,
                }
            except Exception:
                continue
    return results

def settle(game_date: str) -> None:
    print(f"Settling {game_date}...")
    scores = fetch_scores(game_date)
    if not scores:
        print("No final scores found.")
        return

    print(f"Found {len(scores)} final games from MLB API")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute(
        "SELECT id, game_id, home_team, away_team, model_home_win_prob, "
        "market_total, model_total_runs "
        "FROM mlb_model_predictions WHERE game_date = ? AND settled_at IS NULL",
        (game_date,)
    )
    rows = cur.fetchall()

    if not rows:
        print(f"No unsettled predictions found for {game_date}.")
        con.close()
        return

    print(f"Found {len(rows)} unsettled predictions")
    settled = 0

    for row_id, game_id, home_team, away_team, home_win_prob, market_total, model_total in rows:
        home_br = TEAM_NAME_TO_BR.get(home_team)
        away_br = TEAM_NAME_TO_BR.get(away_team)

        if not home_br or not away_br:
            print(f"  Could not resolve: {home_team} / {away_team}")
            continue

        score = scores.get((home_br, away_br))
        if not score:
            print(f"  No score found for {away_br} @ {home_br}")
            continue

        home_score = score["home_score"]
        away_score = score["away_score"]
        actual_total = home_score + away_score

        # Grade ML
        model_picked_home = (home_win_prob or 0.5) >= 0.5
        home_won = home_score > away_score
        home_win_correct = 1 if (model_picked_home == home_won) else 0

        # Grade total — over/under vs market line
        total_correct = None
        if market_total is not None and model_total is not None:
            if abs(actual_total - market_total) < 0.01:
                total_correct = None  # push
            else:
                model_over = model_total > market_total
                actual_over = actual_total > market_total
                total_correct = 1 if (model_over == actual_over) else 0

        cur.execute("""
            UPDATE mlb_model_predictions SET
                actual_home_score = ?,
                actual_away_score = ?,
                home_win_correct  = ?,
                total_correct     = ?,
                settled_at        = datetime('now')
            WHERE id = ?
        """, (home_score, away_score, home_win_correct, total_correct, row_id))

        ml_str = "WIN" if home_win_correct else "LOSS"
        tot_str = "WIN" if total_correct == 1 else ("PUSH" if total_correct is None else "LOSS")
        print(f"  {away_br} @ {home_br}: {away_score}-{home_score} | ML={ml_str} | Total={tot_str}")
        settled += 1

    con.commit()
    con.close()
    print(f"\nSettled {settled}/{len(rows)} predictions for {game_date}.")

    # Print running record
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        SELECT
            COUNT(*) as total,
            SUM(home_win_correct) as ml_wins,
            SUM(CASE WHEN total_correct = 1 THEN 1 ELSE 0 END) as tot_wins,
            SUM(CASE WHEN total_correct = 0 THEN 1 ELSE 0 END) as tot_losses
        FROM mlb_model_predictions
        WHERE settled_at IS NOT NULL
    """)
    r = cur.fetchone()
    con.close()
    if r and r[0]:
        total, ml_wins, tot_wins, tot_losses = r
        ml_losses = total - (ml_wins or 0)
        print(f"\nAll-time DB record:")
        print(f"  ML:     {int(ml_wins or 0)}-{ml_losses} ({100*(ml_wins or 0)/total:.1f}%)")
        print(f"  Totals: {int(tot_wins or 0)}-{int(tot_losses or 0)} ({100*int(tot_wins or 0)/(int(tot_wins or 0)+int(tot_losses or 0)+0.001):.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        game_date = sys.argv[1]
    else:
        game_date = (date.today() - timedelta(days=1)).isoformat()
    settle(game_date)
