"""NRFI/YRFI first-inning probability estimates for MLB games."""

from __future__ import annotations

from datetime import date

EDGE_THRESHOLD = 0.05
BREAKEVEN_PROB = 0.524  # win rate needed to break even at -110 odds

_LEAGUE_AVG_ERA = 4.20
_BASE_PER_TEAM_YRFI = 0.30  # ~51% combined YRFI at league-avg ERA
_ERA_SCALE = 0.025  # probability shift per ERA point above/below league avg


def _per_team_yrfi(era: float | None) -> float:
    if era is None:
        return _BASE_PER_TEAM_YRFI
    return max(0.10, min(0.60, _BASE_PER_TEAM_YRFI + (era - _LEAGUE_AVG_ERA) * _ERA_SCALE))


def _sp_fi_yrfi_rate(fi_history, sp_id) -> float | None:
    if fi_history is None or fi_history.empty or sp_id is None:
        return None
    try:
        sp_id_int = int(sp_id)
    except (TypeError, ValueError):
        return None
    mask = (
        (fi_history["home_sp_id"] == sp_id_int)
        | (fi_history["away_sp_id"] == sp_id_int)
    )
    sp_games = fi_history[mask].sort_values("date").tail(10)
    if len(sp_games) < 5:
        return None
    return float(sp_games["yrfi"].mean())


def predict_nrfi(target_date: date | None = None) -> list[dict]:
    """Return NRFI/YRFI probability estimates for each game on target_date.

    Wraps predict_mlb_games() and estimates combined first-inning scoring
    probability from each starter's ERA vs the league average.

    Keys in each returned dict:
        game_pk, away_team, home_team, away_team_br, home_team_br,
        away_pitcher, home_pitcher, away_pitcher_era, home_pitcher_era,
        yrfi_prob, nrfi_prob, commence_time
    """
    if target_date is None:
        from datetime import date as _date
        target_date = _date.today()
    current_year = target_date.year

    fi_history = None
    try:
        from line_tracker.model.mlb_data import fetch_first_inning_data
        df = fetch_first_inning_data(
            seasons=[current_year - 1, current_year],
            force_refresh=False,
        )
        if df is not None and not df.empty:
            fi_history = df
    except Exception as exc:
        pass

    try:
        from line_tracker.model.mlb_predict import predict_mlb_games
        preds = predict_mlb_games(target_date=target_date)
    except Exception:
        return []

    results = []
    for p in preds:
        away_era = p.get("away_pitcher_era")
        home_era = p.get("home_pitcher_era")
        home_pid = p.get("home_pitcher_id")
        away_pid = p.get("away_pitcher_id")

        home_fi_rate = _sp_fi_yrfi_rate(fi_history, home_pid)
        away_fi_rate = _sp_fi_yrfi_rate(fi_history, away_pid)

        # Blend fi_rate with ERA-based estimate (50/50) to reduce small sample noise
        era_home = _per_team_yrfi(home_era)
        era_away = _per_team_yrfi(away_era)
        home_yrfi = (0.5 * home_fi_rate + 0.5 * era_home) if home_fi_rate is not None else era_home
        away_yrfi = (0.5 * away_fi_rate + 0.5 * era_away) if away_fi_rate is not None else era_away
        combined_yrfi = round(1.0 - (1.0 - away_yrfi) * (1.0 - home_yrfi), 4)
        results.append({
            "game_pk": p.get("game_id", ""),
            "away_team": p.get("away_team", ""),
            "home_team": p.get("home_team", ""),
            "away_team_br": p.get("away_team_br", ""),
            "home_team_br": p.get("home_team_br", ""),
            "away_pitcher": p.get("away_pitcher") or "TBD",
            "home_pitcher": p.get("home_pitcher") or "TBD",
            "away_pitcher_era": away_era,
            "home_pitcher_era": home_era,
            "away_sp_fi_yrfi_rate": round(away_fi_rate, 3) if away_fi_rate is not None else None,
            "home_sp_fi_yrfi_rate": round(home_fi_rate, 3) if home_fi_rate is not None else None,
            "yrfi_prob": combined_yrfi,
            "nrfi_prob": round(1.0 - combined_yrfi, 4),
            "commence_time": p.get("commence_time", ""),
        })
    return results
