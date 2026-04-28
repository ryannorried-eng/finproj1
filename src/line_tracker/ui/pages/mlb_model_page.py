"""MLB Model Picks page for the Streamlit dashboard.

Pitcher-aware lineup model v4 — powered by MLB Stats API + The Odds API.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

log = logging.getLogger(__name__)

is_deployed = bool(
    os.environ.get("RAILWAY_ENVIRONMENT")
    or os.environ.get("STREAMLIT_SHARING_MODE")
    or os.environ.get("HOME", "").startswith("/home/adminuser")
)


def _fmt_commence(ct_str: str, tz_name: str = "America/Chicago") -> str:
    """Format ISO commence_time string to local time like '3:05 PM CT'."""
    if not ct_str:
        return ""
    try:
        dt = datetime.fromisoformat(ct_str.replace("Z", "+00:00"))
        local = dt.astimezone(ZoneInfo(tz_name))
        suffix = tz_name.split("/")[-1][:2].upper() + "T"
        return local.strftime(f"%-I:%M %p {suffix}")
    except Exception:
        return ct_str


def _display_tz() -> str:
    try:
        from line_tracker.config import get_display_timezone
        return get_display_timezone()
    except Exception:
        return "America/Chicago"


def _american_to_implied(odds_str):
    """Convert American odds string to implied probability."""
    try:
        odds = int(str(odds_str).replace("+", ""))
        if odds < 0:
            return (-odds) / (-odds + 100)
        else:
            return 100 / (odds + 100)
    except Exception:
        return None


def _american_to_prob(odds):
    if not odds or pd.isna(odds):
        return None
    try:
        o = float(odds)
        if o < 0:
            return -o / (-o + 100)
        return 100 / (o + 100)
    except Exception:
        return None


def fmt_ml(odds):
    if not odds:
        return "—"
    return f"+{int(odds)}" if odds > 0 else str(int(odds))


def _format_sp(name, era):
    if not name or name == "TBD":
        return "TBD"
    era_str = f"{era:.2f}" if era and not pd.isna(era) else "---"
    return f"{name} ({era_str})"


def _format_weather(temp, wind, wind_factor):
    if not temp or pd.isna(temp):
        return "—"
    if wind_factor > 0.1:
        direction = "↑out"
    elif wind_factor < -0.1:
        direction = "↓in"
    else:
        direction = "→"
    return f"{temp:.0f}°F {wind:.0f}mph {direction}"


def _format_ml_odds(prob):
    """Convert win probability to American odds string."""
    if not prob or pd.isna(prob):
        return "—"
    prob = max(0.001, min(0.999, prob))
    if prob >= 0.5:
        odds = -(prob / (1 - prob)) * 100
        return f"{int(odds)}"
    else:
        odds = ((1 - prob) / prob) * 100
        return f"+{int(odds)}"


def _market_implied_prob(american_odds):
    """Convert American odds to implied probability."""
    if not american_odds or pd.isna(american_odds):
        return None
    if american_odds < 0:
        return (-american_odds) / (-american_odds + 100)
    else:
        return 100 / (american_odds + 100)


def _predicted_scores(model_total, model_run_diff):
    """
    Derive predicted scores for each team.
    Returns (away_score, home_score) tuple.
    """
    if model_total is None or model_run_diff is None:
        return None, None
    if pd.isna(model_total) or pd.isna(model_run_diff):
        return None, None
    home_score = (model_total + model_run_diff) / 2
    away_score = (model_total - model_run_diff) / 2
    return round(away_score, 1), round(home_score, 1)


def _pred_score_with_flag(model_total, model_run_diff, home_prob):
    """Return predicted score string, flagging model contradictions with ⚠️."""
    away_score, home_score = _predicted_scores(model_total, model_run_diff)
    if away_score is None:
        return "—"
    run_diff = model_run_diff if model_run_diff is not None else 0
    contradiction = (home_prob > 0.5 and run_diff < 0) or (home_prob < 0.5 and run_diff > 0)
    flag = " \u26a0\ufe0f" if contradiction else ""
    return f"{away_score} - {home_score}{flag}"


def _pred_score_abbr(model_total, model_run_diff, home_prob, away_br, home_br):
    """Return predicted score string with team abbreviations and contradiction flag."""
    away_score, home_score = _predicted_scores(model_total, model_run_diff)
    if away_score is None:
        return "—"
    run_diff = model_run_diff if model_run_diff is not None else 0
    contradiction = (home_prob > 0.5 and run_diff < 0) or (home_prob < 0.5 and run_diff > 0)
    flag = " \u26a0\ufe0f" if contradiction else ""
    return f"{away_br} {away_score} - {home_br} {home_score}{flag}"


def _model_spread_pick(model_run_diff, market_spread=None):
    """
    Determine model's run line pick based on predicted margin.
    model_run_diff is always from the HOME team's perspective:
      positive = home team wins by that many runs
      negative = away team wins by that many runs

    market_spread: home team spread point (e.g. -1.5, -2.5). Defaults to -1.5.

    Returns string like "Home -1.5 (+2.2 edge)" or "Away +1.5 (-2.2 edge)" or "—"
    """
    if model_run_diff is None or pd.isna(model_run_diff):
        return "—"

    # Use actual market spread point, defaulting to -1.5 (standard run line)
    if market_spread is not None and not pd.isna(market_spread):
        home_point = float(market_spread)
    else:
        home_point = -1.5
    away_point = -home_point

    if abs(home_point) < 0.5:
        return "Pick'em"

    # Edge = how many runs the model beats the spread by
    edge = abs(model_run_diff) - abs(home_point)
    def _fmt_pt(pt):
        return f"+{pt:.1f}" if pt > 0 else f"{pt:.1f}"

    if model_run_diff > abs(home_point):
        return f"Home {_fmt_pt(home_point)} ({edge:+.1f})"
    elif model_run_diff < -abs(home_point):
        return f"Away {_fmt_pt(away_point)} ({edge:+.1f})"
    else:
        return f"Push zone ({model_run_diff:+.1f})"


def _value_score(model_prob, market_ml_odds):
    """Edge quality score: (model - market) * 100"""
    market_prob = _market_implied_prob(market_ml_odds)
    if not market_prob:
        return None
    return (model_prob - market_prob) * 100


def _format_time(ct_str):
    return _fmt_commence(ct_str or "", _display_tz())


def _best_bet(p):
    home_prob = p.get("model_home_win_prob", 0.5)
    away_prob = 1 - home_prob
    home_ml = p.get("market_home_ml")
    away_ml = p.get("market_away_ml")
    run_diff = p.get("model_run_diff", 0) or 0
    total_edge = p.get("total_edge", 0) or 0
    market_total = p.get("market_total")
    home_br = p.get("home_team_br", "")
    away_br = p.get("away_team_br", "")

    def fmt_odds(o):
        if o is None:
            return ""
        return f"+{int(o)}" if o > 0 else str(int(o))

    # ML value
    home_val = None
    away_val = None
    if home_ml:
        mp = _american_to_prob(home_ml)
        if mp:
            home_val = (home_prob - mp) * 100
    if away_ml:
        mp = _american_to_prob(away_ml)
        if mp:
            away_val = (away_prob - mp) * 100

    best_ml_val = None
    best_ml_team = None
    best_ml_odds = None
    if home_val is not None and away_val is not None:
        if home_val >= away_val and home_val >= 4.0:
            best_ml_val = home_val
            best_ml_team = home_br
            best_ml_odds = home_ml
        elif away_val >= 4.0:
            best_ml_val = away_val
            best_ml_team = away_br
            best_ml_odds = away_ml
    elif home_val is not None and home_val >= 4.0:
        best_ml_val = home_val
        best_ml_team = home_br
        best_ml_odds = home_ml
    elif away_val is not None and away_val >= 4.0:
        best_ml_val = away_val
        best_ml_team = away_br
        best_ml_odds = away_ml

    # Total play — Overs only
    # Only bet Overs — Under signal is 50% (coin flip) based on 46-game sample
    # Over signal is 64% — meaningful edge
    total_play = None
    if total_edge >= 0.8 and market_total:
        total_play = f"Over {market_total} ({total_edge:+.1f})"
    else:
        total_play = None  # Skip unders entirely

    # Run line play — use actual market spread point and odds
    market_spread = p.get("market_spread")
    home_point = float(market_spread) if market_spread is not None else -1.5
    away_point = -home_point
    home_spread_odds = p.get("market_home_spread_odds")
    away_spread_odds = p.get("market_away_spread_odds")

    def fmt_spread_odds(o):
        if o is None:
            return ""
        return f" (+{int(o)})" if o > 0 else f" ({int(o)})"

    def _fmt_pt(pt):
        return f"+{pt:.1f}" if pt > 0 else f"{pt:.1f}"

    rl_play = None
    if abs(home_point) < 0.5:
        rl_play = None  # skip spread bet for pick'em games
    elif run_diff > abs(home_point):
        rl_play = f"{home_br} {_fmt_pt(home_point)}{fmt_spread_odds(home_spread_odds)} ({run_diff:+.1f})"
    elif run_diff < -abs(home_point):
        rl_play = f"{away_br} {_fmt_pt(away_point)}{fmt_spread_odds(away_spread_odds)} ({run_diff:+.1f})"

    # Only surface ML picks where model has sufficient conviction
    # Threshold: favored team >= 57% AND edge >= 4%
    # Based on 4-day data: losses concentrated in 51-56% range
    ml_qualified = p.get("ml_bet_qualified", False)

    if best_ml_val and best_ml_val >= 5.0 and ml_qualified:
        return f"{best_ml_team} ML {fmt_odds(best_ml_odds)}"

    if total_play and abs(total_edge) >= 1.2:
        return total_play

    if best_ml_val and best_ml_val >= 4.0 and ml_qualified:
        return f"{best_ml_team} ML {fmt_odds(best_ml_odds)}"

    if rl_play:
        return rl_play

    if total_play:
        return total_play

    return "—"


def _find_best_bets(predictions):
    """
    Identify actual betting opportunities from predictions.
    Returns list of bet dicts, sorted by value score descending.
    """
    bets = []

    for p in predictions:
        home_prob = p.get("model_home_win_prob", 0.5)
        away_prob = 1 - home_prob
        home_ml = p.get("market_home_ml")
        away_ml = p.get("market_away_ml")
        market_total = p.get("market_total")
        model_total = p.get("model_total_runs")
        total_edge = p.get("total_edge", 0)
        ml_edge = p.get("ml_edge", 0)

        matchup = f"{p['away_team']} @ {p['home_team']}"

        _spread_fields = {
            "market_spread": p.get("market_spread"),
            "market_home_spread_odds": p.get("market_home_spread_odds"),
            "market_away_spread_odds": p.get("market_away_spread_odds"),
            "home_team_br": p.get("home_team_br", ""),
            "away_team_br": p.get("away_team_br", ""),
        }

        # --- Moneyline bets ---
        # Only show ML bets meeting the 57% threshold
        if p.get("ml_bet_qualified", False):
            # Home ML value
            if home_ml and abs(ml_edge) >= 0.04:
                market_prob = _market_implied_prob(home_ml)
                if market_prob and home_prob > market_prob:
                    edge_pct = (home_prob - market_prob) * 100
                    bets.append({
                        "matchup": matchup,
                        "bet_type": "Moneyline",
                        "pick": f"{p['home_team']} ML",
                        "odds": f"+{home_ml}" if home_ml > 0 else str(home_ml),
                        "model_prob": home_prob,
                        "market_prob": market_prob,
                        "edge": edge_pct,
                        "edge_type": "ML",
                        "value_score": edge_pct,
                        "confidence": p.get("confidence"),
                        "away_sp": p.get("away_pitcher", "TBD"),
                        "home_sp": p.get("home_pitcher", "TBD"),
                        "away_era": p.get("away_pitcher_era"),
                        "home_era": p.get("home_pitcher_era"),
                        "temp_f": p.get("temp_f"),
                        "wind_mph": p.get("wind_mph"),
                        "wind_out_factor": p.get("wind_out_factor", 0),
                        "model_total": model_total,
                        "market_total": market_total,
                        "total_edge": total_edge,
                        "model_run_diff": p.get("model_run_diff"),
                        "ml_bet_qualified": True,
                        **_spread_fields,
                    })

            # Away ML value
            if away_ml and abs(ml_edge) >= 0.04:
                market_prob = _market_implied_prob(away_ml)
                if market_prob and away_prob > market_prob:
                    edge_pct = (away_prob - market_prob) * 100
                    bets.append({
                        "matchup": matchup,
                        "bet_type": "Moneyline",
                        "pick": f"{p['away_team']} ML",
                        "odds": f"+{away_ml}" if away_ml > 0 else str(away_ml),
                        "model_prob": away_prob,
                        "market_prob": market_prob,
                        "edge": edge_pct,
                        "edge_type": "ML",
                        "value_score": edge_pct,
                        "confidence": p.get("confidence"),
                        "away_sp": p.get("away_pitcher", "TBD"),
                        "home_sp": p.get("home_pitcher", "TBD"),
                        "away_era": p.get("away_pitcher_era"),
                        "home_era": p.get("home_pitcher_era"),
                        "temp_f": p.get("temp_f"),
                        "wind_mph": p.get("wind_mph"),
                        "wind_out_factor": p.get("wind_out_factor", 0),
                        "model_total": model_total,
                        "market_total": market_total,
                        "total_edge": total_edge,
                        "model_run_diff": p.get("model_run_diff"),
                        "ml_bet_qualified": True,
                        **_spread_fields,
                    })

        # --- Totals bets ---
        # Only Overs — Under signal is 50% (coin flip) based on 46-game sample
        if market_total and model_total and total_edge >= 0.8:
            bets.append({
                "matchup": matchup,
                "bet_type": "Total",
                "pick": f"Over {market_total}",
                "odds": "-110",  # standard juice
                "direction": "OVER",
                "model_total": model_total,
                "market_total": market_total,
                "edge": total_edge,
                "edge_type": "Total",
                "value_score": total_edge * 3,  # scale to compare with ML edge
                "confidence": p.get("confidence"),
                "away_sp": p.get("away_pitcher", "TBD"),
                "home_sp": p.get("home_pitcher", "TBD"),
                "away_era": p.get("away_pitcher_era"),
                "home_era": p.get("home_pitcher_era"),
                "temp_f": p.get("temp_f"),
                "wind_mph": p.get("wind_mph"),
                "wind_out_factor": p.get("wind_out_factor", 0),
                "ml_edge": ml_edge,
                "model_run_diff": p.get("model_run_diff"),
                **_spread_fields,
            })

    # Sort by value score descending, dedupe by matchup+type
    bets.sort(key=lambda x: x["value_score"], reverse=True)
    return bets


def _render_best_bet_card(bet):
    """Render a single best bet card."""
    with st.container(border=True):
        # Header
        st.markdown(f"**{bet['matchup']}**")

        # Pitcher matchup
        away_era = f"{bet['away_era']:.2f}" if bet.get("away_era") else "---"
        home_era = f"{bet['home_era']:.2f}" if bet.get("home_era") else "---"
        st.caption(
            f"{bet.get('away_sp','TBD')} ({away_era}) "
            f"vs {bet.get('home_sp','TBD')} ({home_era})"
        )

        # Weather
        weather = _format_weather(
            bet.get("temp_f"),
            bet.get("wind_mph"),
            bet.get("wind_out_factor", 0),
        )
        if weather != "—":
            st.caption(f"🌡️ {weather}")

        st.divider()

        # Pick + confidence
        conf_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(
            bet.get("confidence", "Low"), "⚪"
        )
        st.markdown(f"**{_best_bet(bet)}**")
        st.caption(f"{conf_emoji} {bet.get('confidence','')}")

        # Reason line
        home_prob = bet.get("model_prob", 0.5)
        market_prob = bet.get("market_prob", 0.5)
        total_edge = bet.get("total_edge", 0) or 0
        run_diff = bet.get("model_run_diff", 0) or 0
        model_total = bet.get("model_total", 0)
        market_total = bet.get("market_total", 0)
        edge_type = bet.get("edge_type", "")

        if edge_type == "ML":
            st.caption(
                f"Market implies {market_prob:.0%} — "
                f"model says {home_prob:.0%} — "
                f"{abs((home_prob - market_prob) * 100):.1f}% edge"
            )
        elif edge_type == "Total":
            st.caption(
                f"Model sees {model_total:.1f} runs "
                f"vs market {market_total} — "
                f"{total_edge:+.1f} edge"
            )


def render_mlb_model_page(conn: sqlite3.Connection) -> None:
    """Render the full MLB Model Picks page."""
    st.header("⚾ MLB Model Picks")
    st.caption("Pitcher-aware lineup model · v4 · Powered by MLB Stats API + The Odds API")
    st.info(
        "Opening Day baseline uses 2025 season-end team form. "
        "Predictions sharpen as 2026 games accumulate."
    )

    # ------------------------------------------------------------------
    # Model Status card
    # ------------------------------------------------------------------
    with st.container(border=True):
        st.subheader("Model Status")
        try:
            from line_tracker.model.mlb_predict import load_mlb_artifacts
            artifacts = load_mlb_artifacts()
            ml_meta = artifacts["moneyline"]["metadata"]
            mg_meta = artifacts["margin"]["metadata"]
            tot_meta = artifacts["totals"]["metadata"]

            trained_at = ml_meta.get("trained_at", "")
            if trained_at:
                try:
                    dt = datetime.fromisoformat(trained_at)
                    trained_str = dt.strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    trained_str = trained_at
            else:
                trained_str = "unknown"

            ml_score = ml_meta.get("cv_score", 0.0)
            mg_score = mg_meta.get("cv_score", 0.0)
            tot_score = tot_meta.get("cv_score", 0.0)

            artifact_name = ml_meta.get("artifact_file", ml_meta.get("artifact", "mlb_models.pkl"))

            st.success(
                f"✅ All 3 models loaded · Trained {trained_str}\n\n"
                f"Moneyline: {ml_score:.1%} acc | "
                f"Margin MAE: {mg_score:.2f} | "
                f"Totals MAE: {tot_score:.2f}\n\n"
                f"Artifact: {artifact_name}"
            )
            if st.button("🔄 Reload Artifacts"):
                for key in list(st.session_state.keys()):
                    if "mlb" in key.lower():
                        del st.session_state[key]
                st.rerun()
        except FileNotFoundError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.warning(f"Could not load model artifacts: {exc}")

    # ------------------------------------------------------------------
    # Train section
    # ------------------------------------------------------------------
    with st.expander("🔧 Train Model", expanded=False):
        sel_seasons = st.multiselect(
            "Seasons",
            [2022, 2023, 2024, 2025],
            default=[2022, 2023, 2024, 2025],
            key="mlb_train_seasons",
        )
        force_refresh = st.checkbox("Force refresh data", key="mlb_force_refresh")
        if is_deployed:
            st.warning(
                "⚠️ Model training must be run locally. "
                "Current deployed model was trained 2026-03-25."
            )
        else:
            if st.button("Train MLB Models", key="mlb_train_btn"):
                with st.spinner("Training MLB models — this takes 3–5 minutes..."):
                    try:
                        from line_tracker.model.mlb_train import train_mlb_models
                        result = train_mlb_models(
                            seasons=sel_seasons,
                            force_refresh=force_refresh,
                        )
                        st.success(
                            f"✅ Training complete!\n"
                            f"Moneyline: {result['moneyline']}\n"
                            f"Margin: {result['margin']}\n"
                            f"Totals: {result['totals']}"
                        )
                    except Exception as exc:
                        st.error(f"Training failed: {exc}")

    # ------------------------------------------------------------------
    # Predictions section
    # ------------------------------------------------------------------
    col1, col2 = st.columns([2, 1])
    with col1:
        pred_date = st.date_input(
            "Prediction date",
            value=date.today(),
            key="mlb_pred_date",
        )

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col1:
        load_btn = st.button("Load Predictions", key="mlb_load_preds")
    with btn_col2:
        refresh_clicked = st.button("🔄 Refresh Odds", key="mlb_refresh_odds")

    save_to_db = st.checkbox("Save to DB", value=False, key="mlb_save_db")

    if load_btn:
        with st.spinner("Fetching odds + building predictions..."):
            try:
                from line_tracker.model.mlb_predict import predict_mlb_games
                preds = predict_mlb_games(target_date=pred_date)

                if save_to_db:
                    from line_tracker.db.repos.mlb_predictions_repo import upsert_prediction
                    for p in preds:
                        try:
                            upsert_prediction(conn, p)
                        except Exception as e:
                            log.warning("Failed to upsert prediction %s: %s", p.get("game_id"), e)

                st.session_state["mlb_predictions"] = preds
                st.session_state["mlb_odds_fetched_at"] = datetime.now()
                if preds:
                    st.success(f"Loaded {len(preds)} games for {pred_date}.")
                else:
                    st.info(f"No MLB games found for {pred_date}.")
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.session_state["mlb_predictions"] = []
            except Exception as exc:
                st.error(f"Error loading predictions: {exc}")
                log.exception("Error in mlb_model_page predictions")
                st.session_state["mlb_predictions"] = []

    if refresh_clicked and st.session_state.get("mlb_predictions"):
        with st.spinner("Refreshing odds..."):
            try:
                from line_tracker.model.mlb_predict import (
                    _fetch_mlb_odds,
                    _american_to_prob,
                    prob_to_american_odds,
                    ODDS_TO_BR,
                )

                # Fetch fresh odds
                fresh_odds = _fetch_mlb_odds()

                # Build lookup: (home_br, away_br) -> odds dict
                odds_lookup = {}
                for g in fresh_odds:
                    home_br = ODDS_TO_BR.get(g.get("home_team", ""), "")
                    away_br = ODDS_TO_BR.get(g.get("away_team", ""), "")
                    if home_br and away_br:
                        odds_lookup[(home_br, away_br)] = g

                # Update each prediction with fresh odds
                updated_preds = []
                for p in st.session_state["mlb_predictions"]:
                    home_br = p.get("home_team_br", "")
                    away_br = p.get("away_team_br", "")
                    fresh = odds_lookup.get((home_br, away_br))

                    if fresh:
                        new_home_ml = fresh.get("market_home_ml") or p.get("market_home_ml")
                        new_away_ml = fresh.get("market_away_ml") or p.get("market_away_ml")
                        new_market_total = fresh.get("market_total") or p.get("market_total")

                        home_prob = p.get("model_home_win_prob", 0.5)
                        away_prob = 1 - home_prob

                        if new_home_ml:
                            market_home_prob = _american_to_prob(new_home_ml)
                            new_ml_edge = home_prob - market_home_prob
                        else:
                            new_ml_edge = p.get("ml_edge", 0)

                        if new_market_total:
                            new_total_edge = p.get("model_total_runs", 0) - new_market_total
                        else:
                            new_total_edge = p.get("total_edge", 0)

                        if abs(new_ml_edge) >= 0.05 or abs(new_total_edge) >= 1.0:
                            new_confidence = "High"
                        elif abs(new_ml_edge) >= 0.03 or abs(new_total_edge) >= 0.6:
                            new_confidence = "Medium"
                        else:
                            new_confidence = "Low"

                        p = {
                            **p,
                            "market_home_ml": new_home_ml,
                            "market_away_ml": new_away_ml,
                            "market_total": new_market_total,
                            "ml_edge": new_ml_edge,
                            "total_edge": new_total_edge,
                            "confidence": new_confidence,
                        }

                    updated_preds.append(p)

                st.session_state["mlb_predictions"] = updated_preds
                st.session_state["mlb_odds_fetched_at"] = datetime.now()
                st.success("✅ Odds refreshed successfully")
                st.rerun()

            except Exception as e:
                st.error(f"Failed to refresh odds: {e}")
    elif refresh_clicked:
        st.warning("Load predictions first before refreshing odds.")

    # ------------------------------------------------------------------
    # Predictions table
    # ------------------------------------------------------------------
    # Auto-expire session state after 30 minutes to match lineup cache TTL
    fetched_at = st.session_state.get("mlb_odds_fetched_at")
    if fetched_at:
        age_minutes = (datetime.now() - fetched_at).seconds // 60
        if age_minutes >= 30:
            del st.session_state["mlb_predictions"]
            del st.session_state["mlb_odds_fetched_at"]

    preds = st.session_state.get("mlb_predictions")
    if preds is None:
        pass  # Haven't loaded yet
    elif not preds:
        st.info(f"No MLB games found for {pred_date}.")
    else:
        fetched_at = st.session_state.get("mlb_odds_fetched_at")
        if fetched_at:
            age_minutes = (datetime.now() - fetched_at).seconds // 60
            if age_minutes < 5:
                st.caption("✅ Odds fetched just now")
            elif age_minutes < 30:
                st.caption(f"⚠️ Odds fetched {age_minutes} min ago — consider refreshing")
            else:
                st.caption(f"🔴 Odds fetched {age_minutes} min ago — likely stale, please refresh")
        # Diagnostic for games where model total matches market total exactly
        for p in preds:
            mt = p.get("market_total")
            pt = p.get("model_total_runs")
            if mt and pt and abs(pt - mt) < 0.1:
                print(f"WARNING: model total matches market exactly for "
                      f"{p['away_team']} @ {p['home_team']}: "
                      f"model={pt}, market={mt}")
            if "Mets" in p.get("home_team", ""):
                print(f"NYM run_diff: {p.get('model_run_diff')}")

        rows = []
        for p in preds:
            home_prob = p.get("model_home_win_prob", 0.5)
            away_prob = 1 - home_prob
            home_ml = p.get("market_home_ml")
            away_ml = p.get("market_away_ml")
            market_spread = p.get("market_spread")

            # Extract team abbreviations from BR codes
            home_br = p.get("home_team_br", "")
            away_br = p.get("away_team_br", "")
            if not home_br:
                home_br = p.get("home_team", "")[-3:].upper()
            if not away_br:
                away_br = p.get("away_team", "")[-3:].upper()

            win_pct = f"{away_br} {away_prob:.1%} / {home_br} {home_prob:.1%}"

            pred_score = _pred_score_abbr(
                p.get("model_total_runs"),
                p.get("model_run_diff"),
                home_prob,
                away_br,
                home_br,
            )

            home_spread_odds = p.get("market_home_spread_odds")
            away_spread_odds = p.get("market_away_spread_odds")

            def _fmt_odds(o):
                if o is None:
                    return ""
                return f" (+{int(o)})" if o > 0 else f" ({int(o)})"

            if market_spread is not None:
                home_pt = float(market_spread)
                away_pt = -home_pt

                def _fmt_pt(pt):
                    if pt > 0:
                        return f"+{pt:.1f}"
                    else:
                        return f"{pt:.1f}"

                if abs(home_pt) < 0.5:
                    spread_str = "Pick'em"
                else:
                    spread_str = (
                        f"Away {_fmt_pt(away_pt)}{_fmt_odds(away_spread_odds)} / "
                        f"Home {_fmt_pt(home_pt)}{_fmt_odds(home_spread_odds)}"
                    )
            else:
                spread_str = "Away +1.5 / Home -1.5" if p.get("market_total") else "—"

            # Model agreement indicator
            disagreement = p.get("model_disagreement")
            is_flagged = p.get("flagged", False)
            if disagreement is None:
                agreement_str = "—"
            elif disagreement >= 0.20:
                agreement_str = f"🔴 {disagreement:.3f}"
            elif disagreement >= 0.10:
                agreement_str = f"🟡 {disagreement:.3f}"
            else:
                agreement_str = f"🟢 {disagreement:.3f}"

            ensemble_prob = p.get("ensemble_prob")
            if ensemble_prob is not None:
                if ensemble_prob >= 0.5:
                    ensemble_str = f"{p.get('home_team_br','HOM')} {ensemble_prob:.1%}"
                else:
                    ensemble_str = f"{p.get('away_team_br','AWY')} {1-ensemble_prob:.1%}"
            else:
                ensemble_str = "—"

            rows.append({
                "Matchup":       f"{p['away_team']} @ {p['home_team']}",
                "Time":          _format_time(p.get("commence_time")),
                "Away SP":       _format_sp(p.get("away_pitcher"), p.get("away_pitcher_era")),
                "Home SP":       _format_sp(p.get("home_pitcher"), p.get("home_pitcher_era")),
                "Weather":       _format_weather(
                                     p.get("temp_f"),
                                     p.get("wind_mph"),
                                     p.get("wind_out_factor", 0),
                                 ),
                "Win%":          win_pct,
                "Ensemble":      ensemble_str,
                "Agreement":     agreement_str,
                "Pred Score":    pred_score,
                "Pred Total":    f"{p.get('model_total_runs', 0):.1f}",
                "Mkt Total":     str(p.get("market_total", "—")),
                "Total Edge":    f"{p.get('total_edge', 0):+.1f}" if p.get("market_total") else "—",
                "Away ML":       fmt_ml(away_ml),
                "Home ML":       fmt_ml(home_ml),
                "Mkt Spread":    spread_str,
                "Model Spread":  _model_spread_pick(p.get("model_run_diff"), market_spread),
                "Best Bet":      (
                    "⚠️ DATA CHECK — verify data before betting"
                    if is_flagged
                    else _best_bet(p)
                ),
                "Confidence":    (
                    f"{p.get('confidence', '—')} ⚠️"
                    if p.get("blind_spot", False)
                    else p.get("confidence", "—")
                ),
            })

        # NRFI/YRFI enrichment — best-effort (silently skipped if model not trained)
        nrfi_lookup: dict[str, dict] = {}
        try:
            from line_tracker.model.mlb_nrfi import predict_nrfi
            nrfi_preds = predict_nrfi(game_date=pred_date)
            for np_ in nrfi_preds:
                key = (np_["away_team"], np_["home_team"])
                nrfi_lookup[key] = np_
        except Exception:
            pass

        if nrfi_lookup:
            for row, p in zip(rows, preds):
                away_br = p.get("away_team_br", "") or p.get("away_team", "")[-3:].upper()
                home_br = p.get("home_team_br", "") or p.get("home_team", "")[-3:].upper()
                np_ = nrfi_lookup.get((away_br, home_br)) or {}
                yrfi_prob = np_.get("yrfi_prob")
                row["YRFI%"] = f"{yrfi_prob:.1%}" if yrfi_prob else "—"
                bet = np_.get("bet")
                edge = np_.get("edge")
                row["NRFI/YRFI Bet"] = bet or "—"
                row["NRFI/YRFI Edge"] = f"{edge*100:+.1f}%" if edge else "—"

        df_display = pd.DataFrame(rows)
        st.dataframe(
            df_display,
            width='stretch',
            hide_index=True,
            column_config={
                "Win%": st.column_config.TextColumn(
                    "Win%",
                    help="Away team win probability / Home team win probability"
                ),
                "Ensemble": st.column_config.TextColumn(
                    "Ensemble",
                    help="GBM meta-model home win probability (stacks all three base models)"
                ),
                "Agreement": st.column_config.TextColumn(
                    "Agreement",
                    help=(
                        "Model agreement: 🟢 < 0.10 (models aligned) · "
                        "🟡 0.10–0.20 (some conflict — use with caution) · "
                        "🔴 >= 0.20 (extreme disagreement — likely data issue)"
                    ),
                ),
                "Total Edge": st.column_config.TextColumn(
                    "Total Edge",
                    help="Model predicted total minus market total. Positive = lean over."
                ),
                "Best Bet": st.column_config.TextColumn(
                    "Best Bet",
                    help="Best betting opportunity. ⚠️ DATA CHECK = extreme model disagreement, verify data first."
                ),
                "YRFI%": st.column_config.TextColumn(
                    "YRFI%",
                    help="Probability that at least one run scores in the first inning (NRFI/YRFI model)."
                ),
                "NRFI/YRFI Bet": st.column_config.TextColumn(
                    "NRFI/YRFI Bet",
                    help="Recommended NRFI or YRFI bet when model edge exceeds 5%."
                ),
                "NRFI/YRFI Edge": st.column_config.TextColumn(
                    "NRFI/YRFI Edge",
                    help="Model probability minus market implied probability for the recommended NRFI/YRFI bet."
                ),
            }
        )
        st.caption(
            "⚠️ = Known blind spot matchup (model historically less accurate). "
            "Agreement: 🟢 < 0.10 (aligned) · 🟡 0.10–0.20 (caution) · 🔴 >= 0.20 (data check). "
            "Based on 46-game sample — revisit at 100 games."
        )

        # ------------------------------------------------------------------
        # Best Bets section
        # ------------------------------------------------------------------
        st.subheader("🎯 Best Bets")
        st.caption(
            "📊 Filters applied based on 46-game sample: "
            "ML picks require 57%+ model confidence · "
            "Totals: Overs only (64% hit rate) · "
            "Unders skipped (50% hit rate)"
        )
        fetched_at = st.session_state.get("mlb_odds_fetched_at")
        if fetched_at:
            age_minutes = (datetime.now() - fetched_at).seconds // 60
            if age_minutes >= 15:
                st.warning(
                    f"⚠️ Odds are {age_minutes} min old. "
                    "Click '🔄 Refresh Odds' before placing bets."
                )
        bets = _find_best_bets(st.session_state.get("mlb_predictions", []))
        if not bets:
            st.info("No high-confidence bets found for today.")
        else:
            # Show top 6 bets max, 3 per row
            top_bets = bets[:6]
            for i in range(0, len(top_bets), 3):
                cols = st.columns(3)
                for j, bet in enumerate(top_bets[i:i + 3]):
                    with cols[j]:
                        _render_best_bet_card(bet)

            # Summary line
            ml_bets = [b for b in top_bets if b["edge_type"] == "ML"]
            total_bets = [b for b in top_bets if b["edge_type"] == "Total"]
            st.caption(
                f"Showing top {len(top_bets)} bets: "
                f"{len(ml_bets)} moneyline, {len(total_bets)} totals"
            )

    # ------------------------------------------------------------------
    # Historical Accuracy section
    # ------------------------------------------------------------------
    st.subheader("📊 Model Accuracy")
    try:
        from line_tracker.db.repos.mlb_predictions_repo import get_model_accuracy
        acc = get_model_accuracy(conn)
        n = acc["n_settled"]
        if n >= 10:
            c1, c2, c3 = st.columns(3)
            c1.metric("Moneyline", f"{acc['moneyline_acc']:.1%}")
            c2.metric("Totals", f"{acc['total_acc']:.1%}")
            c3.metric("Games Settled", n)
        else:
            st.info(f"Need 10 settled games to show accuracy. Currently: {n}/10")
    except Exception as exc:
        st.info(f"No accuracy data yet: {exc}")
