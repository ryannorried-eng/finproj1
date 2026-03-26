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


def _model_spread_pick(model_run_diff):
    """
    Determine model's run line pick based on predicted margin.
    model_run_diff is always from the HOME team's perspective:
      positive = home team wins by that many runs
      negative = away team wins by that many runs

    Returns string like "Home -1.5 (+2.2)" or "Away +1.5 (-2.2)" or "—"
    """
    if model_run_diff is None or pd.isna(model_run_diff):
        return "—"

    if model_run_diff > 1.5:
        # Home wins by more than 1.5 → Home covers -1.5
        return f"Home -1.5 ({model_run_diff:+.1f})"
    elif model_run_diff < -1.5:
        # Away wins by more than 1.5 → Away covers +1.5
        return f"Away +1.5 ({model_run_diff:+.1f})"
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

        # --- Moneyline bets ---
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
                })

        # --- Totals bets ---
        # Only show totals with meaningful edge (>= 0.8 runs)
        if market_total and model_total and abs(total_edge) >= 0.8:
            if total_edge > 0:
                pick = f"Over {market_total}"
                direction = "OVER"
            else:
                pick = f"Under {market_total}"
                direction = "UNDER"

            bets.append({
                "matchup": matchup,
                "bet_type": "Total",
                "pick": pick,
                "odds": "-110",  # standard juice
                "direction": direction,
                "model_total": model_total,
                "market_total": market_total,
                "edge": abs(total_edge),
                "edge_type": "Total",
                "value_score": abs(total_edge) * 3,  # scale to compare with ML edge
                "confidence": p.get("confidence"),
                "away_sp": p.get("away_pitcher", "TBD"),
                "home_sp": p.get("home_pitcher", "TBD"),
                "away_era": p.get("away_pitcher_era"),
                "home_era": p.get("home_pitcher_era"),
                "temp_f": p.get("temp_f"),
                "wind_mph": p.get("wind_mph"),
                "wind_out_factor": p.get("wind_out_factor", 0),
                "ml_edge": ml_edge,
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

        # Pick + odds
        conf_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(
            bet.get("confidence", "Low"), "⚪"
        )
        st.markdown(
            f"**{bet['pick']}** &nbsp; `{bet['odds']}` &nbsp; "
            f"{conf_emoji} {bet.get('confidence','')}"
        )

        spread_pick = _model_spread_pick(bet.get("model_run_diff"))
        if spread_pick != "—" and "Push" not in spread_pick:
            st.caption(f"Run line: {spread_pick}")

        # Edge display
        if bet["edge_type"] == "ML":
            edge_color = "green" if bet["edge"] > 0 else "red"
            st.markdown(
                f"ML Edge: :{edge_color}[**{bet['edge']:+.1f}%**] &nbsp;|&nbsp; "
                f"Model: {bet['model_prob']:.1%} vs Mkt: {bet['market_prob']:.1%}"
            )
        else:
            direction = bet.get("direction", "")
            edge_color = "green" if bet["edge"] > 0 else "red"
            st.markdown(
                f"Total Edge: :{edge_color}[**{bet['edge']:+.1f} runs**] &nbsp;|&nbsp; "
                f"Model: {bet['model_total']:.1f} vs Mkt: {bet['market_total']}"
            )

        # Warning for bad juice situations
        if bet["bet_type"] == "Moneyline":
            odds_val = int(bet["odds"].replace("+", ""))
            if odds_val < -300:
                st.warning(
                    f"⚠️ Heavy juice ({bet['odds']}) — "
                    "need high confidence to find value here"
                )
            elif odds_val > 200:
                st.info(
                    f"💰 Big dog ({bet['odds']}) — "
                    "small edge goes a long way at this price"
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
    with col2:
        load_btn = st.button("Load Predictions", key="mlb_load_preds")

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

    # ------------------------------------------------------------------
    # Predictions table
    # ------------------------------------------------------------------
    preds = st.session_state.get("mlb_predictions")
    if preds is None:
        pass  # Haven't loaded yet
    elif not preds:
        st.info(f"No MLB games found for {pred_date}.")
    else:
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

        tz_name = _display_tz()
        rows = []
        for p in preds:
            home_prob = p.get("model_home_win_prob", 0.5)
            away_prob = 1 - home_prob
            home_ml = p.get("market_home_ml")
            away_ml = p.get("market_away_ml")
            market_spread = p.get("market_spread")

            # Fix 2: Show both sides in Win% column
            win_pct = f"{home_prob:.1%} / {away_prob:.1%}"

            # Show the side with positive edge (model prob > market prob)
            home_val = _value_score(home_prob, home_ml)
            away_val = _value_score(away_prob, away_ml)
            if home_val is not None and home_val > 0:
                value_str = f"+{home_val:.1f} (H)"
            elif away_val is not None and away_val > 0:
                value_str = f"+{away_val:.1f} (A)"
            elif home_val is not None and away_val is not None:
                # Neither side underpriced — show least negative
                best = max(home_val, away_val)
                side = "H" if home_val >= away_val else "A"
                value_str = f"{best:+.1f} ({side})"
            elif home_val is not None:
                value_str = f"{home_val:+.1f} (H)"
            elif away_val is not None:
                value_str = f"{away_val:+.1f} (A)"
            else:
                value_str = "—"

            # Fix 4: Show both sides of spread
            if market_spread is not None:
                if market_spread < 0:
                    spread_str = f"{market_spread:.1f} / +{abs(market_spread):.1f}"
                else:
                    spread_str = f"+{market_spread:.1f} / -{market_spread:.1f}"
            else:
                spread_str = "-1.5 / +1.5" if p.get("market_total") else "—"

            rows.append({
                "Matchup":       f"{p['away_team']} @ {p['home_team']}",
                "Time":          _fmt_commence(p.get("commence_time", ""), tz_name),
                "Away SP":       _format_sp(p.get("away_pitcher"), p.get("away_pitcher_era")),
                "Home SP":       _format_sp(p.get("home_pitcher"), p.get("home_pitcher_era")),
                "Weather":       _format_weather(
                                     p.get("temp_f"),
                                     p.get("wind_mph"),
                                     p.get("wind_out_factor", 0),
                                 ),
                "Home Win%":     win_pct,
                "Home Model ML": _format_ml_odds(home_prob),
                "Away ML":       f"+{away_ml}" if away_ml and away_ml > 0 else str(away_ml) if away_ml else "—",
                "Home ML":       f"+{home_ml}" if home_ml and home_ml > 0 else str(home_ml) if home_ml else "—",
                "Pred Total":    f"{p.get('model_total_runs', 0):.1f}",
                "Pred Score":    _pred_score_with_flag(
                                     p.get("model_total_runs"),
                                     p.get("model_run_diff"),
                                     p.get("model_home_win_prob", 0.5),
                                 ),
                "Mkt Total":     str(p.get("market_total", "—")),
                "Total Edge":    f"{p.get('total_edge', 0):+.1f}" if p.get("market_total") else "—",
                "Mkt Spread":    spread_str,
                "Model Spread":  _model_spread_pick(p.get("model_run_diff")),
                "ML Edge":       f"{p.get('ml_edge', 0):+.1%}",
                "Value":         value_str,
                "Confidence":    p.get("confidence", "—"),
            })

        df_display = pd.DataFrame(rows)
        st.dataframe(
            df_display,
            width='stretch',
            hide_index=True,
            column_config={
                "Home Win%": st.column_config.TextColumn(
                    "Home Win%",
                    help="Home team win probability / Away team win probability"
                ),
                "Value": st.column_config.TextColumn(
                    "Value",
                    help="Edge quality: (model prob - market prob) × 100. H=home edge, A=away edge"
                ),
                "Total Edge": st.column_config.TextColumn(
                    "Total Edge",
                    help="Model predicted total minus market total. Positive = lean over."
                ),
                "ML Edge": st.column_config.TextColumn(
                    "ML Edge",
                    help="Model win prob minus market implied prob (home team perspective)"
                ),
            }
        )
        st.caption("Win% and Model ML reflect the HOME team's probability and implied odds.")

        # ------------------------------------------------------------------
        # Best Bets section
        # ------------------------------------------------------------------
        st.subheader("🎯 Best Bets")
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
