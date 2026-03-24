"""MLB Model Picks page for the Streamlit dashboard.

Team-form model v1 — powered by pybaseball + The Odds API.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

log = logging.getLogger(__name__)


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


def render_mlb_model_page(conn: sqlite3.Connection) -> None:
    """Render the full MLB Model Picks page."""
    st.header("⚾ MLB Model Picks")
    st.caption("Team-form model · v1 · Powered by pybaseball + The Odds API")
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

            st.success(
                f"✅ All 3 models loaded · Trained {trained_str}\n\n"
                f"Moneyline: {ml_score:.1%} acc | "
                f"Margin MAE: {mg_score:.2f} | "
                f"Totals MAE: {tot_score:.2f}"
            )
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
        tz_name = _display_tz()
        rows = []
        for p in preds:
            mt = p.get("market_total")
            te = p.get("total_edge")
            me = p.get("ml_edge")
            rows.append({
                "Matchup": f"{p['away_team']} @ {p['home_team']}",
                "Time": _fmt_commence(p.get("commence_time", ""), tz_name),
                "Win Prob": f"{p['model_home_win_prob']:.1%}",
                "Pred Margin": f"{p['model_run_diff']:+.1f}",
                "Pred Total": f"{p['model_total_runs']:.1f}",
                "Mkt Total": f"{mt}" if mt is not None else "—",
                "Total Edge": f"{te:+.1f}" if te is not None and mt is not None else "—",
                "ML Edge": f"{me:+.1%}" if me is not None and p.get("market_home_ml") else "—",
                "Pick": p.get("model_spread_pick", ""),
                "Confidence": p.get("confidence", ""),
                "Source": p.get("data_source", ""),
            })

        display_df = pd.DataFrame(rows)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ------------------------------------------------------------------
        # Best Bets section
        # ------------------------------------------------------------------
        st.subheader("🎯 Best Bets")
        best = [
            p for p in preds
            if abs(p.get("ml_edge") or 0) >= 0.05
            or abs(p.get("total_edge") or 0) >= 1.0
        ]

        if not best:
            st.info("No high-confidence bets for today.")
        else:
            cols = st.columns(min(3, len(best)))
            for i, p in enumerate(best):
                col = cols[i % 3]
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{p['away_team']} @ {p['home_team']}**")
                        st.markdown(f"Pick: **{p.get('model_spread_pick', '—')}**")
                        st.caption(p.get("confidence", ""))
                        me = p.get("ml_edge") or 0
                        te = p.get("total_edge") or 0
                        if abs(me) >= abs(te):
                            st.metric("ML Edge", f"{me:+.1%}")
                        else:
                            st.metric("Total Edge", f"{te:+.1f}")
                        mkt_home = p.get("market_home_ml")
                        mkt_away = p.get("market_away_ml")
                        if mkt_home and mkt_away:
                            from line_tracker.model.mlb_predict import _american_to_prob
                            implied_home = _american_to_prob(mkt_home)
                            implied_away = _american_to_prob(mkt_away)
                            denom = implied_home + implied_away
                            devig_home = implied_home / denom if denom else 0.5
                            st.caption(
                                f"Model: {p['model_home_win_prob']:.1%} | "
                                f"Market implied: {devig_home:.1%}"
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
