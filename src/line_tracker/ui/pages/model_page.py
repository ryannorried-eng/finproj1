"""Model Picks page for the Streamlit dashboard.

Displays NCAAB model predictions, top picks, and model metadata.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from line_tracker.config import data_mode
from line_tracker.model.matching import TEAM_NAME_MAP, normalize_team_name
from line_tracker.model.predict import margin_to_spread_prob
from line_tracker.models import BetType
from line_tracker.storage import LineStore

log = logging.getLogger(__name__)

# Minimum model edge (in %) to highlight a row as actionable.
MODEL_MIN_EDGE = 1.5


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_predictions(store: LineStore, game_date: str) -> list[dict]:
    """Load cached predictions for a date from the DB."""
    return store.predictions_repo.get_predictions_for_date(game_date)


def _resolve_to_torvik(odds_name: str, pred_teams: set[str]) -> str:
    """Resolve an Odds API team name to its Torvik equivalent.

    Uses TEAM_NAME_MAP first, then falls back to mascot-stripped
    normalized matching against known prediction team names.
    Returns the original name unchanged if no match is found.
    """
    # Tier 1: curated map
    torvik = TEAM_NAME_MAP.get(odds_name.lower().strip())
    if torvik:
        return torvik

    # Tier 2: normalize (strip mascot) and match against pred teams
    norm = normalize_team_name(odds_name)
    for t in pred_teams:
        if norm == t.lower() or norm == normalize_team_name(t):
            return t

    return odds_name


def _load_spread_data(
    store: LineStore, sport: str, pred_teams: set[str],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Load latest spreads and best odds, keyed by Torvik team names.

    Returns ``(market_spreads, best_odds)`` dicts both keyed by
    ``"torvik_home|torvik_away"``.
    """
    all_lines = store.get_latest_for_sport(sport=sport)
    lines = [ln for ln in all_lines if ln.bet_type == BetType.SPREAD]

    spreads: dict[str, dict] = {}
    best: dict[str, dict] = {}

    for line in lines:
        home = _resolve_to_torvik(line.home_team, pred_teams)
        away = _resolve_to_torvik(line.away_team, pred_teams)
        key = f"{home}|{away}"

        # --- best spread (lowest vig) ---
        existing = spreads.get(key)
        hp = line.home_price if line.home_price is not None else -110
        if (
            existing is None
            or abs(hp) < abs(existing.get("home_price", -110) or -110)
        ):
            spreads[key] = {
                "spread": line.home_value,
                "home_price": line.home_price,
                "away_price": line.away_price,
                "sportsbook": line.sportsbook,
            }

        # --- best available odds per side ---
        entry = best.setdefault(key, {
            "best_home_price": None,
            "best_away_price": None,
            "best_home_book": "",
            "best_away_book": "",
        })
        if line.home_price is not None and (
            entry["best_home_price"] is None
            or line.home_price > entry["best_home_price"]
        ):
            entry["best_home_price"] = line.home_price
            entry["best_home_book"] = line.sportsbook
        if line.away_price is not None and (
            entry["best_away_price"] is None
            or line.away_price > entry["best_away_price"]
        ):
            entry["best_away_price"] = line.away_price
            entry["best_away_book"] = line.sportsbook

    return spreads, best


def _load_model_metadata() -> dict | None:
    """Load metadata from the most recent trained model file."""
    models_dir = Path.home() / ".line_tracker" / "models"
    if not models_dir.exists():
        return None
    candidates = sorted(models_dir.glob("ncaab_margin_*.joblib"))
    if not candidates:
        return None
    try:
        import joblib

        data = joblib.load(candidates[-1])
        meta = data.get("metadata", {})
        meta["model_path"] = str(candidates[-1])
        return meta
    except Exception:
        log.exception("Failed to load model metadata")
        return None


def _american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    if odds >= 100:
        return odds / 100.0 + 1.0
    if odds <= -100:
        return 100.0 / abs(odds) + 1.0
    return 2.0


def _format_american(odds: float | None) -> str:
    if odds is None:
        return "-"
    return f"+{int(odds)}" if odds > 0 else str(int(odds))


# ---------------------------------------------------------------------------
# Build predictions table DataFrame
# ---------------------------------------------------------------------------

def _build_predictions_df(
    predictions: list[dict],
    market_spreads: dict[str, dict],
    best_odds: dict[str, dict],
) -> pd.DataFrame:
    """Build a DataFrame for the predictions table."""
    rows = []
    for pred in predictions:
        home = pred["home_team"]
        away = pred["away_team"]
        margin = pred.get("predicted_margin", 0.0)
        home_ml = pred.get("home_ml_prob", 0.0)
        away_ml = pred.get("away_ml_prob", 0.0)

        key = f"{home}|{away}"
        spread_info = market_spreads.get(key, {})
        odds_info = best_odds.get(key, {})

        market_spread = spread_info.get("spread")

        # Model edge: predicted_margin + market_spread (positive = model
        # likes home cover).  We report absolute edge since the table shows
        # which side to take.
        if market_spread is not None:
            model_edge = margin + market_spread
        else:
            model_edge = None

        # Model probability that favoured side covers
        if market_spread is not None:
            home_cover_prob = margin_to_spread_prob(margin, market_spread)
        else:
            home_cover_prob = None

        # Determine which side the model favours for display
        if margin > 0:
            predicted_label = f"{home} -{abs(margin):.1f}"
        else:
            predicted_label = f"{away} -{abs(margin):.1f}"

        # Market spread label
        if market_spread is not None:
            if market_spread < 0:
                mkt_label = f"{home} {market_spread:+.1f}"
            elif market_spread > 0:
                mkt_label = f"{away} {-market_spread:+.1f}"
            else:
                mkt_label = "PK"
        else:
            mkt_label = "-"

        # Best odds info
        best_home_price = odds_info.get("best_home_price")
        best_away_price = odds_info.get("best_away_price")
        best_home_book = odds_info.get("best_home_book", "")
        best_away_book = odds_info.get("best_away_book", "")

        # Pick the best odds on the side the model favours
        if model_edge is not None and model_edge > 0:
            best_price = best_home_price
            best_book = best_home_book
        elif model_edge is not None and model_edge < 0:
            best_price = best_away_price
            best_book = best_away_book
        else:
            best_price = best_home_price
            best_book = best_home_book

        rows.append({
            "Matchup": f"{away} @ {home}",
            "Model Predicted Margin": predicted_label,
            "Market Spread": mkt_label,
            "Model Edge (pts)": (
                round(abs(model_edge), 1) if model_edge is not None else None
            ),
            "Edge Side": (
                home if (model_edge or 0) > 0 else away
            ) if model_edge is not None else "-",
            "Home Win %": f"{home_ml:.1%}",
            "Away Win %": f"{away_ml:.1%}",
            "Home Cover %": (
                f"{home_cover_prob:.1%}" if home_cover_prob is not None else "-"
            ),
            "Best Odds": _format_american(best_price),
            "Book": best_book,
            "_abs_edge": abs(model_edge) if model_edge is not None else 0.0,
            "_edge_val": model_edge,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("_abs_edge", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Top Picks card
# ---------------------------------------------------------------------------

def _render_top_picks(
    predictions: list[dict],
    market_spreads: dict[str, dict],
    best_odds: dict[str, dict],
) -> None:
    """Render top model picks card at the top of the page."""
    picks = []
    for pred in predictions:
        home = pred["home_team"]
        away = pred["away_team"]
        margin = pred.get("predicted_margin", 0.0)
        key = f"{home}|{away}"
        spread_info = market_spreads.get(key, {})
        odds_info = best_odds.get(key, {})

        market_spread = spread_info.get("spread")
        if market_spread is None:
            continue

        edge = margin + market_spread
        abs_edge = abs(edge)

        if abs_edge < MODEL_MIN_EDGE:
            continue

        # Determine side
        if edge > 0:
            selection = f"{home} (cover)"
            model_prob = margin_to_spread_prob(margin, market_spread)
            best_price = odds_info.get("best_home_price")
            best_book = odds_info.get("best_home_book", "")
        else:
            selection = f"{away} (cover)"
            model_prob = 1.0 - margin_to_spread_prob(margin, market_spread)
            best_price = odds_info.get("best_away_price")
            best_book = odds_info.get("best_away_book", "")

        # Market implied probability from best odds
        if best_price is not None:
            dec = _american_to_decimal(best_price)
            market_prob = 1.0 / dec if dec > 0 else 0.5
        else:
            market_prob = 0.5

        edge_pct = (model_prob - market_prob) * 100.0

        # Simple quality score: edge * confidence
        confidence = pred.get("model_confidence") or 0.5
        quality = abs_edge * confidence

        picks.append({
            "matchup": f"{away} @ {home}",
            "market": "spread",
            "selection": selection,
            "odds": _format_american(best_price),
            "book": best_book,
            "model_edge_pct": edge_pct,
            "model_prob": model_prob,
            "market_prob": market_prob,
            "edge_pts": abs_edge,
            "quality": quality,
        })

    # Sort by edge descending
    picks.sort(key=lambda p: p["edge_pts"], reverse=True)

    if not picks:
        st.info("No picks meet the minimum edge threshold today.")
        return

    st.markdown("### Top Model Picks")

    for i, pick in enumerate(picks[:5]):
        with st.container():
            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
            with c1:
                st.markdown(f"**{pick['matchup']}**")
                st.caption(f"{pick['market'].upper()} | {pick['selection']}")
            with c2:
                st.metric("Model Edge", f"{pick['edge_pts']:.1f} pts")
            with c3:
                st.metric(
                    "Model vs Market",
                    f"{pick['model_prob']:.1%}",
                    delta=f"{pick['model_edge_pct']:+.1f}%",
                )
            with c4:
                st.metric("Best Odds", f"{pick['odds']} ({pick['book']})")
            if i < len(picks[:5]) - 1:
                st.divider()


# ---------------------------------------------------------------------------
# Model info sidebar
# ---------------------------------------------------------------------------

def _render_model_sidebar() -> None:
    """Render model metadata in the sidebar."""
    meta = _load_model_metadata()
    if meta is None:
        st.sidebar.info("No trained model found.")
        return

    st.sidebar.markdown("### Model Info")

    model_path = meta.get("model_path", "")
    model_name = Path(model_path).stem if model_path else "unknown"
    st.sidebar.text(f"Version: {model_name}")

    trained_at = meta.get("trained_at", "")
    if trained_at:
        try:
            dt = datetime.fromisoformat(trained_at)
            st.sidebar.text(f"Trained: {dt.strftime('%Y-%m-%d %H:%M UTC')}")
        except (ValueError, TypeError):
            st.sidebar.text(f"Trained: {trained_at}")

    n_train = meta.get("n_train", 0)
    n_val = meta.get("n_val", 0)
    st.sidebar.metric("Training games", n_train)
    st.sidebar.metric("Validation games", n_val)

    val_rmse = meta.get("val_rmse")
    if val_rmse is not None:
        st.sidebar.metric("Val RMSE", f"{val_rmse:.3f}")

    # Top 5 features
    importances = meta.get("feature_importances", {})
    if importances:
        st.sidebar.markdown("**Top 5 Features**")
        top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        for name, imp in top5:
            st.sidebar.text(f"  {name}: {imp:.3f}")


# ---------------------------------------------------------------------------
# Refresh buttons
# ---------------------------------------------------------------------------

def _render_refresh_buttons(
    store: LineStore, sport: str, api_key: str | None,
) -> None:
    """Render action buttons for odds, predictions, and cycle."""
    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Fetch Latest Odds", key="model_fetch_odds"):
            if not api_key:
                st.error("No API key configured.")
            else:
                with st.spinner("Fetching odds..."):
                    from line_tracker.services.ingestion_service import (
                        fetch_and_persist_snapshot,
                    )

                    result = fetch_and_persist_snapshot(store, api_key, sport=sport)
                    saved = result.get("saved_count", 0)
                st.success(f"Fetched and saved {saved} lines.")
                st.rerun()

    with c2:
        if st.button("Generate Predictions", key="model_gen_preds"):
            with st.spinner("Generating predictions..."):
                try:
                    _run_predictions(store)
                    st.success("Predictions generated and saved.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

    with c3:
        if st.button("Run Full Cycle", key="model_run_cycle"):
            with st.spinner("Running cycle with model predictions..."):
                try:
                    from line_tracker.services.automation_service import run_cycle

                    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    preds = store.predictions_repo.get_predictions_for_date(today)
                    result = run_cycle(
                        store,
                        sport=sport,
                        model_predictions=preds or None,
                    )
                    st.success(
                        f"Cycle complete: {result['slate_entries']} entries, "
                        f"{len(result['top_picks'])} picks"
                    )
                    st.session_state["last_cycle_result"] = {
                        "picks_generated": len(result["top_picks"]),
                        "events_processed": result["slate_entries"],
                    }
                    st.rerun()
                except Exception as exc:
                    st.error(f"Cycle failed: {exc}")


def _run_predictions(store: LineStore) -> None:
    """Generate predictions for today's games and save to DB."""
    from line_tracker.model.data import fetch_schedule
    from line_tracker.model.predict import predict_games

    schedule = fetch_schedule()
    if schedule.empty:
        raise RuntimeError("No upcoming NCAAB games found.")

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    games = []
    for _, row in schedule.iterrows():
        games.append({
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "neutral_site": row.get("neutral_site", False),
            "game_date": today_str,
        })

    predictions = predict_games(games)
    if not predictions:
        raise RuntimeError("No predictions generated (team matching issues?).")

    for p in predictions:
        p["sport"] = "basketball_ncaab"
    with store.transaction():
        store.predictions_repo.save_predictions(predictions)


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------

def render_model_picks_page(db_path: str, sport_key: str, api_key: str | None) -> None:
    """Render the full Model Picks page."""
    st.header("Model Picks")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Sidebar model info
    _render_model_sidebar()

    if data_mode() != "db":
        st.warning("Model Picks requires DB mode. Switch to DB mode to use this page.")
        return

    try:
        with LineStore(db_path) as store:
            predictions = _load_predictions(store, today)

            # Collect Torvik team names from predictions so the
            # market-data loader can resolve Odds API names.
            pred_teams: set[str] = set()
            for p in predictions:
                pred_teams.add(p["home_team"])
                pred_teams.add(p["away_team"])

            market_spreads, best_odds = _load_spread_data(
                store, sport_key, pred_teams,
            )

            # Summary metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Games with Predictions", len(predictions))
            c2.metric("Games with Market Lines", len(market_spreads))

            # Count picks above threshold
            above_threshold = 0
            for pred in predictions:
                key = f"{pred['home_team']}|{pred['away_team']}"
                spread_info = market_spreads.get(key, {})
                ms = spread_info.get("spread")
                if ms is not None:
                    edge = abs(pred.get("predicted_margin", 0) + ms)
                    if edge >= MODEL_MIN_EDGE:
                        above_threshold += 1
            c3.metric(
                f"Picks (edge >= {MODEL_MIN_EDGE})",
                above_threshold,
            )

            if not predictions:
                st.info(
                    "No predictions for today. Click **Generate Predictions** below "
                    "to run the model."
                )
            else:
                # Top Picks card
                _render_top_picks(predictions, market_spreads, best_odds)

                st.markdown("---")

                # Predictions table
                st.subheader("All Predictions")
                df = _build_predictions_df(predictions, market_spreads, best_odds)

                if not df.empty:
                    # Style: highlight rows with edge >= threshold
                    display_df = df.drop(columns=["_abs_edge", "_edge_val"])

                    def _highlight_edge(row):
                        edge = df.loc[row.name, "_abs_edge"]
                        if edge >= MODEL_MIN_EDGE:
                            bg = "background-color: rgba(0,200,0,0.15)"
                            return [bg] * len(row)
                        return [""] * len(row)

                    styled = display_df.style.apply(_highlight_edge, axis=1)
                    st.dataframe(
                        styled,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No prediction data to display.")

            # Refresh buttons
            _render_refresh_buttons(store, sport_key, api_key)

    except Exception as exc:
        st.error(f"Error loading model data: {exc}")
        log.exception("Error in model picks page")
