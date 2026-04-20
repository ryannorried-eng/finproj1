"""NRFI/YRFI Streamlit page — first-inning props with manual odds input."""

from __future__ import annotations

import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

log = logging.getLogger(__name__)


@st.cache_data(ttl=3600)
def _cached_fi_history(current_year: int):
    """Load first-inning history once per hour; shared across all button clicks."""
    try:
        from line_tracker.model.mlb_data import fetch_first_inning_data
        df = fetch_first_inning_data(
            seasons=[current_year - 1, current_year],
            force_refresh=False,
        )
        return df if df is not None and not df.empty else None
    except Exception:
        return None


def _fmt_commence(ct_str: str, tz_name: str = "America/Chicago") -> str:
    if not ct_str:
        return ""
    try:
        dt = datetime.fromisoformat(ct_str.replace("Z", "+00:00"))
        local = dt.astimezone(ZoneInfo(tz_name))
        suffix = tz_name.split("/")[-1][:2].upper() + "T"
        return local.strftime(f"%-I:%M %p {suffix}")
    except Exception:
        return ct_str


def _american_to_implied(odds: int) -> float | None:
    """Convert American odds integer to raw implied probability (with vig)."""
    if odds == 0:
        return None
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def _calc_edge(yrfi_prob: float, nrfi_prob: float, yrfi_odds: int, nrfi_odds: int) -> dict | None:
    """Devig and calculate edges for a single game. Returns None if odds are 0."""
    yrfi_implied = _american_to_implied(yrfi_odds)
    nrfi_implied = _american_to_implied(nrfi_odds)
    if yrfi_implied is None or nrfi_implied is None:
        return None

    total = yrfi_implied + nrfi_implied
    if total <= 0:
        return None

    market_yrfi = yrfi_implied / total
    market_nrfi = nrfi_implied / total

    yrfi_edge = yrfi_prob - market_yrfi
    nrfi_edge = nrfi_prob - market_nrfi

    from line_tracker.model.mlb_nrfi import EDGE_THRESHOLD
    if yrfi_edge >= EDGE_THRESHOLD:
        bet, edge = "YRFI", yrfi_edge
    elif nrfi_edge >= EDGE_THRESHOLD:
        bet, edge = "NRFI", nrfi_edge
    else:
        bet, edge = None, None

    confidence = None
    if edge is not None:
        if edge >= 0.08:
            confidence = "High"
        elif edge >= 0.05:
            confidence = "Medium"
        else:
            confidence = "Low"

    return {
        "market_yrfi_prob": market_yrfi,
        "market_nrfi_prob": market_nrfi,
        "yrfi_edge": yrfi_edge,
        "nrfi_edge": nrfi_edge,
        "bet": bet,
        "edge": edge,
        "confidence": confidence,
    }


def _fmt_era(era: float | None) -> str:
    return f"{era:.2f}" if era is not None else "---"


def _edge_cell(edge: float | None) -> str:
    if edge is None:
        return "—"
    edge_str = f"{edge:+.1%}"
    if edge > 0.05:
        return f"🟢 {edge_str}"
    if edge < 0:
        return f"🔴 {edge_str}"
    return f"⬜ {edge_str}"


def render_nrfi_page(conn=None) -> None:
    """Render the NRFI/YRFI page."""
    st.header("🎯 NRFI / YRFI")
    st.caption(
        "First-inning runs props · Model estimates YRFI probability from starter ERA · "
        "Enter market odds to calculate edge"
    )

    # -----------------------------------------------------------------------
    # Date selector
    # -----------------------------------------------------------------------
    pred_date = st.date_input("Date", value=date.today(), key="nrfi_pred_date")

    # Clear per-game odds when the date changes
    last_loaded = st.session_state.get("nrfi_loaded_date")
    if last_loaded is not None and last_loaded != str(pred_date):
        for k in list(st.session_state.keys()):
            if k.startswith("yrfi_odds_") or k.startswith("nrfi_odds_"):
                del st.session_state[k]
        st.session_state.pop("nrfi_edge_results", None)
        st.session_state.pop("nrfi_predictions", None)

    load_btn = st.button("Load Predictions", key="nrfi_load_btn", type="primary")

    if load_btn:
        with st.spinner("Fetching game data and pitcher stats…"):
            try:
                from line_tracker.model.mlb_nrfi import predict_nrfi
                fi_history = _cached_fi_history(pred_date.year)
                preds = predict_nrfi(target_date=pred_date, fi_history=fi_history)
                st.session_state["nrfi_predictions"] = preds
                st.session_state["nrfi_loaded_date"] = str(pred_date)
                st.session_state.pop("nrfi_edge_results", None)
                if preds:
                    st.success(f"Loaded {len(preds)} games for {pred_date}.")
                else:
                    st.info(f"No MLB games found for {pred_date}.")
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.session_state["nrfi_predictions"] = []
            except Exception as exc:
                st.error(f"Error loading predictions: {exc}")
                log.exception("nrfi_page load error")
                st.session_state["nrfi_predictions"] = []

    preds = st.session_state.get("nrfi_predictions")
    if not preds:
        if preds is not None:
            st.info("No games found. Try a different date or check that the MLB model is trained.")
        return

    edge_results: dict = st.session_state.get("nrfi_edge_results", {})

    # -----------------------------------------------------------------------
    # Predictions table (updates after edge calculation)
    # -----------------------------------------------------------------------
    st.subheader("📊 Predictions")

    rows = []
    for p in preds:
        game_pk = p["game_pk"]
        away_br = p.get("away_team_br") or p["away_team"][:3].upper()
        home_br = p.get("home_team_br") or p["home_team"][:3].upper()
        label = f"{away_br} @ {home_br}"
        time_str = _fmt_commence(p.get("commence_time", ""))

        row: dict = {
            "Game": label,
            "Time": time_str,
            "Away SP": f"{p['away_pitcher']} ({_fmt_era(p['away_pitcher_era'])})",
            "Home SP": f"{p['home_pitcher']} ({_fmt_era(p['home_pitcher_era'])})",
            "YRFI%": f"{p['yrfi_prob']:.1%}",
        }

        er = edge_results.get(game_pk)
        if er:
            row["Market YRFI%"] = f"{er['market_yrfi_prob']:.1%}"
            bet = er.get("bet")
            edge = er.get("edge")
            row["Bet"] = bet if bet else "—"
            row["Edge"] = _edge_cell(edge)
            row["Confidence"] = er.get("confidence") or "—"
        else:
            row["Market YRFI%"] = "—"
            row["Bet"] = "—"
            row["Edge"] = "—"
            row["Confidence"] = "—"

        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "YRFI%": st.column_config.TextColumn(
                "YRFI%",
                help="Model estimate: probability that at least one run scores in the 1st inning",
            ),
            "Market YRFI%": st.column_config.TextColumn(
                "Market YRFI%",
                help="Devigged market probability derived from entered American odds",
            ),
            "Edge": st.column_config.TextColumn(
                "Edge",
                help="🟢 > +5% edge  ⬜ < 5% edge  🔴 negative edge",
            ),
        },
    )

    if edge_results:
        n_bets = sum(1 for er in edge_results.values() if er.get("bet"))
        st.caption(
            f"Edge calculated for {len(edge_results)} games · "
            f"{n_bets} bet{'s' if n_bets != 1 else ''} flagged (edge ≥ 5%)"
        )

    # -----------------------------------------------------------------------
    # Manual odds input
    # -----------------------------------------------------------------------
    st.subheader("📥 Enter Market Odds")
    st.caption(
        "Enter American odds from your sportsbook (e.g. -120 for YRFI, +100 for NRFI). "
        "Leave at 0 to skip a game."
    )

    # Header row
    hc1, hc2, hc3 = st.columns([2, 1, 1])
    hc1.markdown("**Game**")
    hc2.markdown("**YRFI Odds**")
    hc3.markdown("**NRFI Odds**")

    for p in preds:
        game_pk = p["game_pk"]
        away_br = p.get("away_team_br") or p["away_team"][:3].upper()
        home_br = p.get("home_team_br") or p["home_team"][:3].upper()
        label = f"{away_br} @ {home_br}"

        c1, c2, c3 = st.columns([2, 1, 1])
        c1.write(label)
        c2.number_input(
            f"YRFI odds for {label}",
            value=0,
            step=5,
            min_value=-10000,
            max_value=10000,
            key=f"yrfi_odds_{game_pk}",
            label_visibility="collapsed",
        )
        c3.number_input(
            f"NRFI odds for {label}",
            value=0,
            step=5,
            min_value=-10000,
            max_value=10000,
            key=f"nrfi_odds_{game_pk}",
            label_visibility="collapsed",
        )

    st.divider()

    if st.button("⚡ Calculate Edge", key="nrfi_calc_edge_btn", type="primary"):
        results: dict = {}
        n_entered = 0
        for p in preds:
            game_pk = p["game_pk"]
            yrfi_odds = int(st.session_state.get(f"yrfi_odds_{game_pk}", 0))
            nrfi_odds = int(st.session_state.get(f"nrfi_odds_{game_pk}", 0))
            if yrfi_odds == 0 or nrfi_odds == 0:
                continue
            n_entered += 1
            er = _calc_edge(p["yrfi_prob"], p["nrfi_prob"], yrfi_odds, nrfi_odds)
            if er is not None:
                results[game_pk] = er

        st.session_state["nrfi_edge_results"] = results

        if n_entered == 0:
            st.warning("Enter odds for at least one game first.")
        else:
            st.rerun()
