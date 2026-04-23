"""MLB Player Props page — HR, Total Bases, Pitcher Ks."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

log = logging.getLogger(__name__)


def _pct(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.1f}%"


def _fmt(val: float | None, decimals: int = 2) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def render_mlb_props_page() -> None:
    st.header("⚾ MLB Player Props")
    st.caption("Predictions derived from MLB Stats API season data — no prop lines required.")

    with st.spinner("Loading predictions…"):
        try:
            from line_tracker.model.mlb_predict import predict_mlb_games
            preds = predict_mlb_games()
        except Exception as exc:
            st.error(f"Could not load predictions: {exc}")
            return

    if not preds:
        st.info("No games found for today.")
        return

    games_with_props = [
        p for p in preds
        if p.get("hr_props") or p.get("home_k_props") or p.get("away_k_props")
    ]

    if not games_with_props:
        st.info("No prop data available yet — lineups may not be posted.")
        # Still render all games to show K props from pitching stats
        games_with_props = preds

    for pred in games_with_props:
        home = pred.get("home_team", "")
        away = pred.get("away_team", "")
        home_p = pred.get("home_pitcher") or "TBD"
        away_p = pred.get("away_pitcher") or "TBD"

        with st.expander(f"**{away} @ {home}**  ·  {home_p} vs {away_p}", expanded=False):

            # ── HR Props ────────────────────────────────────────────────
            hr_props = pred.get("hr_props") or []
            if hr_props:
                st.subheader("💥 HR Props")
                hr_rows = []
                for b in hr_props:
                    hr_rows.append({
                        "Batter":        b.get("batter_name", ""),
                        "Team":          b.get("team", ""),
                        "Order":         b.get("batting_order", ""),
                        "HR Prob":       _pct(b.get("hr_prob")),
                        "Exp HRs":       _fmt(b.get("expected_hrs"), 3),
                        "AB/HR":         _fmt(b.get("ab_per_hr"), 1),
                        "Pitcher Adj":   _fmt(b.get("vs_pitcher_adj"), 3),
                    })
                if hr_rows:
                    st.dataframe(pd.DataFrame(hr_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("HR props unavailable — lineup not yet posted.")

            # ── Total Bases Props ────────────────────────────────────────
            home_tb = pred.get("home_tb_props") or []
            away_tb = pred.get("away_tb_props") or []
            if home_tb or away_tb:
                st.subheader("🏃 Total Bases Props")
                tb_rows = []
                for b in (away_tb[:5] + home_tb[:5]):
                    tb_rows.append({
                        "Batter":    b.get("batter_name", ""),
                        "Team":      b.get("team", ""),
                        "Order":     b.get("batting_order", ""),
                        "Exp TB":    _fmt(b.get("expected_tbs"), 2),
                        "Over 1.5":  _pct(b.get("over_1_5")),
                        "Over 2.5":  _pct(b.get("over_2_5")),
                    })
                if tb_rows:
                    st.dataframe(pd.DataFrame(tb_rows), use_container_width=True, hide_index=True)

            # ── Pitcher K Props ──────────────────────────────────────────
            st.subheader("🎯 Pitcher K Props")
            k_cols = st.columns(2)
            for col, side, k_props in [
                (k_cols[0], f"{home_p} ({home})", pred.get("home_k_props") or {}),
                (k_cols[1], f"{away_p} ({away})", pred.get("away_k_props") or {}),
            ]:
                with col:
                    st.markdown(f"**{side}**")
                    if not k_props or not k_props.get("expected_ks"):
                        st.caption("No data")
                        continue
                    lines = k_props.get("prop_lines") or {}
                    k_data = {
                        "Metric": [
                            "Expected Ks",
                            "K prob / PA",
                            "Exp innings",
                            "Opp lineup K%",
                            "Over 4.5",
                            "Over 5.5",
                            "Over 6.5",
                            "Over 7.5",
                        ],
                        "Value": [
                            _fmt(k_props.get("expected_ks")),
                            _pct(k_props.get("k_prob_per_pa")),
                            _fmt(k_props.get("expected_innings"), 1),
                            _pct(k_props.get("vs_lineup_k_rate")),
                            _pct(lines.get("over_4.5")),
                            _pct(lines.get("over_5.5")),
                            _pct(lines.get("over_6.5")),
                            _pct(lines.get("over_7.5")),
                        ],
                    }
                    st.dataframe(
                        pd.DataFrame(k_data),
                        use_container_width=True,
                        hide_index=True,
                    )
