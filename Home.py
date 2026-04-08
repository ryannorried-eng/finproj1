"""BoomTown — Sports Betting Analytics landing page.

Run with:
    streamlit run Home.py
"""

import streamlit as st

st.set_page_config(
    page_title="BoomTown | Sports Betting Analytics",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
<style>
/* ── Layout ── */
.block-container { padding-top: 3rem; padding-bottom: 3rem; }

/* ── Hero ── */
.hero-title {
    font-size: 4rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-accent { color: #FF4B4B; }
.hero-tagline {
    font-size: 1.15rem;
    color: #9ca3af;
    margin-bottom: 2.5rem;
    font-weight: 400;
}

/* ── Feature cards ── */
.feature-card {
    background: #1e2129;
    border: 1px solid #2d3139;
    border-radius: 12px;
    padding: 1.5rem 1.25rem;
    height: 100%;
    transition: border-color 0.2s ease;
}
.feature-card:hover { border-color: #FF4B4B; }
.feature-icon { font-size: 2rem; margin-bottom: 0.75rem; }
.feature-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.4rem;
}
.feature-desc {
    font-size: 0.83rem;
    color: #9ca3af;
    line-height: 1.55;
}

/* ── Footer note ── */
.footer-note {
    color: #4b5563;
    font-size: 0.78rem;
    margin-top: 2.5rem;
    text-align: left;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Hero ────────────────────────────────────────────────────────────────────
_, center, _ = st.columns([1, 3, 1])
with center:
    st.markdown(
        '<div class="hero-title">'
        'Boom<span class="hero-accent">Town</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-tagline">'
        "Live odds, ML models, and arbitrage detection across MLB, NBA, and NCAAB"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Feature cards ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4, gap="small")
    _cards = [
        (
            "🏆",
            "Multi-Sport Coverage",
            "MLB, NBA, NCAAB, NFL, NHL, and more — all in one dashboard.",
        ),
        (
            "📊",
            "Live Odds from 4+ Books",
            "Real-time lines from DraftKings, FanDuel, BetMGM, and Caesars.",
        ),
        (
            "🤖",
            "ML-Powered Predictions",
            "Pitcher-aware MLB model and NCAAB ensemble picks with confidence tiers.",
        ),
        (
            "⚡",
            "Arbitrage Alerts",
            "Automatic moneyline and spread arb detection across all active markets.",
        ),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], _cards):
        with col:
            st.markdown(
                f'<div class="feature-card">'
                f'<div class="feature-icon">{icon}</div>'
                f'<div class="feature-title">{title}</div>'
                f'<div class="feature-desc">{desc}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CTA ──────────────────────────────────────────────────────────────────
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        if st.button("Enter App →", type="primary", use_container_width=True):
            st.switch_page("pages/Dashboard.py")

    st.markdown(
        '<div class="footer-note">'
        "Powered by The Odds API &nbsp;·&nbsp; Built with Streamlit"
        "&nbsp;·&nbsp; Data for entertainment purposes only"
        "</div>",
        unsafe_allow_html=True,
    )
