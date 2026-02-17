"""Smoke tests for the explainability UI component."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from line_tracker.models import BestBetResult
from line_tracker.ui.components.explainability import render_pick_explanation


def _make_bbr(**overrides) -> BestBetResult:
    """Build a minimal BestBetResult with sensible defaults."""
    defaults = dict(
        edge_pct=2.5,
        consensus_prob=0.55,
        best_odds_american=-110,
        best_odds_decimal=1.909,
        books_used=["FanDuel", "DraftKings", "BetMGM"],
        volatility_sigma=0.015,
        recency_weight=0.92,
        outliers_removed=1,
        explanation={
            "edge_breakdown": {"breakeven_prob": 0.5238, "edge_pp": 0.0262},
            "consensus_method": "trimmed_mean",
            "quality_factors": {
                "edge_score": 72,
                "agreement_score": 85,
                "coverage_score": 60,
                "freshness_score": 90,
            },
            "outlier_info": {
                "outlier_filtered": True,
                "outliers_removed": 1,
                "outlier_rate": 0.125,
            },
            "kelly": {"sizing_note": "1-unit"},
        },
        market="moneyline",
        selection="Celtics",
        side="home",
        confidence="High",
        quality_score=78,
        quality_tier="Strong",
        kelly_suggested=0.025,
        sizing_note="1-unit",
        best_sportsbook="FanDuel",
        books_used_count=3,
    )
    defaults.update(overrides)
    return BestBetResult(**defaults)


def _make_slate_entry(**overrides) -> dict:
    """Build a minimal Daily Slate entry dict."""
    bbr = _make_bbr()
    defaults = dict(
        event_id="lakers_celtics_12345",
        event="Lakers @ Celtics",
        market="moneyline",
        selection="Celtics",
        best_odds=-110,
        best_sportsbook="FanDuel",
        edge_pct=2.5,
        consensus_prob=0.55,
        quality_score=78,
        quality_tier="Strong",
        confidence="High",
        market_volatility_sigma=0.015,
        kelly_suggested=0.025,
        sizing_note="1-unit",
        best_bet_result=bbr,
    )
    defaults.update(overrides)
    return defaults


def _make_shopping_entry(**overrides) -> dict:
    """Build a minimal Best Lines standout dict."""
    defaults = dict(
        event="Lakers @ Celtics",
        market="ML",
        selection="Home",
        sportsbook="FanDuel",
        odds=-110,
        line=None,
        book_prob=0.5238,
        consensus_prob=0.55,
        edge=0.0262,
        dollar_impact=3.50,
        median_odds=-115,
        books_used_excl=4,
        consensus_method="median",
        d_best=1.909,
        d_ref=1.870,
        exec_adv_100=2.15,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("line_tracker.ui.components.explainability.st")
def test_render_pick_explanation_noop_when_debug_off(mock_st: MagicMock) -> None:
    """When debug_enabled=False, no Streamlit calls should be made."""
    entry = _make_slate_entry()
    render_pick_explanation(entry, debug_enabled=False)
    mock_st.expander.assert_not_called()


@patch("line_tracker.ui.components.explainability.st")
def test_render_pick_explanation_slate_no_exception(mock_st: MagicMock) -> None:
    """Slate context should render without raising."""
    # Set up mock context managers
    mock_expander = MagicMock()
    mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
    mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
    mock_st.checkbox.return_value = False

    entry = _make_slate_entry()
    render_pick_explanation(entry, debug_enabled=True, context="slate")

    mock_st.expander.assert_called_once()


@patch("line_tracker.ui.components.explainability.st")
def test_render_pick_explanation_shopping_no_exception(mock_st: MagicMock) -> None:
    """Shopping context should render without raising."""
    mock_expander = MagicMock()
    mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
    mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

    entry = _make_shopping_entry()
    render_pick_explanation(entry, debug_enabled=True, context="shopping")

    mock_st.expander.assert_called_once()


@patch("line_tracker.ui.components.explainability.st")
def test_render_slate_with_no_bbr(mock_st: MagicMock) -> None:
    """Slate entry without best_bet_result should not crash."""
    mock_expander = MagicMock()
    mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
    mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
    mock_st.checkbox.return_value = False

    entry = _make_slate_entry()
    entry.pop("best_bet_result")
    render_pick_explanation(entry, debug_enabled=True, context="slate")

    mock_st.expander.assert_called_once()


@patch("line_tracker.ui.components.explainability.st")
def test_raw_json_only_when_checkbox_checked(mock_st: MagicMock) -> None:
    """st.json should only be called when checkbox returns True."""
    mock_expander = MagicMock()
    mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
    mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

    # Checkbox unchecked
    mock_st.checkbox.return_value = False
    entry = _make_slate_entry()
    render_pick_explanation(entry, debug_enabled=True, context="slate")
    mock_st.json.assert_not_called()


def test_import_render_pick_explanation() -> None:
    """Verify the public API is importable."""
    from line_tracker.ui.components.explainability import render_pick_explanation

    assert callable(render_pick_explanation)
