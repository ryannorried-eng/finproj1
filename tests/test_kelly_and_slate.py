"""Tests for Kelly stake sizing and Daily Slate page routing."""

import ast
import inspect

from line_tracker.bet_slip import kelly_stake

# ---------------------------------------------------------------------------
# Kelly formula unit tests
# ---------------------------------------------------------------------------

def test_kelly_positive_edge():
    """Fair-coin bet at +150 -> positive Kelly."""
    # prob=0.5, odds=+150 -> b=1.5, Kelly f = (1.5*0.5 - 0.5)/1.5 = 1/6
    # Full Kelly on $1000 = $166.67; 25% Kelly = $41.67
    result = kelly_stake(prob=0.5, odds=150, bankroll=1000, fraction=0.25)
    assert result > 0
    full = kelly_stake(prob=0.5, odds=150, bankroll=1000, fraction=1.0)
    assert abs(full - 166.67) < 0.01
    assert abs(result - 41.67) < 0.01


def test_kelly_no_edge_clamps_to_zero():
    """When odds imply fair value, Kelly should be 0."""
    # prob=0.4, odds=+150 -> b=1.5, Kelly = (1.5*0.4 - 0.6)/1.5 = 0
    result = kelly_stake(prob=0.4, odds=150, bankroll=1000, fraction=0.25)
    assert result == 0.0


def test_kelly_negative_edge_clamps_to_zero():
    """When the bet has negative edge, Kelly should be 0."""
    # prob=0.3, odds=+150 -> b=1.5, Kelly = (1.5*0.3 - 0.7)/1.5 < 0
    result = kelly_stake(prob=0.3, odds=150, bankroll=1000, fraction=1.0)
    assert result == 0.0


def test_kelly_fractional_scales():
    """Fractional Kelly should scale linearly with the fraction."""
    full = kelly_stake(prob=0.6, odds=100, bankroll=1000, fraction=1.0)
    half = kelly_stake(prob=0.6, odds=100, bankroll=1000, fraction=0.5)
    assert full > 0
    assert abs(half - full * 0.5) < 0.01


def test_kelly_favorite_odds():
    """Kelly on a strong favorite line like -200."""
    # prob=0.75, odds=-200 -> b=0.5, Kelly = (0.5*0.75 - 0.25)/0.5 = 0.25
    full = kelly_stake(prob=0.75, odds=-200, bankroll=1000, fraction=1.0)
    assert abs(full - 250.0) < 0.01


# ---------------------------------------------------------------------------
# Daily Slate routing integration test
# ---------------------------------------------------------------------------

def test_daily_slate_in_nav_options():
    """'Daily Slate' must appear in the sidebar page list."""
    from line_tracker import dashboard

    source = inspect.getsource(dashboard._sidebar)
    tree = ast.parse(source)

    # Walk AST looking for the list containing "Daily Slate"
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.List):
            values = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    values.append(elt.value)
            if "Daily Slate" in values:
                found = True
                break
    assert found, "Daily Slate not found in sidebar nav radio options"


def test_daily_slate_route_in_main():
    """The main() function must route to _page_daily_slate."""
    from line_tracker import dashboard

    source = inspect.getsource(dashboard.main)
    assert "_page_daily_slate" in source, (
        "main() does not route to _page_daily_slate"
    )
    assert '"Daily Slate"' in source, (
        "main() does not check for 'Daily Slate' nav value"
    )
