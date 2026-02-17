"""Smoke tests for Streamlit dashboard startup."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from streamlit.testing.v1 import AppTest
except Exception:  # pragma: no cover - depends on installed streamlit version
    AppTest = None


@pytest.mark.skipif(AppTest is None, reason="streamlit.testing.v1 is unavailable")
def test_dashboard_smoke_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dashboard should render key UI without raising exceptions."""
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    monkeypatch.syspath_prepend(str(src_path))

    app_file = repo_root / "src" / "line_tracker" / "dashboard.py"
    at = AppTest.from_file(str(app_file))
    at.run(timeout=30)

    assert not at.exception
    assert any(title.value == "Sports Betting Line Tracker" for title in at.title)

    page_radios = [radio for radio in at.radio if radio.label == "Page"]
    assert page_radios, "Expected sidebar Page navigation radio"
    options = set(page_radios[0].options)
    assert {"Dashboard", "Best Lines to Shop", "Daily Slate", "Performance"}.issubset(options)
