"""Performance analytics orchestration service."""

from __future__ import annotations

import pandas as pd

from line_tracker.performance import build_clv_dataframe, summary_kpis as _summary_kpis


def load_clv_df(store, filters: dict | None = None) -> pd.DataFrame:
    """Load raw CLV rows from storage and build analytics DataFrame."""
    _ = filters
    rows = store.get_all_clv()
    return build_clv_dataframe(rows)


def summary_kpis(df: pd.DataFrame) -> dict:
    """Passthrough for top-level KPI summary calculation."""
    return _summary_kpis(df)
