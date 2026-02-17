"""Performance analytics orchestration service."""

from __future__ import annotations

import pandas as pd

from line_tracker.core.logging import get_logger
from line_tracker.performance import build_clv_dataframe, summary_kpis as _summary_kpis


def load_clv_df(store, filters: dict | None = None) -> pd.DataFrame:
    """Load raw CLV rows from storage and build analytics DataFrame."""
    log = get_logger(__name__, source="performance_service")
    rows = store.get_all_clv()
    df = build_clv_dataframe(rows)
    log.info(
        "load_clv_df filters=%s raw_rows=%s df_rows=%s",
        filters or {},
        len(rows),
        len(df),
    )
    return df


def summary_kpis(df: pd.DataFrame) -> dict:
    """Passthrough for top-level KPI summary calculation."""
    return _summary_kpis(df)
