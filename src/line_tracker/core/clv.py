"""Shared CLV formulas."""

from __future__ import annotations


def compute_clv_metrics(*, pick_dec, close_dec, pick_prob, close_prob) -> dict[str, object]:
    """Canonical CLV metrics (positive values indicate beating the close)."""
    if hasattr(close_prob, "isna"):
        close_prob = close_prob.where(~close_prob.isna(), pick_prob)
    elif close_prob is None:
        close_prob = pick_prob

    clv_decimal = pick_dec - close_dec
    clv_prob = close_prob - pick_prob

    if hasattr(clv_decimal, "round"):
        clv_decimal = clv_decimal.round(4)
    else:
        clv_decimal = round(clv_decimal, 4)

    if hasattr(clv_prob, "round"):
        clv_prob = clv_prob.round(4)
    else:
        clv_prob = round(clv_prob, 4)

    return {"clv_decimal": clv_decimal, "clv_prob": clv_prob}
