"""Daily slate orchestration service."""

from __future__ import annotations

from line_tracker.slate import build_daily_slate, thresholds_from_calibration


def build_daily_slate_service(
    lines_by_event: dict,
    *,
    filters: dict,
    mode: str,
    store,
    tiering_method: str = "hybrid",
):
    """Build daily slate using optional calibration thresholds in Auto mode."""
    thresholds = None
    if mode == "Auto":
        cal_json = store.load_calibration("global")
        if cal_json:
            from line_tracker.calibration import calibration_from_json

            cal_dict = calibration_from_json(cal_json)
            thresholds = thresholds_from_calibration(cal_dict)
    return build_daily_slate(
        lines_by_event,
        filters=filters,
        thresholds=thresholds,
        tiering_method=tiering_method,
    )
