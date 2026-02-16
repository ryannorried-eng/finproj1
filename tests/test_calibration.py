"""Tests for tier calibration: grid search, monotonicity, fallback, storage, slate."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from line_tracker.calibration import (
    _DEFAULTS,
    _enforce_monotonicity,
    calibrate_thresholds,
    calibration_from_json,
    calibration_to_json,
    format_calibration_report,
    grid_search_thresholds,
)
from line_tracker.slate import (
    STANDARD_THRESHOLDS,
    classify_rec,
    thresholds_from_calibration,
)
from line_tracker.storage import LineStore

# ── Helpers ──────────────────────────────────────────────────────────


def _td(ev, ez, hmax, bmin):
    """Shorthand for a tier-threshold dict."""
    return {
        "edge_ev_100": ev,
        "edge_z": ez,
        "hold_max": hmax,
        "books_min": bmin,
    }


def _make_training_df(
    n: int = 300,
    *,
    edge_ev_100: float = 2.0,
    edge_z: float = 2.0,
    hold: float = 5.0,
    books_used: int = 6,
    beat_rate: float = 0.60,
    avg_clv: float = 0.003,
    days: int = 90,
) -> pd.DataFrame:
    """Create a synthetic training DataFrame."""
    import random

    random.seed(42)
    start = date.today() - timedelta(days=days)
    rows = []
    for i in range(n):
        beat = 1 if random.random() < beat_rate else 0
        if beat:
            clv = avg_clv + random.gauss(0, 0.001)
        else:
            clv = -abs(avg_clv) + random.gauss(0, 0.001)
        rows.append({
            "edge_ev_100": edge_ev_100,
            "edge_z": edge_z,
            "hold": hold,
            "books_used": books_used,
            "beat": beat,
            "clv": clv,
            "date": start + timedelta(days=i % days),
        })
    return pd.DataFrame(rows)


def _make_bimodal_df() -> pd.DataFrame:
    """Two clusters: strict-good and loose-bad."""
    import random

    random.seed(99)
    start = date.today() - timedelta(days=90)
    rows = []
    # 80 rows: high edge, good beat rate
    for i in range(80):
        rows.append({
            "edge_ev_100": 2.5,
            "edge_z": 2.0,
            "hold": 5.0,
            "books_used": 6,
            "beat": 1 if random.random() < 0.60 else 0,
            "clv": 0.004 + random.gauss(0, 0.001),
            "date": start + timedelta(days=i % 90),
        })
    # 220 rows: low edge, bad beat rate
    for i in range(220):
        rows.append({
            "edge_ev_100": 0.5,
            "edge_z": 0.5,
            "hold": 7.0,
            "books_used": 4,
            "beat": 1 if random.random() < 0.48 else 0,
            "clv": -0.001 + random.gauss(0, 0.002),
            "date": start + timedelta(days=i % 90),
        })
    return pd.DataFrame(rows)


# ── 1) Grid search picks stricter threshold when needed ─────────


class TestGridSearchPicksStricter:
    def test_strict_threshold_chosen_for_bimodal_data(self):
        """Only high-edge subset has good CLV; picks high score."""
        df = _make_bimodal_df()
        # Use tight volume band so loose combo is excluded
        result = grid_search_thresholds(
            df,
            "tier1a",
            constraints={
                "beat_pct_min": 55.0,
                "avg_clv_min": 0.002,
                "n_min": 10,
                "legs_day_lo": 0.1,
                "legs_day_hi": 2.0,
            },
        )
        assert not result["fallback_used"]
        # Strict subset has better beat_rate than mixed
        assert result["beat_rate"] > 50.0
        assert result["score"] > 0

    def test_grid_search_returns_metrics(self):
        df = _make_training_df()
        result = grid_search_thresholds(df, "tier1a")
        assert "n" in result
        assert "beat_rate" in result
        assert "avg_clv" in result
        assert "score" in result
        assert "legs_per_day" in result

    def test_grid_search_score_formula(self):
        """Score = 100*(beat_rate/100 - 0.50) + 500*avg_clv."""
        df = _make_training_df()
        result = grid_search_thresholds(
            df, "tier2",
            constraints={
                "n_min": 5,
                "legs_day_lo": 0.1,
                "legs_day_hi": 100.0,
                "beat_pct_min": 50.0,
                "avg_clv_min": 0.0,
            },
        )
        br = result["beat_rate"] / 100.0
        expected = 100.0 * (br - 0.50) + 500.0 * result["avg_clv"]
        assert abs(result["score"] - round(expected, 4)) < 0.1


# ── 2) Monotonicity enforcement ─────────────────────────────────


class TestMonotonicity:
    def test_violated_edge_snapped(self):
        """tier1b.edge_ev_100 > tier1a => snapped down."""
        tiers = {
            "tier1a": _td(1.5, 1.5, 6.0, 6),
            "tier1b": _td(2.0, 1.0, 7.0, 5),
            "tier2": _td(0.5, 0.5, 8.0, 4),
        }
        fixed = _enforce_monotonicity(tiers)
        t1a = fixed["tier1a"]["edge_ev_100"]
        t1b = fixed["tier1b"]["edge_ev_100"]
        assert t1b <= t1a

    def test_violated_edge_z_snapped(self):
        tiers = {
            "tier1a": _td(2.0, 1.0, 6.0, 6),
            "tier1b": _td(1.0, 1.5, 7.0, 5),
            "tier2": _td(0.5, 0.5, 8.0, 4),
        }
        fixed = _enforce_monotonicity(tiers)
        assert fixed["tier1b"]["edge_z"] <= fixed["tier1a"]["edge_z"]

    def test_violated_hold_snapped(self):
        """tier1b.hold_max < tier1a => snapped up."""
        tiers = {
            "tier1a": _td(2.0, 2.0, 7.0, 6),
            "tier1b": _td(1.0, 1.0, 6.0, 5),
            "tier2": _td(0.5, 0.5, 8.0, 4),
        }
        fixed = _enforce_monotonicity(tiers)
        t1a = fixed["tier1a"]["hold_max"]
        t1b = fixed["tier1b"]["hold_max"]
        assert t1b >= t1a

    def test_violated_books_snapped(self):
        tiers = {
            "tier1a": _td(2.0, 2.0, 6.0, 4),
            "tier1b": _td(1.0, 1.0, 7.0, 6),
            "tier2": _td(0.5, 0.5, 8.0, 5),
        }
        fixed = _enforce_monotonicity(tiers)
        t1a = fixed["tier1a"]["books_min"]
        t1b = fixed["tier1b"]["books_min"]
        t2 = fixed["tier2"]["books_min"]
        assert t1b <= t1a
        assert t2 <= t1b

    def test_already_monotonic_unchanged(self):
        import copy

        tiers = {
            "tier1a": _td(3.0, 2.5, 6.0, 7),
            "tier1b": _td(2.0, 1.5, 7.0, 5),
            "tier2": _td(1.0, 1.0, 8.0, 4),
        }
        original = copy.deepcopy(tiers)
        fixed = _enforce_monotonicity(tiers)
        assert fixed == original

    def test_calibrate_enforces_monotonicity(self):
        df = _make_training_df()
        result = calibrate_thresholds(df)
        t1a = result["tier1a"]
        t1b = result["tier1b"]
        t2 = result["tier2"]
        ev1a = t1a["edge_ev_100"]
        ev1b = t1b["edge_ev_100"]
        ev2 = t2["edge_ev_100"]
        assert ev1a >= ev1b >= ev2
        assert t1a["hold_max"] <= t1b["hold_max"] <= t2["hold_max"]


# ── 3) Fallback when insufficient sample ────────────────────────


class TestFallbackInsufficientSample:
    def test_empty_df_returns_defaults(self):
        result = calibrate_thresholds(pd.DataFrame())
        assert result["fallback_used"] is True
        assert result["training_rows"] == 0
        for tier in ("tier1a", "tier1b", "tier2"):
            assert result[tier]["fallback_used"] is True

    def test_tiny_df_returns_fallback(self):
        """3 rows can't meet n_min for any tier."""
        df = _make_training_df(n=3, days=1)
        result = calibrate_thresholds(df)
        any_fallback = any(
            result[t].get("fallback_used", False)
            for t in ("tier1a", "tier1b", "tier2")
        )
        assert any_fallback

    def test_grid_search_fallback_has_default_values(self):
        result = grid_search_thresholds(pd.DataFrame(), "tier1a")
        assert result["fallback_used"] is True
        expected = _DEFAULTS["tier1a"]["edge_ev_100"]
        assert result["edge_ev_100"] == expected


# ── 4) Storage roundtrip ────────────────────────────────────────


class TestStorageRoundtrip:
    def test_save_load_calibration(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = LineStore(db_path)

        result = {
            "tier1a": {"edge_ev_100": 2.5},
            "training_rows": 100,
        }
        json_str = calibration_to_json(result)
        store.save_calibration("global", json_str)

        loaded = store.load_calibration("global")
        assert loaded is not None
        parsed = calibration_from_json(loaded)
        assert parsed == result
        store.close()

    def test_load_missing_key(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = LineStore(db_path)
        assert store.load_calibration("nonexistent") is None
        store.close()

    def test_overwrite_calibration(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = LineStore(db_path)

        store.save_calibration("global", '{"v": 1}')
        store.save_calibration("global", '{"v": 2}')
        raw = store.load_calibration("global")
        loaded = calibration_from_json(raw)
        assert loaded["v"] == 2
        store.close()

    def test_sport_specific_key(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = LineStore(db_path)

        store.save_calibration(
            "NCAAB:spread", '{"tier1a": {}}',
        )
        store.save_calibration(
            "global", '{"tier1a": {"edge_ev_100": 2}}',
        )

        ncaab = calibration_from_json(
            store.load_calibration("NCAAB:spread")
        )
        glob = calibration_from_json(
            store.load_calibration("global")
        )
        assert ncaab != glob
        store.close()


# ── 5) Slate uses Auto thresholds ──────────────────────────────


class TestSlateUsesAutoThresholds:
    def test_thresholds_from_calibration_maps_fields(self):
        cal = {
            "tier1a": _td(3.0, 2.0, 5.5, 7),
            "tier1b": _td(1.5, 1.0, 7.0, 5),
            "tier2": _td(0.75, 1.0, 8.0, 4),
        }
        th = thresholds_from_calibration(cal)
        assert th.mode == "Auto"
        assert th.tier1a_base_edge == 3.0
        assert th.tier1a_hold_max == 5.5
        assert th.tier1b_base_edge == 1.5
        assert th.tier2_edge_min == 0.75
        assert th.tier2_edge_z_min == 1.0
        assert th.min_books == 4

    def test_auto_thresholds_change_classification(self):
        """Rec passes Standard 1A but fails stricter Auto."""
        entry = {
            "edge_pct": 2.5,
            "confidence": "High",
            "quality_tier": "Elite",
            "market_volatility_sigma": 0.01,
            "edge_z": 2.0,
            "market_unstable": False,
            "books_used": 6,
            "oldest_update_age_min": 5.0,
            "market_hold_median": 5.0,
        }
        # Standard: floor = max(2.0, 100*0.01) = 2.0 → pass
        std = classify_rec(
            entry, thresholds=STANDARD_THRESHOLDS,
        )
        assert std["tier"] == "tier1a"

        # Auto with stricter base_edge = 3.0
        cal = {
            "tier1a": _td(3.0, 2.0, 5.5, 6),
            "tier1b": _td(1.5, 1.0, 7.0, 5),
            "tier2": _td(0.5, 0.5, 8.0, 4),
        }
        auto_th = thresholds_from_calibration(cal)
        auto = classify_rec(entry, thresholds=auto_th)
        # 2.5 < 3.0 floor → not tier1a
        assert auto["tier"] != "tier1a"

    def test_auto_thresholds_loosen_tier2(self):
        """Auto with edge_z=0 lets more recs into tier2."""
        entry = {
            "edge_pct": 0.8,
            "confidence": "Medium",
            "quality_tier": "Moderate",
            "market_volatility_sigma": 0.01,
            "edge_z": 0.3,
            "market_unstable": False,
            "books_used": 5,
            "oldest_update_age_min": 5.0,
            "market_hold_median": 5.0,
        }
        cal = {
            "tier1a": _td(2.0, 2.0, 6.0, 6),
            "tier1b": _td(1.0, 1.0, 7.0, 5),
            "tier2": _td(0.5, 0.0, 8.0, 4),
        }
        auto_th = thresholds_from_calibration(cal)
        auto = classify_rec(entry, thresholds=auto_th)
        # edge_z gate disabled (0.0) → tier2
        assert auto["tier"] == "tier2"


# ── 6) Report formatting ───────────────────────────────────────


class TestFormatReport:
    def test_report_contains_tiers(self):
        result = calibrate_thresholds(pd.DataFrame())
        report = format_calibration_report(result)
        assert "TIER1A" in report
        assert "TIER1B" in report
        assert "TIER2" in report

    def test_report_shows_fallback_warning(self):
        result = calibrate_thresholds(pd.DataFrame())
        report = format_calibration_report(result)
        assert "fallback" in report.lower()


# ── 7) JSON roundtrip ──────────────────────────────────────────


class TestJsonRoundtrip:
    def test_serialize_deserialize(self):
        df = _make_training_df()
        result = calibrate_thresholds(df)
        j = calibration_to_json(result)
        parsed = calibration_from_json(j)
        for tier in ("tier1a", "tier1b", "tier2"):
            assert (
                parsed[tier]["edge_ev_100"]
                == result[tier]["edge_ev_100"]
            )
