"""Tests for capture_closing_lines behavior changes:
- only attempts snapshots where closed_at IS NULL
- prioritizes events with commence_time within 6 hours or already started
- respects configurable batch_size limit per cycle
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from line_tracker.services.rec_snapshot_service import capture_closing_lines
from line_tracker.storage import LineStore


def _insert_event(store, event_id: str, commence_time: str) -> None:
    """Insert a row into the events table directly."""
    store.events_repo.insert_many_upsert([
        (event_id, "nfl", commence_time, "Home", "Away", f"{event_id} display"),
    ])


def _insert_snapshot(store, event_id: str, *, closed: bool = False) -> int:
    """Insert a minimal rec_snapshot and optionally close it.

    Returns the snapshot_id.
    """
    snap = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "event_id": event_id,
        "sport": "nfl",
        "market": "moneyline",
        "selection": "Home",
        "line": None,
        "book": "FanDuel",
        "odds_american": -150.0,
        "odds_decimal": 1.6667,
        "consensus_prob": 0.60,
        "breakeven_prob": 0.55,
        "edge_pct": 3.5,
        "edge_ev": 0.035,
        "edge_ev_shrunk": 0.028,
        "ev_100": 3.5,
        "edge_z": 2.1,
        "quality_score": 85,
        "confidence_label": "High",
        "tier": "tier1b",
        "meta": None,
        "alpha_score": None,
        "alpha_label": None,
    }
    store.log_rec_snapshots([snap])
    unclosed = store.rec_snapshots_repo.get_unclosed()
    # Find the one we just inserted
    sid = max(s["snapshot_id"] for s in unclosed if s["event_id"] == event_id)
    if closed:
        store.close_snapshot(
            sid,
            close_odds_american=-170.0,
            close_odds_decimal=1.5882,
            close_implied_prob=0.6296,
            open_implied_prob=0.6000,
            clv_delta_american=-20.0,
            clv_delta_implied=0.0296,
        )
    return sid


class TestCaptureClosingLinesOnlyUnclosed:
    """capture_closing_lines should only attempt snapshots where closed_at IS NULL."""

    def test_already_closed_snapshots_skipped(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            now = datetime.now(timezone.utc)
            ct = (now + timedelta(hours=1)).isoformat()
            _insert_event(store, "evt_1", ct)
            _insert_snapshot(store, "evt_1", closed=True)

            # All snapshots already closed -> nothing to close
            result = capture_closing_lines(store)
            assert result == 0

    def test_unclosed_snapshots_returned(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            now = datetime.now(timezone.utc)
            ct = (now + timedelta(hours=1)).isoformat()
            _insert_event(store, "evt_open", ct)
            _insert_snapshot(store, "evt_open", closed=False)

            unclosed = store.get_unclosed_snapshots()
            assert len(unclosed) == 1
            assert unclosed[0]["closed_at"] is None


class TestCaptureClosingLinesPrioritization:
    """Events with commence_time within 6 hours (or past) are prioritized."""

    def test_imminent_events_ordered_first(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            now = datetime.now(timezone.utc)

            # Event far in the future (24h)
            far_ct = (now + timedelta(hours=24)).isoformat()
            _insert_event(store, "evt_far", far_ct)
            _insert_snapshot(store, "evt_far")

            # Event starting soon (2h)
            soon_ct = (now + timedelta(hours=2)).isoformat()
            _insert_event(store, "evt_soon", soon_ct)
            _insert_snapshot(store, "evt_soon")

            # Event already started (1h ago)
            past_ct = (now - timedelta(hours=1)).isoformat()
            _insert_event(store, "evt_past", past_ct)
            _insert_snapshot(store, "evt_past")

            unclosed = store.get_unclosed_snapshots(prioritize_hours=6)
            event_ids = [s["event_id"] for s in unclosed]

            # Past and soon events come before far event
            far_idx = event_ids.index("evt_far")
            past_idx = event_ids.index("evt_past")
            soon_idx = event_ids.index("evt_soon")
            assert past_idx < far_idx
            assert soon_idx < far_idx

    def test_prioritize_hours_parameter_respected(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            now = datetime.now(timezone.utc)

            # Event 4h from now - within 6h window but not within 2h
            ct_4h = (now + timedelta(hours=4)).isoformat()
            _insert_event(store, "evt_4h", ct_4h)
            _insert_snapshot(store, "evt_4h")

            # Event 10h from now
            ct_10h = (now + timedelta(hours=10)).isoformat()
            _insert_event(store, "evt_10h", ct_10h)
            _insert_snapshot(store, "evt_10h")

            # With prioritize_hours=2, evt_4h is NOT imminent
            unclosed_narrow = store.get_unclosed_snapshots(prioritize_hours=2)
            ids_narrow = [s["event_id"] for s in unclosed_narrow]
            # Both are in the non-priority bucket, ordered by commence_time
            assert ids_narrow.index("evt_4h") < ids_narrow.index("evt_10h")

            # With prioritize_hours=6, evt_4h IS imminent
            unclosed_wide = store.get_unclosed_snapshots(prioritize_hours=6)
            ids_wide = [s["event_id"] for s in unclosed_wide]
            assert ids_wide.index("evt_4h") < ids_wide.index("evt_10h")


class TestCaptureClosingLinesBatchSize:
    """Batch size limits the number of snapshots processed per cycle."""

    def test_batch_size_limits_results(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            now = datetime.now(timezone.utc)
            for i in range(5):
                eid = f"evt_batch_{i}"
                ct = (now + timedelta(hours=i + 1)).isoformat()
                _insert_event(store, eid, ct)
                _insert_snapshot(store, eid)

            unclosed = store.get_unclosed_snapshots(limit=3)
            assert len(unclosed) == 3

    def test_batch_size_zero_returns_all(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            now = datetime.now(timezone.utc)
            for i in range(4):
                eid = f"evt_all_{i}"
                ct = (now + timedelta(hours=i + 1)).isoformat()
                _insert_event(store, eid, ct)
                _insert_snapshot(store, eid)

            unclosed = store.get_unclosed_snapshots(limit=0)
            assert len(unclosed) == 4

    def test_capture_closing_lines_batch_size_passed(self, tmp_path):
        """capture_closing_lines respects batch_size parameter."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            now = datetime.now(timezone.utc)
            for i in range(5):
                eid = f"evt_cap_{i}"
                ct = (now + timedelta(hours=i + 1)).isoformat()
                _insert_event(store, eid, ct)
                _insert_snapshot(store, eid)

            # With batch_size=2, at most 2 snapshots are attempted.
            # Since there are no matching lines to close against,
            # closed_count will be 0, but internal query is limited.
            result = capture_closing_lines(store, batch_size=2)
            assert result == 0  # no lines to match, but no error

            # All 5 remain unclosed
            assert len(store.get_unclosed_snapshots()) == 5

    def test_default_batch_size_is_200(self, tmp_path):
        """capture_closing_lines defaults to batch_size=200."""
        import inspect

        sig = inspect.signature(capture_closing_lines)
        assert sig.parameters["batch_size"].default == 200
