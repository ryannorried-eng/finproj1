"""Tests for debug pipeline: PRUNE_DEBUG_MODE, SLATE_MAX_PER_EVENT, debug logs."""

from __future__ import annotations

import logging

from line_tracker.config import get_debug_prune_profile, get_prune_debug_mode, get_slate_max_per_event
from line_tracker.services.automation_service import _log_candidate_summary
from line_tracker.services.pick_pruning_service import prune_picks, prune_picks_with_reasons


def _make_entry(**overrides) -> dict:
    """Create a pick entry that passes all default gates."""
    base = {
        "event_id": "e1",
        "market": "spread",
        "selection": "Chiefs",
        "tier": "tier1b",
        "alpha_label": "Strong",
        "edge_z": 2.5,
        "edge_ev_shrunk": 0.03,
        "quality_score": 85,
        "books_used": 7,
        "market_hold_median": 5.0,
        "best_odds": -110,
        "best_sportsbook": "FanDuel",
        "consensus_prob": 0.55,
    }
    base.update(overrides)
    return base


class TestPruneDebugModeTierGate:
    """PRUNE_DEBUG_MODE allows tier3 through tier gate."""

    def test_tier3_blocked_in_normal_mode(self, monkeypatch):
        monkeypatch.delenv("PRUNE_DEBUG_MODE", raising=False)
        entry = _make_entry(tier="tier3")
        result = prune_picks([entry])
        assert len(result) == 0

    def test_tier3_allowed_in_debug_mode(self, monkeypatch):
        monkeypatch.setenv("PRUNE_DEBUG_MODE", "1")
        entry = _make_entry(tier="tier3", edge_z=0.8, edge_ev_shrunk=0.01)
        result = prune_picks([entry])
        assert len(result) == 1

    def test_tier4_stay_away_blocked_in_debug_mode(self, monkeypatch):
        monkeypatch.setenv("PRUNE_DEBUG_MODE", "1")
        entry = _make_entry(tier="stay_away")
        result = prune_picks([entry])
        assert len(result) == 0

    def test_avoid_blocked_in_debug_mode(self, monkeypatch):
        monkeypatch.setenv("PRUNE_DEBUG_MODE", "1")
        entry = _make_entry(tier="avoid")
        result = prune_picks([entry])
        assert len(result) == 0


class TestPruneDebugModeAlphaGate:
    """PRUNE_DEBUG_MODE does not block on Weak/missing alpha."""

    def test_weak_alpha_blocked_in_normal_mode(self, monkeypatch):
        monkeypatch.delenv("PRUNE_DEBUG_MODE", raising=False)
        entry = _make_entry(alpha_label="Weak")
        result = prune_picks([entry])
        assert len(result) == 0

    def test_weak_alpha_allowed_in_debug_mode(self, monkeypatch):
        monkeypatch.setenv("PRUNE_DEBUG_MODE", "1")
        entry = _make_entry(alpha_label="Weak", edge_z=0.8, edge_ev_shrunk=0.01)
        result = prune_picks([entry])
        assert len(result) == 1

    def test_missing_alpha_allowed_in_debug_mode(self, monkeypatch):
        monkeypatch.setenv("PRUNE_DEBUG_MODE", "1")
        entry = _make_entry(alpha_label="", edge_z=0.8, edge_ev_shrunk=0.01)
        result = prune_picks([entry])
        assert len(result) == 1

    def test_alpha_still_in_output_in_debug_mode(self, monkeypatch):
        monkeypatch.setenv("PRUNE_DEBUG_MODE", "1")
        entry = _make_entry(alpha_label="Weak", edge_z=0.8, edge_ev_shrunk=0.01)
        result = prune_picks([entry])
        assert len(result) == 1
        assert result[0]["alpha_label"] == "Weak"


class TestNormalModeBehaviorUnchanged:
    """Normal mode behavior remains unchanged with debug mode off."""

    def test_normal_tier_gate(self, monkeypatch):
        monkeypatch.delenv("PRUNE_DEBUG_MODE", raising=False)
        entry = _make_entry(tier="tier3")
        result = prune_picks([entry])
        assert len(result) == 0

    def test_normal_alpha_gate(self, monkeypatch):
        monkeypatch.delenv("PRUNE_DEBUG_MODE", raising=False)
        entry = _make_entry(alpha_label="Weak")
        result = prune_picks([entry])
        assert len(result) == 0

    def test_normal_passing_entry(self, monkeypatch):
        monkeypatch.delenv("PRUNE_DEBUG_MODE", raising=False)
        entry = _make_entry()
        result = prune_picks([entry])
        assert len(result) == 1

    def test_reasons_histogram_tier_in_debug(self, monkeypatch):
        monkeypatch.setenv("PRUNE_DEBUG_MODE", "1")
        entry = _make_entry(tier="avoid")
        _, reasons = prune_picks_with_reasons([entry])
        assert reasons["tier"] == 1


class TestSlateMaxPerEventConfig:
    """SLATE_MAX_PER_EVENT config is respected."""

    def test_default_max_per_event_is_1(self, monkeypatch):
        monkeypatch.delenv("SLATE_MAX_PER_EVENT", raising=False)
        assert get_slate_max_per_event() == 1

    def test_custom_max_per_event(self, monkeypatch):
        monkeypatch.setenv("SLATE_MAX_PER_EVENT", "3")
        assert get_slate_max_per_event() == 3


class TestDebugPruneProfile:
    """get_debug_prune_profile returns correct values."""

    def test_profile_includes_tier3(self):
        profile = get_debug_prune_profile()
        assert "tier3" in profile["allowed_tiers"]

    def test_profile_excludes_avoid(self):
        profile = get_debug_prune_profile()
        assert "avoid" not in profile["allowed_tiers"]
        assert "stay_away" not in profile["allowed_tiers"]

    def test_profile_includes_weak_alpha(self):
        profile = get_debug_prune_profile()
        assert "Weak" in profile["allowed_alpha"]
        assert "" in profile["allowed_alpha"]

    def test_profile_edge_z_capped(self, monkeypatch):
        monkeypatch.setenv("PRUNE_MIN_EDGE_Z", "2.0")
        profile = get_debug_prune_profile()
        assert profile["min_edge_z"] == 0.50

    def test_profile_edge_z_uses_existing_if_lower(self, monkeypatch):
        monkeypatch.setenv("PRUNE_MIN_EDGE_Z", "0.30")
        profile = get_debug_prune_profile()
        assert profile["min_edge_z"] == 0.30


class TestCandidateSummaryLog:
    """Debug logs execute without crashing and include expected fragments."""

    def test_summary_log_no_crash(self, caplog):
        logger = logging.getLogger("line_tracker.services.automation_service")
        logger.propagate = True
        try:
            entries = [
                _make_entry(tier="tier3", market="spread", alpha_label="Weak", edge_z=0.5, edge_ev_shrunk=0.01),
                _make_entry(tier="tier1b", market="moneyline", alpha_label="Strong", edge_z=2.0, edge_ev_shrunk=0.03),
            ]
            with caplog.at_level(logging.INFO, logger="line_tracker.services.automation_service"):
                _log_candidate_summary(entries)
            combined = " ".join(caplog.messages)
            assert "candidate_summary" in combined
            assert "n=2" in combined
        finally:
            logger.propagate = False

    def test_summary_log_includes_tier_counts(self, caplog):
        logger = logging.getLogger("line_tracker.services.automation_service")
        logger.propagate = True
        try:
            entries = [
                _make_entry(tier="tier3"),
                _make_entry(tier="tier3"),
                _make_entry(tier="tier1b"),
            ]
            with caplog.at_level(logging.INFO, logger="line_tracker.services.automation_service"):
                _log_candidate_summary(entries)
            combined = " ".join(caplog.messages)
            assert "tier3=2" in combined
            assert "tier1b=1" in combined
        finally:
            logger.propagate = False

    def test_summary_log_includes_edge_stats(self, caplog):
        logger = logging.getLogger("line_tracker.services.automation_service")
        logger.propagate = True
        try:
            entries = [
                _make_entry(edge_z=1.0, edge_ev_shrunk=0.02),
                _make_entry(edge_z=3.0, edge_ev_shrunk=0.04),
            ]
            with caplog.at_level(logging.INFO, logger="line_tracker.services.automation_service"):
                _log_candidate_summary(entries)
            combined = " ".join(caplog.messages)
            assert "edge_z(min/med/max)=" in combined
            assert "ev_shrunk(min/med/max)=" in combined
        finally:
            logger.propagate = False

    def test_summary_empty_entries(self):
        # Should not crash even with empty list
        _log_candidate_summary([])


class TestPruneDebugModeConfig:
    """Config getter for PRUNE_DEBUG_MODE."""

    def test_default_is_off(self, monkeypatch):
        monkeypatch.delenv("PRUNE_DEBUG_MODE", raising=False)
        assert get_prune_debug_mode() is False

    def test_enabled(self, monkeypatch):
        monkeypatch.setenv("PRUNE_DEBUG_MODE", "1")
        assert get_prune_debug_mode() is True

    def test_enabled_true(self, monkeypatch):
        monkeypatch.setenv("PRUNE_DEBUG_MODE", "true")
        assert get_prune_debug_mode() is True
