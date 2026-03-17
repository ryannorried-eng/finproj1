"""Tests for pick-pruning config-driven thresholds."""

from __future__ import annotations

import os
from unittest.mock import patch

from line_tracker.services.pick_pruning_service import (
    prune_picks,
    prune_picks_with_reasons,
)


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


class TestConfigDrivenDefaults:
    """Thresholds pulled from env/secrets when no explicit kwarg given."""

    def test_env_min_edge_z_filters(self, monkeypatch):
        """edge_z is a hard gate — entries below the env threshold are pruned."""
        monkeypatch.setenv("PRUNE_MIN_EDGE_Z", "3.0")
        result = prune_picks([_make_entry(edge_z=0.5)])
        assert len(result) == 0

    def test_env_min_ev_shrunk(self, monkeypatch):
        monkeypatch.setenv("PRUNE_MIN_EV_SHRUNK", "0.05")
        result = prune_picks([_make_entry(edge_ev_shrunk=0.03)])
        assert len(result) == 0

    def test_env_min_ev_shrunk_pass(self, monkeypatch):
        monkeypatch.setenv("PRUNE_MIN_EV_SHRUNK", "0.01")
        result = prune_picks([_make_entry(edge_ev_shrunk=0.03)])
        assert len(result) == 1

    def test_env_min_quality(self, monkeypatch):
        monkeypatch.setenv("PRUNE_MIN_QUALITY", "90")
        result = prune_picks([_make_entry(quality_score=85)])
        assert len(result) == 0

    def test_env_min_quality_pass(self, monkeypatch):
        monkeypatch.setenv("PRUNE_MIN_QUALITY", "80")
        result = prune_picks([_make_entry(quality_score=85)])
        assert len(result) == 1

    def test_env_min_books(self, monkeypatch):
        monkeypatch.setenv("PRUNE_MIN_BOOKS", "10")
        result = prune_picks([_make_entry(books_used=7)])
        assert len(result) == 0

    def test_env_min_books_pass(self, monkeypatch):
        monkeypatch.setenv("PRUNE_MIN_BOOKS", "5")
        result = prune_picks([_make_entry(books_used=7)])
        assert len(result) == 1

    def test_env_max_hold(self, monkeypatch):
        monkeypatch.setenv("PRUNE_MAX_HOLD", "4.0")
        result = prune_picks([_make_entry(market_hold_median=5.0)])
        assert len(result) == 0

    def test_env_max_hold_pass(self, monkeypatch):
        monkeypatch.setenv("PRUNE_MAX_HOLD", "6.0")
        result = prune_picks([_make_entry(market_hold_median=5.0)])
        assert len(result) == 1

    def test_env_allowed_tiers(self, monkeypatch):
        monkeypatch.setenv("PRUNE_ALLOWED_TIERS", "tier1a")
        result = prune_picks([_make_entry(tier="tier1b")])
        assert len(result) == 0

    def test_env_allowed_tiers_pass(self, monkeypatch):
        monkeypatch.setenv("PRUNE_ALLOWED_TIERS", "tier1a,tier1b")
        result = prune_picks([_make_entry(tier="tier1b")])
        assert len(result) == 1

    def test_env_allowed_alpha_labels(self, monkeypatch):
        monkeypatch.setenv("PRUNE_ALLOWED_ALPHA_LABELS", "Strong")
        result = prune_picks([_make_entry(alpha_label="Neutral")])
        assert len(result) == 0

    def test_env_allowed_alpha_labels_pass(self, monkeypatch):
        monkeypatch.setenv("PRUNE_ALLOWED_ALPHA_LABELS", "Strong,Neutral")
        result = prune_picks([_make_entry(alpha_label="Neutral")])
        assert len(result) == 1


class TestEdgeZHardGate:
    """edge_z is a hard pruning gate — entries below min_edge_z are pruned."""

    def test_low_edge_z_filtered(self, monkeypatch):
        monkeypatch.delenv("PRUNE_MIN_EDGE_Z", raising=False)
        result = prune_picks([_make_entry(edge_z=0.10)])
        assert len(result) == 0

    def test_zero_edge_z_filtered(self, monkeypatch):
        monkeypatch.delenv("PRUNE_MIN_EDGE_Z", raising=False)
        result = prune_picks([_make_entry(edge_z=0.0)])
        assert len(result) == 0


class TestExplicitKwargsOverrideConfig:
    """Explicit keyword arguments take precedence over env config."""

    def test_explicit_min_edge_z_filters(self, monkeypatch):
        """min_edge_z kwarg filters entries below the explicit threshold."""
        monkeypatch.setenv("PRUNE_MIN_EDGE_Z", "3.0")
        result = prune_picks([_make_entry(edge_z=0.5)], min_edge_z=5.0)
        assert len(result) == 0

    def test_explicit_allowed_tiers_overrides_env(self, monkeypatch):
        monkeypatch.setenv("PRUNE_ALLOWED_TIERS", "tier1a")
        result = prune_picks(
            [_make_entry(tier="tier1b")],
            allowed_tiers=frozenset({"tier1b"}),
        )
        assert len(result) == 1

    def test_explicit_allowed_alpha_overrides_env(self, monkeypatch):
        monkeypatch.setenv("PRUNE_ALLOWED_ALPHA_LABELS", "Strong")
        result = prune_picks(
            [_make_entry(alpha_label="Neutral")],
            allowed_alpha=frozenset({"Neutral"}),
        )
        assert len(result) == 1


class TestModelEdgeSourceBypassesEdgeZ:
    """Model-driven entries bypass edge_z and use edge_pct instead."""

    def test_model_entry_survives_low_edge_z(self, monkeypatch):
        """Model entry with low edge_z but high edge_pct survives pruning."""
        monkeypatch.delenv("PRUNE_MIN_EDGE_Z", raising=False)
        monkeypatch.delenv("MODEL_MIN_EDGE", raising=False)
        entry = _make_entry(edge_z=0.10, edge_pct=3.5, edge_source="model")
        result = prune_picks([entry])
        assert len(result) == 1

    def test_model_entry_pruned_when_edge_pct_below_threshold(self, monkeypatch):
        """Model entry with edge_pct below MODEL_MIN_EDGE is pruned."""
        monkeypatch.delenv("PRUNE_MIN_EDGE_Z", raising=False)
        monkeypatch.setenv("MODEL_MIN_EDGE", "3.0")
        entry = _make_entry(edge_z=0.10, edge_pct=1.5, edge_source="model")
        result = prune_picks([entry])
        assert len(result) == 0

    def test_consensus_entry_still_uses_edge_z(self, monkeypatch):
        """Non-model entries still use edge_z gating as before."""
        monkeypatch.delenv("PRUNE_MIN_EDGE_Z", raising=False)
        entry = _make_entry(edge_z=0.10, edge_pct=5.0, edge_source="consensus")
        result = prune_picks([entry])
        assert len(result) == 0

    def test_model_edge_reason_tracked(self, monkeypatch):
        """model_edge reason is tracked in the histogram."""
        monkeypatch.delenv("MODEL_MIN_EDGE", raising=False)
        entry = _make_entry(edge_z=0.10, edge_pct=0.5, edge_source="model")
        survivors, reasons = prune_picks_with_reasons([entry])
        assert len(survivors) == 0
        assert reasons["model_edge"] == 1
        assert reasons["edge_z"] == 0

    def test_mixed_entries_independent_gates(self, monkeypatch):
        """Model and consensus entries use their respective gates."""
        monkeypatch.delenv("PRUNE_MIN_EDGE_Z", raising=False)
        monkeypatch.delenv("MODEL_MIN_EDGE", raising=False)
        model_good = _make_entry(
            edge_z=0.10, edge_pct=3.0, edge_source="model", event_id="m1",
        )
        consensus_good = _make_entry(
            edge_z=2.0, edge_pct=1.0, edge_source="consensus", event_id="c1",
        )
        consensus_bad = _make_entry(
            edge_z=0.10, edge_pct=5.0, edge_source="consensus", event_id="c2",
        )
        survivors, reasons = prune_picks_with_reasons(
            [model_good, consensus_good, consensus_bad],
        )
        assert len(survivors) == 2
        assert reasons["edge_z"] == 1
        assert reasons["model_edge"] == 0

    def test_model_default_threshold_is_2(self, monkeypatch):
        """Default MODEL_MIN_EDGE is 2.0."""
        monkeypatch.delenv("MODEL_MIN_EDGE", raising=False)
        # edge_pct=1.9 should be pruned (below default 2.0)
        entry = _make_entry(edge_z=0.10, edge_pct=1.9, edge_source="model")
        result = prune_picks([entry])
        assert len(result) == 0
        # edge_pct=2.1 should survive
        entry2 = _make_entry(edge_z=0.10, edge_pct=2.1, edge_source="model")
        result2 = prune_picks([entry2])
        assert len(result2) == 1
