"""Tests for Step 2 – Aggressive pick pruning."""

from __future__ import annotations

import pytest

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


class TestPrunePicks:
    def test_passing_entry_survives(self):
        result = prune_picks([_make_entry()])
        assert len(result) == 1

    def test_tier3_allowed(self):
        result = prune_picks([_make_entry(tier="tier3")])
        assert len(result) == 1

    def test_avoid_filtered(self):
        result = prune_picks([_make_entry(tier="avoid")])
        assert len(result) == 0

    def test_weak_alpha_allowed(self):
        result = prune_picks([_make_entry(alpha_label="Weak")])
        assert len(result) == 1

    def test_unknown_alpha_filtered(self):
        result = prune_picks([_make_entry(alpha_label="Unknown")])
        assert len(result) == 0

    def test_empty_alpha_filtered(self):
        result = prune_picks([_make_entry(alpha_label="")])
        assert len(result) == 0

    def test_low_edge_z_filtered(self):
        result = prune_picks([_make_entry(edge_z=1.0)])
        assert len(result) == 0

    def test_negative_ev_shrunk_filtered(self):
        result = prune_picks([_make_entry(edge_ev_shrunk=-0.01)])
        assert len(result) == 0

    def test_zero_ev_shrunk_filtered(self):
        result = prune_picks([_make_entry(edge_ev_shrunk=0)])
        assert len(result) == 0

    def test_low_quality_filtered(self):
        result = prune_picks([_make_entry(quality_score=40)])
        assert len(result) == 0

    def test_few_books_filtered(self):
        result = prune_picks([_make_entry(books_used=3)])
        assert len(result) == 0

    def test_high_hold_filtered(self):
        result = prune_picks([_make_entry(market_hold_median=8.5)])
        assert len(result) == 0

    def test_clv_filter_blocks(self):
        profile = {
            "groups": {
                ("spread", "tier1b", "Strong"): {"passes": False},
            },
        }
        result = prune_picks([_make_entry()], clv_profile=profile)
        assert len(result) == 0

    def test_clv_filter_passes(self):
        profile = {
            "groups": {
                ("spread", "tier1b", "Strong"): {"passes": True},
            },
        }
        result = prune_picks([_make_entry()], clv_profile=profile)
        assert len(result) == 1

    def test_multiple_entries_mixed(self):
        entries = [
            _make_entry(event_id="e1"),
            _make_entry(event_id="e2", tier="avoid"),
            _make_entry(event_id="e3", edge_z=1.0),
            _make_entry(event_id="e4"),
        ]
        result = prune_picks(entries)
        assert len(result) == 2
        assert {e["event_id"] for e in result} == {"e1", "e4"}

    def test_tier1a_tier2_tier3_allowed(self):
        entries = [
            _make_entry(tier="tier1a"),
            _make_entry(tier="tier2"),
            _make_entry(tier="tier3"),
        ]
        result = prune_picks(entries)
        assert len(result) == 3

    def test_neutral_alpha_allowed(self):
        result = prune_picks([_make_entry(alpha_label="Neutral")])
        assert len(result) == 1

    def test_custom_thresholds(self):
        result = prune_picks(
            [_make_entry(edge_z=1.2)],
            min_edge_z=1.0,
        )
        assert len(result) == 1


class TestAlphaLabelNormalization:
    """Verify that alpha_label comparison is case-insensitive."""

    @pytest.mark.parametrize(
        "label",
        ["weak", "WEAK", "Weak", " weak ", "NEUTRAL", "neutral", "Neutral",
         "strong", "STRONG", "Strong"],
    )
    def test_case_variants_pass(self, label):
        """All case variants of valid labels should survive the alpha gate."""
        result = prune_picks([_make_entry(alpha_label=label)])
        assert len(result) == 1, f"alpha_label={label!r} was rejected"

    @pytest.mark.parametrize("label", ["weak", "WEAK", " Weak "])
    def test_case_variants_no_alpha_rejection(self, label):
        """Ensure zero alpha rejections for case-variant labels."""
        _, reasons = prune_picks_with_reasons([_make_entry(alpha_label=label)])
        assert reasons["alpha"] == 0, (
            f"alpha_label={label!r} caused alpha rejection"
        )

    @pytest.mark.parametrize("label", [None, "", "  ", "Unknown", "invalid"])
    def test_invalid_labels_rejected(self, label):
        """None, empty, whitespace-only, and unknown labels must be blocked."""
        result = prune_picks([_make_entry(alpha_label=label)])
        assert len(result) == 0, f"alpha_label={label!r} should be rejected"

    def test_lowercase_env_var_config_normalised(self, monkeypatch):
        """Config with lowercase env var should still produce title-cased set."""
        monkeypatch.setenv("PRUNE_ALLOWED_ALPHA_LABELS", "strong,neutral,weak")
        from line_tracker.config import get_prune_allowed_alpha_labels

        allowed = get_prune_allowed_alpha_labels()
        assert allowed == frozenset({"Strong", "Neutral", "Weak"})

    def test_mixed_case_env_var_config_normalised(self, monkeypatch):
        """Config with UPPER env var should still produce title-cased set."""
        monkeypatch.setenv("PRUNE_ALLOWED_ALPHA_LABELS", "STRONG,NEUTRAL")
        from line_tracker.config import get_prune_allowed_alpha_labels

        allowed = get_prune_allowed_alpha_labels()
        assert allowed == frozenset({"Strong", "Neutral"})
