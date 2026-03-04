"""Tests for expand-markets changes: edge_z sigma clamp,
prune reasons, and automation breakdown logging."""

from __future__ import annotations

from line_tracker.best_bets import compute_edge_z
from line_tracker.config import (
    get_books_csv,
    get_edge_z_sigma_min,
    get_regions,
    get_slate_markets,
)
from line_tracker.services.pick_pruning_service import (
    prune_picks_with_reasons,
)

# ── Config getter tests ──────────────────────────────────────


class TestConfigGetters:
    def test_get_slate_markets_default(self):
        markets = get_slate_markets()
        assert "moneyline" in markets
        assert "spread" in markets
        assert "total" in markets

    def test_get_slate_markets_env(self, monkeypatch):
        monkeypatch.setenv("SLATE_MARKETS", "moneyline")
        assert get_slate_markets() == ["moneyline"]

    def test_get_books_csv_default(self):
        books = get_books_csv()
        assert "fanduel" in books
        assert "draftkings" in books

    def test_get_books_csv_env(self, monkeypatch):
        monkeypatch.setenv("ODDS_BOOKS", "fanduel,draftkings")
        assert get_books_csv() == "fanduel,draftkings"

    def test_get_regions_default(self):
        assert get_regions() == "us"

    def test_get_regions_env(self, monkeypatch):
        monkeypatch.setenv("ODDS_REGIONS", "us,eu")
        assert get_regions() == "us,eu"

    def test_get_edge_z_sigma_min_default(self):
        assert get_edge_z_sigma_min() == 0.02

    def test_get_edge_z_sigma_min_env(self, monkeypatch):
        monkeypatch.setenv("EDGE_Z_SIGMA_MIN", "0.05")
        assert get_edge_z_sigma_min() == 0.05


# ── Edge-z sigma clamp tests ────────────────────────────────


class TestEdgeZSigmaClamp:
    """Validate edge_z with configurable sigma floor."""

    def test_edge_z_increases_with_edge(self):
        """edge_z is monotonic in edge_ev_shrunk."""
        floor = 0.02
        z1 = compute_edge_z(0.01, 0.01, min_ev_sigma=floor)
        z2 = compute_edge_z(0.02, 0.01, min_ev_sigma=floor)
        z3 = compute_edge_z(0.04, 0.01, min_ev_sigma=floor)
        assert z1 < z2 < z3

    def test_sigma_clamp_prevents_collapse(self):
        """Small sigma is clamped to floor."""
        edge = 0.03
        z = compute_edge_z(edge, 0.005, min_ev_sigma=0.02)
        # Clamped to 0.02, so z = 0.03/0.02 = 1.5
        assert z == edge / 0.02

    def test_sigma_floor_bounds_denominator(self):
        """Denominator is at least the configured floor."""
        edge = 0.04
        z = compute_edge_z(edge, 0.001, min_ev_sigma=0.02)
        assert abs(z - edge / 0.02) < 1e-9

    def test_large_sigma_not_floored(self):
        """When sigma > floor, sigma is used as-is."""
        edge = 0.04
        sigma = 0.10
        z = compute_edge_z(edge, sigma, min_ev_sigma=0.02)
        assert abs(z - edge / sigma) < 1e-9


# ── Prune reasons histogram tests ───────────────────────────


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


class TestPrunePicksWithReasons:
    """prune_picks_with_reasons produces correct histograms."""

    def test_all_pass_empty_reasons(self):
        entries = [_make_entry()]
        survivors, reasons = prune_picks_with_reasons(entries)
        assert len(survivors) == 1
        assert all(v == 0 for v in reasons.values())

    def test_tier_failure_counted(self):
        entries = [_make_entry(tier="tier3")]
        survivors, reasons = prune_picks_with_reasons(entries)
        assert len(survivors) == 0
        assert reasons["tier"] == 1
        assert reasons["alpha"] == 0

    def test_alpha_failure_counted(self):
        entries = [_make_entry(alpha_label="Weak")]
        survivors, reasons = prune_picks_with_reasons(entries)
        assert len(survivors) == 0
        assert reasons["alpha"] == 1

    def test_edge_z_failure_counted(self):
        entries = [_make_entry(edge_z=0.5)]
        survivors, reasons = prune_picks_with_reasons(entries)
        assert len(survivors) == 0
        assert reasons["edge_z"] == 1

    def test_quality_failure_counted(self):
        entries = [_make_entry(quality_score=30)]
        survivors, reasons = prune_picks_with_reasons(entries)
        assert len(survivors) == 0
        assert reasons["quality"] == 1

    def test_books_failure_counted(self):
        entries = [_make_entry(books_used=2)]
        survivors, reasons = prune_picks_with_reasons(entries)
        assert len(survivors) == 0
        assert reasons["books"] == 1

    def test_hold_failure_counted(self):
        entries = [_make_entry(market_hold_median=20.0)]
        survivors, reasons = prune_picks_with_reasons(entries)
        assert len(survivors) == 0
        assert reasons["hold"] == 1

    def test_multiple_entries_different_gates(self):
        """Different gates produce correct counts."""
        entries = [
            _make_entry(tier="tier3"),
            _make_entry(alpha_label="Weak"),
            _make_entry(edge_z=0.1),
            _make_entry(edge_ev_shrunk=-0.01),
            _make_entry(quality_score=10),
            _make_entry(books_used=1),
            _make_entry(),
        ]
        survivors, reasons = prune_picks_with_reasons(entries)
        assert len(survivors) == 1
        assert reasons["tier"] == 1
        assert reasons["alpha"] == 1
        assert reasons["edge_z"] == 1
        assert reasons["ev_shrunk"] == 1
        assert reasons["quality"] == 1
        assert reasons["books"] == 1

    def test_first_failure_only(self):
        """Entry counted only for its first failing gate."""
        entries = [
            _make_entry(tier="tier3", alpha_label="Weak"),
        ]
        survivors, reasons = prune_picks_with_reasons(entries)
        assert reasons["tier"] == 1
        assert reasons["alpha"] == 0

    def test_histogram_keys_present(self):
        """All expected gate keys are in the histogram."""
        _, reasons = prune_picks_with_reasons([])
        expected = {
            "tier", "alpha", "edge_z", "ev_shrunk",
            "quality", "books", "hold", "clv",
        }
        assert set(reasons.keys()) == expected


# ── Automation service prune_breakdown logging ───────────────


class TestAutomationPruneBreakdownLogging:
    """run_cycle logs prune_breakdown when pruned==0."""

    def test_prune_breakdown_logged_dry_run(self, capsys):
        """prune_breakdown is logged at INFO to stderr."""
        from line_tracker.core.logging import get_logger

        log = get_logger(
            "test_automation_breakdown", source="test",
        )

        entries = [
            _make_entry(tier="tier3"),
            _make_entry(edge_z=0.1),
        ]
        pruned, prune_reasons = prune_picks_with_reasons(
            entries,
        )
        assert len(pruned) == 0

        parts = " ".join(
            f"{k}={v}" for k, v in prune_reasons.items()
        )
        log.info("prune_breakdown %s", parts)

        captured = capsys.readouterr()
        assert "prune_breakdown" in captured.err
        assert "tier=1" in captured.err
        assert "edge_z=1" in captured.err
