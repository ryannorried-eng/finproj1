"""Tests for CLV selection service – min_samples fallback logic."""

from __future__ import annotations

from line_tracker.services.clv_selection_service import (
    DEFAULT_MIN_SAMPLE,
    build_clv_filter_profile,
)


class _FakeStore:
    """Minimal stand-in for LineStore that returns pre-loaded snapshots."""

    def __init__(self, snapshots: list[dict] | None = None):
        self._closed = snapshots or []

    def get_closed_snapshots(self) -> list[dict]:
        return list(self._closed)


def _snap(market="spread", tier="tier1b", alpha="Strong", clv=0.02):
    return {
        "market": market,
        "tier": tier,
        "alpha_label": alpha,
        "clv_delta_implied": clv,
    }


class TestMinSamplesFallback:
    """Verify the (market,tier,alpha) → (market,tier) → (market) → global chain."""

    def test_no_fallback_when_enough_samples(self):
        snaps = [_snap() for _ in range(DEFAULT_MIN_SAMPLE)]
        profile = build_clv_filter_profile(_FakeStore(snaps))
        key = ("spread", "tier1b", "Strong")
        group = profile["groups"][key]
        assert group["count"] == DEFAULT_MIN_SAMPLE
        assert group["fallback_source"] == "market_tier_alpha"

    def test_fallback_to_market_tier(self):
        """Fine bucket has 2 samples, (market, tier) has 6 (2 + 4)."""
        snaps = [_snap(alpha="Strong", clv=0.01) for _ in range(2)]
        snaps += [_snap(alpha="Weak", clv=0.02) for _ in range(4)]
        profile = build_clv_filter_profile(_FakeStore(snaps), min_sample=5)

        # "Strong" bucket has only 2, so it should fall back to (spread, tier1b)
        key_strong = ("spread", "tier1b", "Strong")
        g = profile["groups"][key_strong]
        assert g["fallback_source"] == "market_tier"
        assert g["count"] == 6  # aggregated (market, tier) count

    def test_fallback_to_market(self):
        """Both fine and (market, tier) are below threshold; (market,) is enough."""
        snaps = [_snap(tier="tierA", alpha="A", clv=0.01) for _ in range(2)]
        snaps += [_snap(tier="tierB", alpha="B", clv=0.03) for _ in range(2)]
        snaps += [_snap(tier="tierC", alpha="C", clv=0.02) for _ in range(2)]
        # All share market="spread"; each (market, tier) has only 2.
        # (market,) has 6 total.
        profile = build_clv_filter_profile(_FakeStore(snaps), min_sample=5)

        key = ("spread", "tierA", "A")
        g = profile["groups"][key]
        assert g["fallback_source"] == "market"
        assert g["count"] == 6

    def test_fallback_to_global(self):
        """Every sub-bucket and every market bucket is too small; global used."""
        snaps = [
            _snap(market="spread", tier="t1", alpha="A", clv=0.01),
            _snap(market="total", tier="t2", alpha="B", clv=0.02),
            _snap(market="moneyline", tier="t3", alpha="C", clv=0.03),
            _snap(market="props", tier="t4", alpha="D", clv=0.04),
            _snap(market="alt", tier="t5", alpha="E", clv=0.05),
        ]
        profile = build_clv_filter_profile(_FakeStore(snaps), min_sample=5)

        key = ("spread", "t1", "A")
        g = profile["groups"][key]
        assert g["fallback_source"] == "global"
        assert g["count"] == 5  # all snapshots combined

    def test_fallback_passes_uses_coarse_stats(self):
        """Coarse bucket positive rate decides 'passes' when fallback fires."""
        # 3 positive + 3 negative at market level → 50% positive → fails at 55%
        snaps = [_snap(tier="t1", alpha="X", clv=0.05) for _ in range(3)]
        snaps += [_snap(tier="t2", alpha="Y", clv=-0.05) for _ in range(3)]
        profile = build_clv_filter_profile(
            _FakeStore(snaps), min_sample=5, min_pct_positive=55.0,
        )
        # Each fine bucket has 3 → below min_sample 5 → falls to (market,) with 6
        key = ("spread", "t1", "X")
        g = profile["groups"][key]
        assert g["fallback_source"] == "market"
        assert g["passes"] is False  # 50% < 55%

    def test_fallback_passes_when_coarse_pct_enough(self):
        """Coarse bucket has enough positive rate → passes."""
        snaps = [_snap(tier="t1", alpha="X", clv=0.05) for _ in range(4)]
        snaps += [_snap(tier="t2", alpha="Y", clv=-0.05) for _ in range(1)]
        profile = build_clv_filter_profile(
            _FakeStore(snaps), min_sample=5, min_pct_positive=55.0,
        )
        key = ("spread", "t1", "X")
        g = profile["groups"][key]
        assert g["fallback_source"] == "market"
        assert g["passes"] is True  # 80% >= 55%

    def test_global_fallback_below_min_sample(self):
        """When even global count < min_sample, passes is False."""
        snaps = [_snap(clv=0.05) for _ in range(3)]
        profile = build_clv_filter_profile(_FakeStore(snaps), min_sample=10)
        key = ("spread", "tier1b", "Strong")
        g = profile["groups"][key]
        assert g["fallback_source"] == "global"
        assert g["passes"] is False

    def test_empty_store_unchanged(self):
        profile = build_clv_filter_profile(_FakeStore([]))
        assert profile == {"groups": {}, "total_closed": 0}

    def test_fallback_source_stored_for_every_group(self):
        """Every group entry has a 'fallback_source' key."""
        snaps = [_snap() for _ in range(10)]
        snaps += [_snap(alpha="Weak") for _ in range(2)]
        profile = build_clv_filter_profile(_FakeStore(snaps), min_sample=5)
        for key, g in profile["groups"].items():
            assert "fallback_source" in g, f"missing fallback_source for {key}"
