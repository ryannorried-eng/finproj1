"""Tests for Pro Hybrid confidence-weighted market weighting."""

import pytest

from line_tracker.scoring import (
    compute_hybrid_fields,
    confidence_factor,
    enrich_entry,
    hybrid_market_weight,
    market_weight,
    rank_candidates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    *,
    books_used: int = 7,
    market_hold_median: float = 5.0,
    edge_z: float = 2.0,
    edge_ev_shrunk: float = 0.02,
    agreement_score: float = 85.0,
    market_volatility_sigma: float = 0.005,
    consensus_prob: float = 0.55,
    kelly_suggested: float = 0.02,
    quality_score: int = 80,
    quality_tier: str = "Strong",
    ev_100: float = 3.0,
    alpha_score: int | None = None,
    best_odds: float | None = None,
    market: str | None = None,
    confidence_label: str | None = None,
) -> dict:
    """Build a minimal entry dict for scoring tests."""
    d: dict = {
        "books_used": books_used,
        "market_hold_median": market_hold_median,
        "edge_z": edge_z,
        "edge_ev_shrunk": edge_ev_shrunk,
        "agreement_score": agreement_score,
        "market_volatility_sigma": market_volatility_sigma,
        "consensus_prob": consensus_prob,
        "kelly_suggested": kelly_suggested,
        "quality_score": quality_score,
        "quality_tier": quality_tier,
        "ev_100": ev_100,
    }
    if alpha_score is not None:
        d["alpha_score"] = alpha_score
    if best_odds is not None:
        d["best_odds"] = best_odds
    if market is not None:
        d["market"] = market
    if confidence_label is not None:
        d["confidence_label"] = confidence_label
    return d


# ── confidence_factor mapping ─────────────────────────────────────────


class TestConfidenceFactorMapping:
    """Test confidence_factor returns the correct multiplier per label."""

    def test_high(self):
        assert confidence_factor("High") == pytest.approx(1.15)

    def test_medium(self):
        assert confidence_factor("Medium") == pytest.approx(1.00)

    def test_low(self):
        assert confidence_factor("Low") == pytest.approx(0.85)

    def test_unknown_defaults_to_medium(self):
        assert confidence_factor("Unknown") == pytest.approx(1.00)

    def test_empty_string_defaults_to_medium(self):
        assert confidence_factor("") == pytest.approx(1.00)


# ── hybrid_market_weight with flag OFF ────────────────────────────────


class TestHybridMarketWeightFlagOff:
    """When PRO_HYBRID_MARKET_CONF is off, hybrid_market_weight must
    return exactly the base market_weight."""

    def test_spread_flag_off(self, monkeypatch):
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "0")
        e = _entry(market="spread", confidence_label="High")
        assert hybrid_market_weight(e) == market_weight(e)

    def test_ml_dog_flag_off(self, monkeypatch):
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "0")
        e = _entry(market="h2h", best_odds=250, consensus_prob=0.30,
                   confidence_label="Low")
        assert hybrid_market_weight(e) == market_weight(e)

    def test_totals_flag_off(self, monkeypatch):
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "0")
        e = _entry(market="totals", confidence_label="Medium")
        assert hybrid_market_weight(e) == market_weight(e)


# ── hybrid_market_weight with flag ON ─────────────────────────────────


class TestHybridMarketWeightFlagOn:
    """When PRO_HYBRID_MARKET_CONF is on, hybrid_market_weight applies
    confidence_factor and clamps."""

    def test_spread_high_confidence(self, monkeypatch):
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "1")
        e = _entry(market="spread", confidence_label="High")
        base = market_weight(e)  # 1.00
        expected = min(1.10, max(0.50, base * 1.15))  # 1.15 → clamped to 1.10
        assert hybrid_market_weight(e) == pytest.approx(expected, abs=1e-6)

    def test_spread_low_confidence(self, monkeypatch):
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "1")
        e = _entry(market="spread", confidence_label="Low")
        base = market_weight(e)  # 1.00
        expected = base * 0.85  # 0.85 — within [0.50, 1.10]
        assert hybrid_market_weight(e) == pytest.approx(expected, abs=1e-6)

    def test_ml_dog_high_confidence(self, monkeypatch):
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "1")
        e = _entry(market="h2h", best_odds=250, consensus_prob=0.30,
                   confidence_label="High")
        base = market_weight(e)  # 0.75
        expected = base * 1.15  # 0.8625 — within [0.50, 1.10]
        assert hybrid_market_weight(e) == pytest.approx(expected, abs=1e-6)

    def test_clamps_to_min(self, monkeypatch):
        """A very low base weight * low confidence should clamp at CONF_W_MIN."""
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "1")
        monkeypatch.setenv("CONF_W_MIN", "0.80")  # raise min to force clamp
        e = _entry(market="h2h", best_odds=250, consensus_prob=0.30,
                   confidence_label="Low")
        # base=0.75 * 0.85 = 0.6375 < min 0.80 → clamped
        assert hybrid_market_weight(e) == pytest.approx(0.80, abs=1e-6)

    def test_clamps_to_max(self, monkeypatch):
        """A high base weight * high confidence should clamp at CONF_W_MAX."""
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "1")
        e = _entry(market="spread", confidence_label="High")
        # base=1.00 * 1.15 = 1.15 > max 1.10 → clamped
        assert hybrid_market_weight(e) == pytest.approx(1.10, abs=1e-6)

    def test_medium_confidence_unchanged(self, monkeypatch):
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "1")
        e = _entry(market="spread", confidence_label="Medium")
        base = market_weight(e)  # 1.00
        expected = base * 1.00  # 1.00 — within [0.50, 1.10]
        assert hybrid_market_weight(e) == pytest.approx(expected, abs=1e-6)


# ── hybrid_score uses hybrid_market_weight when flag on ───────────────


class TestHybridScoreUsesHybridMarketWeight:
    """compute_hybrid_fields must use hybrid_market_weight for
    hybrid_score, but keep hybrid_score_raw unchanged."""

    def test_raw_unchanged_flag_on(self, monkeypatch):
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "1")
        e = _entry(alpha_score=80, kelly_suggested=0.025, consensus_prob=0.55,
                   market="spread", confidence_label="High")
        result = compute_hybrid_fields(e)
        # raw is independent of market weight
        assert result["hybrid_score_raw"] == pytest.approx(0.63, abs=0.001)

    def test_raw_unchanged_flag_off(self, monkeypatch):
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "0")
        e = _entry(alpha_score=80, kelly_suggested=0.025, consensus_prob=0.55,
                   market="spread", confidence_label="High")
        result = compute_hybrid_fields(e)
        assert result["hybrid_score_raw"] == pytest.approx(0.63, abs=0.001)

    def test_score_differs_when_flag_on(self, monkeypatch):
        """With flag on + non-medium confidence, hybrid_score should differ
        from raw * base_market_weight."""
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "1")
        e = _entry(alpha_score=80, kelly_suggested=0.025, consensus_prob=0.55,
                   market="h2h", best_odds=250, confidence_label="High")
        result = compute_hybrid_fields(e)
        raw = result["hybrid_score_raw"]
        base_mw = result["market_weight"]
        eff_mw = result["effective_market_weight"]
        assert base_mw != eff_mw
        assert result["hybrid_score"] == pytest.approx(raw * eff_mw, abs=1e-4)

    def test_score_equals_raw_times_base_when_flag_off(self, monkeypatch):
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "0")
        e = _entry(alpha_score=80, kelly_suggested=0.025, consensus_prob=0.55,
                   market="spread", confidence_label="Low")
        result = compute_hybrid_fields(e)
        raw = result["hybrid_score_raw"]
        base_mw = result["market_weight"]
        assert result["hybrid_score"] == pytest.approx(raw * base_mw, abs=1e-4)

    def test_effective_market_weight_in_output(self, monkeypatch):
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "1")
        e = _entry(market="spread", confidence_label="High", alpha_score=70)
        result = compute_hybrid_fields(e)
        assert "effective_market_weight" in result
        assert "effective_market_weight" in result["hybrid_components"]


# ── Hit and value rankings unchanged with flag on ─────────────────────


class TestHitAndValueRankingsUnchanged:
    """PRO_HYBRID_MARKET_CONF must NOT affect hit or value ranking."""

    def _make_candidates(self):
        """Two candidates: high_ev (good EV, low prob) and high_prob
        (low EV, high prob)."""
        high_ev = _entry(
            edge_ev_shrunk=0.04, ev_100=4.0,
            consensus_prob=0.25, agreement_score=70,
            market_volatility_sigma=0.012, market="h2h",
            best_odds=300, confidence_label="Low",
        )
        high_prob = _entry(
            edge_ev_shrunk=0.01, ev_100=1.0,
            consensus_prob=0.70, agreement_score=90,
            market_volatility_sigma=0.003, market="spread",
            confidence_label="High",
        )
        for e in [high_ev, high_prob]:
            enrich_entry(e)
        return high_ev, high_prob

    def test_hit_ranking_unchanged(self, monkeypatch):
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "1")
        high_ev, high_prob = self._make_candidates()
        candidates = [high_ev, high_prob]
        rank_candidates(candidates, mode="hit")
        assert candidates[0] is high_prob  # prob-first

    def test_value_ranking_unchanged(self, monkeypatch):
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "1")
        high_ev, high_prob = self._make_candidates()
        candidates = [high_prob, high_ev]
        rank_candidates(candidates, mode="value")
        assert candidates[0] is high_ev  # EV-first

    def test_hit_sort_key_ignores_market_weight(self, monkeypatch):
        """Hit sort key does not reference hybrid_score at all."""
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "1")
        e = _entry(consensus_prob=0.65, confidence_label="High",
                   market="spread")
        enrich_entry(e)
        from line_tracker.scoring import _hit_sort_key
        key = _hit_sort_key(e)
        # First element is -above_floor, second is -prob — no mw influence
        assert key[1] == pytest.approx(-0.65)

    def test_value_sort_key_ignores_market_weight(self, monkeypatch):
        """Value sort key does not reference hybrid_score at all."""
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "1")
        e = _entry(edge_ev_shrunk=0.04, ev_100=4.0, confidence_label="High",
                   market="spread")
        enrich_entry(e)
        from line_tracker.scoring import _value_sort_key
        key = _value_sort_key(e)
        assert key[0] == pytest.approx(-0.04)


# ── High confidence ML dog less penalized than low confidence ─────────


class TestHighConfMLDogLessPenalized:
    """When the flag is on, a high-confidence ML underdog should get a
    higher effective market weight than a low-confidence ML underdog."""

    def test_high_conf_dog_less_penalized(self, monkeypatch):
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "1")
        high_conf = _entry(
            market="h2h", best_odds=250, consensus_prob=0.30,
            confidence_label="High",
        )
        low_conf = _entry(
            market="h2h", best_odds=250, consensus_prob=0.30,
            confidence_label="Low",
        )
        mw_high = hybrid_market_weight(high_conf)
        mw_low = hybrid_market_weight(low_conf)
        assert mw_high > mw_low

    def test_same_base_weight(self, monkeypatch):
        """Both dogs should have the same base market_weight."""
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "1")
        high_conf = _entry(
            market="h2h", best_odds=250, consensus_prob=0.30,
            confidence_label="High",
        )
        low_conf = _entry(
            market="h2h", best_odds=250, consensus_prob=0.30,
            confidence_label="Low",
        )
        assert market_weight(high_conf) == market_weight(low_conf)

    def test_hybrid_score_reflects_confidence(self, monkeypatch):
        """The hybrid_score for the high-confidence dog should be higher."""
        monkeypatch.setenv("PRO_HYBRID_MARKET_CONF", "1")
        high_conf = _entry(
            market="h2h", best_odds=250, consensus_prob=0.30,
            confidence_label="High", alpha_score=70,
            kelly_suggested=0.02,
        )
        low_conf = _entry(
            market="h2h", best_odds=250, consensus_prob=0.30,
            confidence_label="Low", alpha_score=70,
            kelly_suggested=0.02,
        )
        h_result = compute_hybrid_fields(high_conf)
        l_result = compute_hybrid_fields(low_conf)
        # Same raw
        assert h_result["hybrid_score_raw"] == l_result["hybrid_score_raw"]
        # Higher effective weight → higher final score
        assert h_result["hybrid_score"] > l_result["hybrid_score"]
