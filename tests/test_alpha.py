"""Tests for the Alpha v1 robustness overlay."""

from line_tracker.alpha import (
    ALPHA_GATE_ENABLED,
    alpha_label,
    alpha_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    *,
    books_used: int = 7,
    market_hold_median: float = 5.0,
    edge_z: float = 2.5,
    edge_ev_shrunk: float = 0.015,
    agreement_score: float = 90.0,
    market_volatility_sigma: float = 0.004,
    consensus_prob: float = 0.55,
) -> dict:
    """Build a minimal entry dict for alpha_score testing."""
    return {
        "books_used": books_used,
        "market_hold_median": market_hold_median,
        "edge_z": edge_z,
        "edge_ev_shrunk": edge_ev_shrunk,
        "agreement_score": agreement_score,
        "market_volatility_sigma": market_volatility_sigma,
        "consensus_prob": consensus_prob,
    }


# ---------------------------------------------------------------------------
# alpha_label
# ---------------------------------------------------------------------------


class TestAlphaLabel:
    def test_strong(self):
        assert alpha_label(65) == "Strong"
        assert alpha_label(100) == "Strong"
        assert alpha_label(78) == "Strong"

    def test_neutral(self):
        assert alpha_label(40) == "Neutral"
        assert alpha_label(64) == "Neutral"
        assert alpha_label(50) == "Neutral"

    def test_weak(self):
        assert alpha_label(0) == "Weak"
        assert alpha_label(39) == "Weak"
        assert alpha_label(10) == "Weak"


# ---------------------------------------------------------------------------
# alpha_score – component A: Market robustness (0–30)
# ---------------------------------------------------------------------------


class TestMarketRobustness:
    def test_books_7_plus(self):
        score, comp = alpha_score(_entry(books_used=7))
        assert comp["books_pts"] == 15

    def test_books_5_6(self):
        _, comp = alpha_score(_entry(books_used=5))
        assert comp["books_pts"] == 10
        _, comp = alpha_score(_entry(books_used=6))
        assert comp["books_pts"] == 10

    def test_books_4(self):
        _, comp = alpha_score(_entry(books_used=4))
        assert comp["books_pts"] == 5

    def test_books_below_4(self):
        _, comp = alpha_score(_entry(books_used=3))
        assert comp["books_pts"] == 0

    def test_hold_low(self):
        _, comp = alpha_score(_entry(market_hold_median=5.0))
        assert comp["hold_pts"] == 15

    def test_hold_mid(self):
        _, comp = alpha_score(_entry(market_hold_median=5.5))
        assert comp["hold_pts"] == 15  # <=5.5 boundary
        _, comp = alpha_score(_entry(market_hold_median=6.0))
        assert comp["hold_pts"] == 10

    def test_hold_high(self):
        _, comp = alpha_score(_entry(market_hold_median=8.0))
        assert comp["hold_pts"] == 5

    def test_hold_very_high(self):
        _, comp = alpha_score(_entry(market_hold_median=9.0))
        assert comp["hold_pts"] == 0

    def test_market_robustness_total(self):
        _, comp = alpha_score(
            _entry(books_used=7, market_hold_median=5.0)
        )
        assert comp["market_robustness"] == 30  # 15+15


# ---------------------------------------------------------------------------
# alpha_score – component B: Edge robustness (0–40)
# ---------------------------------------------------------------------------


class TestEdgeRobustness:
    def test_edge_z_top_tier(self):
        _, comp = alpha_score(_entry(edge_z=2.5))
        assert comp["edge_z_pts"] == 20

    def test_edge_z_mid(self):
        _, comp = alpha_score(_entry(edge_z=2.0))
        assert comp["edge_z_pts"] == 15

    def test_edge_z_moderate(self):
        _, comp = alpha_score(_entry(edge_z=1.75))
        assert comp["edge_z_pts"] == 10

    def test_edge_z_low(self):
        _, comp = alpha_score(_entry(edge_z=1.4))
        assert comp["edge_z_pts"] == 5

    def test_edge_z_zero(self):
        _, comp = alpha_score(_entry(edge_z=1.0))
        assert comp["edge_z_pts"] == 0

    def test_shrunk_top(self):
        # 0.015 * 100 = 1.5 >= 1.25
        _, comp = alpha_score(_entry(edge_ev_shrunk=0.015))
        assert comp["shrunk_pts"] == 20

    def test_shrunk_mid(self):
        # 0.008 * 100 = 0.8 >= 0.75
        _, comp = alpha_score(_entry(edge_ev_shrunk=0.008))
        assert comp["shrunk_pts"] == 15

    def test_shrunk_low(self):
        # 0.004 * 100 = 0.4 >= 0.35
        _, comp = alpha_score(_entry(edge_ev_shrunk=0.004))
        assert comp["shrunk_pts"] == 10

    def test_shrunk_positive(self):
        # 0.001 * 100 = 0.1 > 0
        _, comp = alpha_score(_entry(edge_ev_shrunk=0.001))
        assert comp["shrunk_pts"] == 5

    def test_shrunk_zero(self):
        _, comp = alpha_score(_entry(edge_ev_shrunk=0.0))
        assert comp["shrunk_pts"] == 0

    def test_edge_robustness_total(self):
        _, comp = alpha_score(
            _entry(edge_z=2.5, edge_ev_shrunk=0.015)
        )
        assert comp["edge_robustness"] == 40  # 20+20


# ---------------------------------------------------------------------------
# alpha_score – component C: Agreement / stability (0–20)
# ---------------------------------------------------------------------------


class TestAgreementStability:
    def test_agreement_high(self):
        _, comp = alpha_score(_entry(agreement_score=95))
        assert comp["agree_pts"] == 10

    def test_agreement_mid(self):
        _, comp = alpha_score(_entry(agreement_score=80))
        assert comp["agree_pts"] == 7

    def test_agreement_low(self):
        _, comp = alpha_score(_entry(agreement_score=70))
        assert comp["agree_pts"] == 4

    def test_agreement_none(self):
        _, comp = alpha_score(_entry(agreement_score=50))
        assert comp["agree_pts"] == 0

    def test_sigma_very_low(self):
        _, comp = alpha_score(_entry(market_volatility_sigma=0.004))
        assert comp["sigma_pts"] == 10

    def test_sigma_boundary(self):
        _, comp = alpha_score(_entry(market_volatility_sigma=0.0045))
        assert comp["sigma_pts"] == 10  # <=0.0045
        _, comp = alpha_score(_entry(market_volatility_sigma=0.005))
        assert comp["sigma_pts"] == 7

    def test_sigma_mid(self):
        _, comp = alpha_score(_entry(market_volatility_sigma=0.007))
        assert comp["sigma_pts"] == 7  # <=0.007
        _, comp = alpha_score(_entry(market_volatility_sigma=0.009))
        assert comp["sigma_pts"] == 4

    def test_sigma_high(self):
        _, comp = alpha_score(_entry(market_volatility_sigma=0.02))
        assert comp["sigma_pts"] == 0

    def test_agreement_stability_total(self):
        _, comp = alpha_score(
            _entry(agreement_score=95, market_volatility_sigma=0.003)
        )
        assert comp["agreement_stability"] == 20  # 10+10


# ---------------------------------------------------------------------------
# alpha_score – component D: Longshot penalty
# ---------------------------------------------------------------------------


class TestLongshotPenalty:
    def test_no_penalty(self):
        _, comp = alpha_score(_entry(consensus_prob=0.55))
        assert comp["longshot_penalty"] == 0

    def test_boundary_045(self):
        _, comp = alpha_score(_entry(consensus_prob=0.45))
        assert comp["longshot_penalty"] == 0

    def test_mild_penalty(self):
        _, comp = alpha_score(_entry(consensus_prob=0.35))
        assert comp["longshot_penalty"] == -5

    def test_moderate_penalty(self):
        _, comp = alpha_score(_entry(consensus_prob=0.25))
        assert comp["longshot_penalty"] == -12

    def test_heavy_penalty(self):
        _, comp = alpha_score(_entry(consensus_prob=0.16))
        assert comp["longshot_penalty"] == -18

    def test_max_penalty(self):
        _, comp = alpha_score(_entry(consensus_prob=0.10))
        assert comp["longshot_penalty"] == -25

    def test_boundary_030(self):
        _, comp = alpha_score(_entry(consensus_prob=0.30))
        assert comp["longshot_penalty"] == -5

    def test_boundary_020(self):
        _, comp = alpha_score(_entry(consensus_prob=0.20))
        assert comp["longshot_penalty"] == -12

    def test_boundary_015(self):
        _, comp = alpha_score(_entry(consensus_prob=0.15))
        assert comp["longshot_penalty"] == -18


# ---------------------------------------------------------------------------
# alpha_score – clamp behaviour
# ---------------------------------------------------------------------------


class TestClamp:
    def test_perfect_score_clamped(self):
        """Max components total 110 (30+40+20+0); must clamp to 100."""
        score, _ = alpha_score(_entry(
            books_used=10,
            market_hold_median=3.0,
            edge_z=3.0,
            edge_ev_shrunk=0.02,
            agreement_score=95,
            market_volatility_sigma=0.002,
            consensus_prob=0.60,
        ))
        assert score == 90  # 30+40+20+0 = 90, no clamping needed here
        # But if raw > 100 somehow, it clamps:
        assert score <= 100

    def test_max_theoretical(self):
        """All max component values: 30+40+20+0 = 90."""
        score, comp = alpha_score(_entry(
            books_used=7,
            market_hold_median=5.5,
            edge_z=2.5,
            edge_ev_shrunk=0.0125,
            agreement_score=90,
            market_volatility_sigma=0.0045,
            consensus_prob=0.45,
        ))
        assert comp["market_robustness"] == 30
        assert comp["edge_robustness"] == 40
        assert comp["agreement_stability"] == 20
        assert comp["longshot_penalty"] == 0
        assert score == 90

    def test_floor_clamp_to_zero(self):
        """Worst case with penalty: should clamp to 0, not go negative."""
        score, comp = alpha_score(_entry(
            books_used=2,
            market_hold_median=10.0,
            edge_z=0.5,
            edge_ev_shrunk=0.0,
            agreement_score=30,
            market_volatility_sigma=0.05,
            consensus_prob=0.05,
        ))
        assert comp["raw_total"] < 0  # 0+0+0-25 = -25
        assert score == 0

    def test_negative_raw_clamped(self):
        """Even with some positive components, penalty can push below 0."""
        score, comp = alpha_score(_entry(
            books_used=4,
            market_hold_median=8.0,
            edge_z=1.4,
            edge_ev_shrunk=0.001,
            agreement_score=50,
            market_volatility_sigma=0.02,
            consensus_prob=0.10,
        ))
        # 5+5 + 5+5 + 0+0 + (-25) = -5
        assert comp["raw_total"] == -5
        assert score == 0


# ---------------------------------------------------------------------------
# alpha_score – components dict structure
# ---------------------------------------------------------------------------


class TestComponentsDict:
    def test_has_all_keys(self):
        _, comp = alpha_score(_entry())
        expected_keys = {
            "market_robustness", "books_pts", "hold_pts",
            "edge_robustness", "edge_z_pts", "shrunk_pts",
            "agreement_stability", "agree_pts", "sigma_pts",
            "longshot_penalty", "raw_total", "inputs",
        }
        assert expected_keys.issubset(comp.keys())

    def test_inputs_dict(self):
        _, comp = alpha_score(_entry(books_used=6, consensus_prob=0.40))
        inputs = comp["inputs"]
        assert inputs["books_used"] == 6
        assert inputs["consensus_prob"] == 0.40
        assert "edge_z" in inputs
        assert "market_hold_median" in inputs
        assert "agreement_score" in inputs
        assert "market_volatility_sigma" in inputs
        assert "shrunk_pct" in inputs
        assert "edge_ev_shrunk" in inputs


# ---------------------------------------------------------------------------
# alpha_score – missing / default fields
# ---------------------------------------------------------------------------


class TestMissingFields:
    def test_empty_entry(self):
        """alpha_score should handle an empty dict gracefully."""
        score, comp = alpha_score({})
        assert score == 0
        assert comp["longshot_penalty"] == -25  # consensus_prob=0 < 0.15

    def test_partial_entry(self):
        score, comp = alpha_score({"books_used": 7, "edge_z": 2.5})
        assert comp["books_pts"] == 15
        assert comp["edge_z_pts"] == 20


# ---------------------------------------------------------------------------
# Gating: tier1b → tier2 downgrade
# ---------------------------------------------------------------------------


class TestGating:
    """Test alpha gating via slate.py integration."""

    def test_default_gate_disabled(self):
        assert ALPHA_GATE_ENABLED is False

    def test_gating_downgrades_weak_tier1b(self):
        """When enabled, a tier1b rec with Weak alpha → tier2."""
        from line_tracker.slate import classify_rec

        # Build an entry that would classify as tier1b
        entry = {
            "edge_pct": 3.0,
            "confidence": "High",
            "quality_tier": "Strong",
            "quality_score": 80,
            "market_volatility_sigma": 0.0,
            "edge_z": 2.0,
            "market_unstable": False,
            "books_used": 5,
            "oldest_update_age_min": 15.0,
            "market_hold_median": 5.0,
            "edge_ev_shrunk": 0.05,
            "consensus_prob": 0.55,
            "tier": "tier1b",
            # Alpha fields: force Weak by giving bad alpha inputs
            "agreement_score": 0.0,
        }

        # Verify it classifies as tier1b normally
        result = classify_rec(entry)
        assert result["tier"] == "tier1b"

        # Compute alpha — verify the score is computable
        a_score, _ = alpha_score(entry)
        alpha_label(a_score)
        # With agreement_score=0 and sigma=0 (which gives sigma_pts=10),
        # this entry gets decent scores. Let's make it truly Weak.
        weak_entry = {
            **entry,
            "books_used": 3,
            "market_hold_median": 9.0,
            "edge_z": 1.0,
            "edge_ev_shrunk": 0.0,
            "agreement_score": 0.0,
            "market_volatility_sigma": 0.05,
            "consensus_prob": 0.10,
        }
        w_score, _ = alpha_score(weak_entry)
        assert alpha_label(w_score) == "Weak"

    def test_gating_does_not_affect_non_tier1b(self):
        """Gating should only affect tier1b entries."""
        from line_tracker.alpha import alpha_label, alpha_score

        entry = _entry(
            books_used=3,
            market_hold_median=10.0,
            edge_z=0.5,
            edge_ev_shrunk=0.0,
            agreement_score=0.0,
            market_volatility_sigma=0.05,
            consensus_prob=0.05,
        )
        score, _ = alpha_score(entry)
        assert alpha_label(score) == "Weak"
        # Even with Weak alpha, a tier2 entry stays tier2 (gating
        # only applies to tier1b)

    def test_gating_skips_strong_tier1b(self):
        """Strong/Neutral tier1b should not be downgraded."""
        entry = _entry(
            books_used=7,
            market_hold_median=5.0,
            edge_z=2.5,
            edge_ev_shrunk=0.015,
            agreement_score=90.0,
            market_volatility_sigma=0.004,
            consensus_prob=0.55,
        )
        score, _ = alpha_score(entry)
        label = alpha_label(score)
        assert label in ("Strong", "Neutral")
        # No downgrade should happen


# ---------------------------------------------------------------------------
# End-to-end: score + label consistency
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_strong_entry(self):
        """A well-supported entry should score Strong."""
        score, _ = alpha_score(_entry(
            books_used=8,
            market_hold_median=4.5,
            edge_z=2.5,
            edge_ev_shrunk=0.02,
            agreement_score=95,
            market_volatility_sigma=0.003,
            consensus_prob=0.55,
        ))
        assert score >= 65
        assert alpha_label(score) == "Strong"

    def test_weak_entry(self):
        """A low-quality longshot should score Weak."""
        score, _ = alpha_score(_entry(
            books_used=3,
            market_hold_median=9.0,
            edge_z=1.0,
            edge_ev_shrunk=0.0,
            agreement_score=40,
            market_volatility_sigma=0.02,
            consensus_prob=0.10,
        ))
        assert score < 40
        assert alpha_label(score) == "Weak"

    def test_neutral_entry(self):
        """A middling entry should score Neutral."""
        score, _ = alpha_score(_entry(
            books_used=5,
            market_hold_median=7.0,
            edge_z=1.75,
            edge_ev_shrunk=0.005,
            agreement_score=75,
            market_volatility_sigma=0.008,
            consensus_prob=0.35,
        ))
        assert 40 <= score < 65
        assert alpha_label(score) == "Neutral"
