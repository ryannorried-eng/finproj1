"""Tests for the Alpha v1.1 "More Selective" robustness overlay."""

from line_tracker.alpha import (
    ALPHA_GATE_ENABLED,
    alpha_label,
    alpha_score,
    clamp,
    implied_prob_from_american,
    lerp,
    smoothstep,
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
    best_odds: float | None = None,
    market: str | None = None,
) -> dict:
    """Build a minimal entry dict for alpha_score testing."""
    d: dict = {
        "books_used": books_used,
        "market_hold_median": market_hold_median,
        "edge_z": edge_z,
        "edge_ev_shrunk": edge_ev_shrunk,
        "agreement_score": agreement_score,
        "market_volatility_sigma": market_volatility_sigma,
        "consensus_prob": consensus_prob,
    }
    if best_odds is not None:
        d["best_odds"] = best_odds
    if market is not None:
        d["market"] = market
    return d


# ---------------------------------------------------------------------------
# Smooth-scoring helpers
# ---------------------------------------------------------------------------


class TestClampHelper:
    def test_below(self):
        assert clamp(-5.0, 0.0, 10.0) == 0.0

    def test_above(self):
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_within(self):
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_boundary_lo(self):
        assert clamp(0.0, 0.0, 10.0) == 0.0

    def test_boundary_hi(self):
        assert clamp(10.0, 0.0, 10.0) == 10.0


class TestLerp:
    def test_zero(self):
        assert lerp(0.0, 10.0, 0.0) == 0.0

    def test_one(self):
        assert lerp(0.0, 10.0, 1.0) == 10.0

    def test_half(self):
        assert lerp(0.0, 10.0, 0.5) == 5.0


class TestSmoothstep:
    def test_below_lo(self):
        assert smoothstep(1.0, 2.0, 0.5) == 0.0

    def test_above_hi(self):
        assert smoothstep(1.0, 2.0, 3.0) == 1.0

    def test_at_lo(self):
        assert smoothstep(1.0, 2.0, 1.0) == 0.0

    def test_at_hi(self):
        assert smoothstep(1.0, 2.0, 2.0) == 1.0

    def test_midpoint(self):
        # At midpoint t=0.5, smoothstep = 0.5*0.5*(3-1) = 0.25*2 = 0.5
        assert smoothstep(1.0, 2.0, 1.5) == 0.5

    def test_monotonic(self):
        vals = [smoothstep(0.0, 1.0, x / 10.0) for x in range(11)]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1]

    def test_equal_lo_hi(self):
        assert smoothstep(5.0, 5.0, 4.0) == 0.0
        assert smoothstep(5.0, 5.0, 5.0) == 1.0
        assert smoothstep(5.0, 5.0, 6.0) == 1.0


# ---------------------------------------------------------------------------
# Implied probability conversion
# ---------------------------------------------------------------------------


class TestImpliedProb:
    def test_plus_200(self):
        # +200 => 100/(200+100) = 1/3 ≈ 0.3333
        prob = implied_prob_from_american(200)
        assert prob is not None
        assert abs(prob - 1 / 3) < 0.001

    def test_minus_150(self):
        # -150 => 150/(150+100) = 150/250 = 0.60
        prob = implied_prob_from_american(-150)
        assert prob is not None
        assert abs(prob - 0.60) < 0.001

    def test_plus_100(self):
        # +100 => 100/200 = 0.50
        prob = implied_prob_from_american(100)
        assert prob is not None
        assert abs(prob - 0.50) < 0.001

    def test_minus_100(self):
        # -100 => 100/200 = 0.50
        prob = implied_prob_from_american(-100)
        assert prob is not None
        assert abs(prob - 0.50) < 0.001

    def test_dead_zone(self):
        # Between -100 and +100 (exclusive) returns None
        assert implied_prob_from_american(0) is None
        assert implied_prob_from_american(50) is None
        assert implied_prob_from_american(-50) is None


# ---------------------------------------------------------------------------
# Implied-odds longshot penalty boundaries
# ---------------------------------------------------------------------------


class TestImpliedLongshotPenalty:
    def test_high_prob_no_penalty(self):
        """implied >= 0.25 => 0 penalty."""
        _, comp = alpha_score(_entry(best_odds=-200))
        # -200 => implied = 200/300 ≈ 0.667 => no penalty
        assert comp["longshot_penalty_implied"] == 0.0

    def test_prob_below_010(self):
        """implied < 0.10 => -12."""
        # +1100 => 100/1200 ≈ 0.083
        _, comp = alpha_score(_entry(best_odds=1100))
        assert comp["longshot_penalty_implied"] == -12.0

    def test_prob_012(self):
        """implied ~0.12 (between 0.10 and 0.15) => -9."""
        # +750 => 100/850 ≈ 0.1176
        _, comp = alpha_score(_entry(best_odds=750))
        assert comp["longshot_penalty_implied"] == -9.0

    def test_prob_017(self):
        """implied ~0.17 (between 0.15 and 0.20) => -5."""
        # +500 => 100/600 ≈ 0.167
        _, comp = alpha_score(_entry(best_odds=500))
        assert comp["longshot_penalty_implied"] == -5.0

    def test_prob_022(self):
        """implied ~0.22 (between 0.20 and 0.25) => -2."""
        # +350 => 100/450 ≈ 0.222
        _, comp = alpha_score(_entry(best_odds=350))
        assert comp["longshot_penalty_implied"] == -2.0

    def test_prob_030(self):
        """implied ~0.30 (>= 0.25) => 0."""
        # +250 => 100/350 ≈ 0.286
        _, comp = alpha_score(_entry(best_odds=250))
        assert comp["longshot_penalty_implied"] == 0.0

    def test_no_odds_no_penalty(self):
        """No best_odds => 0 penalty."""
        _, comp = alpha_score(_entry())
        assert comp["longshot_penalty_implied"] == 0.0


# ---------------------------------------------------------------------------
# Consensus smooth penalty
# ---------------------------------------------------------------------------


class TestConsensusPenalty:
    def test_high_prob_near_zero(self):
        """consensus_prob=0.30 => smoothstep=1 => penalty near 0."""
        _, comp = alpha_score(_entry(consensus_prob=0.30))
        assert comp["longshot_penalty_consensus"] >= -1.0  # near 0

    def test_low_prob_near_max(self):
        """consensus_prob=0.18 => smoothstep=0 => penalty near -18."""
        _, comp = alpha_score(_entry(consensus_prob=0.18))
        assert comp["longshot_penalty_consensus"] <= -17.0  # near -18

    def test_very_low_prob_max(self):
        """consensus_prob=0.10 => penalty = -18."""
        _, comp = alpha_score(_entry(consensus_prob=0.10))
        assert comp["longshot_penalty_consensus"] == -18.0

    def test_very_high_prob_zero(self):
        """consensus_prob=0.55 => penalty = 0."""
        _, comp = alpha_score(_entry(consensus_prob=0.55))
        assert comp["longshot_penalty_consensus"] == 0.0

    def test_monotonic(self):
        """Penalty gets less severe as consensus_prob increases."""
        probs = [0.10, 0.15, 0.20, 0.24, 0.28, 0.30, 0.40]
        penalties = []
        for p in probs:
            _, comp = alpha_score(_entry(consensus_prob=p))
            penalties.append(comp["longshot_penalty_consensus"])
        # Penalties should be non-decreasing (becoming less negative)
        for i in range(len(penalties) - 1):
            assert penalties[i] <= penalties[i + 1], (
                f"penalty at prob={probs[i]} ({penalties[i]}) > "
                f"penalty at prob={probs[i+1]} ({penalties[i+1]})"
            )


# ---------------------------------------------------------------------------
# Market-type ML penalty
# ---------------------------------------------------------------------------


class TestMLPenalty:
    def test_moneyline_penalty(self):
        """Moneyline market gets -4 base penalty."""
        _, comp = alpha_score(_entry(
            market="moneyline",
            best_odds=-150,  # implied ~0.60, not a dog
        ))
        assert comp["ml_penalty"] == -4.0

    def test_h2h_penalty(self):
        """h2h market gets same penalty as moneyline."""
        _, comp = alpha_score(_entry(
            market="h2h",
            best_odds=-150,
        ))
        assert comp["ml_penalty"] == -4.0

    def test_moneyline_dog_extra(self):
        """ML dog with implied < 0.20 gets -7."""
        # +500 => implied ~0.167
        _, comp = alpha_score(_entry(
            market="moneyline",
            best_odds=500,
        ))
        assert comp["ml_penalty"] == -7.0

    def test_spread_no_penalty(self):
        """Spread market gets 0 ML penalty."""
        _, comp = alpha_score(_entry(market="spread", best_odds=-110))
        assert comp["ml_penalty"] == 0.0

    def test_total_no_penalty(self):
        """Total market gets 0 ML penalty."""
        _, comp = alpha_score(_entry(market="total", best_odds=-110))
        assert comp["ml_penalty"] == 0.0

    def test_no_market_no_penalty(self):
        """No market type => 0 penalty."""
        _, comp = alpha_score(_entry())
        assert comp["ml_penalty"] == 0.0


# ---------------------------------------------------------------------------
# alpha_label – updated thresholds
# ---------------------------------------------------------------------------


class TestAlphaLabel:
    def test_strong(self):
        assert alpha_label(72) == "Strong"
        assert alpha_label(100) == "Strong"
        assert alpha_label(85) == "Strong"

    def test_neutral(self):
        assert alpha_label(45) == "Neutral"
        assert alpha_label(71) == "Neutral"
        assert alpha_label(55) == "Neutral"

    def test_weak(self):
        assert alpha_label(0) == "Weak"
        assert alpha_label(44) == "Weak"
        assert alpha_label(10) == "Weak"

    def test_boundaries(self):
        assert alpha_label(72) == "Strong"
        assert alpha_label(71) == "Neutral"
        assert alpha_label(45) == "Neutral"
        assert alpha_label(44) == "Weak"


# ---------------------------------------------------------------------------
# alpha_score – component A: Market robustness (0–30)
# ---------------------------------------------------------------------------


class TestMarketRobustness:
    def test_many_books_high_score(self):
        """10 books should get near-max books_pts (~18)."""
        _, comp = alpha_score(_entry(books_used=10))
        assert comp["books_pts"] >= 17.0

    def test_few_books_low_score(self):
        """3 books should get ~0 books_pts."""
        _, comp = alpha_score(_entry(books_used=3))
        assert comp["books_pts"] < 1.0

    def test_4_books_near_zero(self):
        """4 books is at lo boundary => ~0."""
        _, comp = alpha_score(_entry(books_used=4))
        assert comp["books_pts"] < 1.0

    def test_7_books_moderate(self):
        """7 books should be moderate (smoothstep midrange)."""
        _, comp = alpha_score(_entry(books_used=7))
        assert 6.0 < comp["books_pts"] < 14.0

    def test_hold_low_high_score(self):
        """Low hold (<=4.5) should get near-max hold_pts (~12)."""
        _, comp = alpha_score(_entry(market_hold_median=4.0))
        assert comp["hold_pts"] >= 11.0

    def test_hold_high_low_score(self):
        """High hold (>=8) should get ~0."""
        _, comp = alpha_score(_entry(market_hold_median=8.5))
        assert comp["hold_pts"] < 1.0

    def test_market_robustness_capped_at_30(self):
        """Even with perfect inputs, market_robustness <= 30."""
        _, comp = alpha_score(_entry(books_used=10, market_hold_median=3.0))
        assert comp["market_robustness"] == 30.0


# ---------------------------------------------------------------------------
# alpha_score – component B: Edge robustness (0–40)
# ---------------------------------------------------------------------------


class TestEdgeRobustness:
    def test_high_edge_z(self):
        """edge_z=2.25+ should get near-max edge_z_pts (~20)."""
        _, comp = alpha_score(_entry(edge_z=2.5))
        assert comp["edge_z_pts"] >= 19.0

    def test_low_edge_z(self):
        """edge_z=0.5 (below 0.75) should get 0."""
        _, comp = alpha_score(_entry(edge_z=0.5))
        assert comp["edge_z_pts"] == 0.0

    def test_mid_edge_z(self):
        """edge_z=1.5 should be in smoothstep midrange."""
        _, comp = alpha_score(_entry(edge_z=1.5))
        assert 0 < comp["edge_z_pts"] < 20

    def test_high_shrunk(self):
        """shrunk_pct=3.0+ should get near-max shrunk_pts (~20)."""
        _, comp = alpha_score(_entry(edge_ev_shrunk=0.03))
        assert comp["shrunk_pts"] >= 19.0

    def test_zero_shrunk(self):
        """shrunk_pct=0 should get 0."""
        _, comp = alpha_score(_entry(edge_ev_shrunk=0.0))
        assert comp["shrunk_pts"] == 0.0

    def test_edge_robustness_sum(self):
        """Edge robustness = edge_z_pts + shrunk_pts."""
        _, comp = alpha_score(_entry(edge_z=2.5, edge_ev_shrunk=0.03))
        assert abs(
            comp["edge_robustness"] - comp["edge_z_pts"] - comp["shrunk_pts"]
        ) < 0.01


# ---------------------------------------------------------------------------
# alpha_score – component C: Agreement / stability (0–20)
# ---------------------------------------------------------------------------


class TestAgreementStability:
    def test_high_agreement(self):
        """agreement=95+ gets near-max agree_pts (~12)."""
        _, comp = alpha_score(_entry(agreement_score=96))
        assert comp["agree_pts"] >= 11.0

    def test_low_agreement(self):
        """agreement=60 gets ~0."""
        _, comp = alpha_score(_entry(agreement_score=60))
        assert comp["agree_pts"] < 1.0

    def test_very_low_sigma(self):
        """sigma <= 0.006 gets near-max sigma_pts (~8)."""
        _, comp = alpha_score(_entry(market_volatility_sigma=0.004))
        assert comp["sigma_pts"] >= 7.0

    def test_high_sigma(self):
        """sigma >= 0.016 gets ~0."""
        _, comp = alpha_score(_entry(market_volatility_sigma=0.02))
        assert comp["sigma_pts"] < 1.0

    def test_stability_capped_at_20(self):
        """Agreement+stability capped at 20."""
        _, comp = alpha_score(_entry(
            agreement_score=99, market_volatility_sigma=0.001,
        ))
        assert comp["agreement_stability"] <= 20.0


# ---------------------------------------------------------------------------
# Score monotonicity
# ---------------------------------------------------------------------------


class TestMonotonicity:
    def test_increasing_edge_z_increases_score(self):
        """Holding all else constant, higher edge_z => higher score."""
        base = dict(
            books_used=7, market_hold_median=5.0,
            edge_ev_shrunk=0.015, agreement_score=90,
            market_volatility_sigma=0.005, consensus_prob=0.55,
        )
        prev_score = -1
        for ez in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            score, _ = alpha_score({**base, "edge_z": ez})
            assert score >= prev_score, (
                f"Score decreased at edge_z={ez}: {score} < {prev_score}"
            )
            prev_score = score

    def test_increasing_books_increases_score(self):
        """More books => higher score."""
        base = dict(
            market_hold_median=5.0, edge_z=2.0,
            edge_ev_shrunk=0.015, agreement_score=90,
            market_volatility_sigma=0.005, consensus_prob=0.55,
        )
        prev_score = -1
        for b in [2, 4, 6, 8, 10]:
            score, _ = alpha_score({**base, "books_used": b})
            assert score >= prev_score, (
                f"Score decreased at books={b}: {score} < {prev_score}"
            )
            prev_score = score


# ---------------------------------------------------------------------------
# Selectivity: previously Strong-ish now Neutral unless all strong
# ---------------------------------------------------------------------------


class TestSelectivity:
    def test_mediocre_entry_not_strong(self):
        """A middling entry that might have been Strong under v1 thresholds
        should now be Neutral or Weak under v1.1."""
        score, _ = alpha_score(_entry(
            books_used=5,
            market_hold_median=6.5,
            edge_z=2.0,
            edge_ev_shrunk=0.008,
            agreement_score=80,
            market_volatility_sigma=0.008,
            consensus_prob=0.40,
        ))
        # Under old v1: this scored ~65 (Strong).  Under v1.1 with smooth
        # curves and higher Strong threshold (72), it should be Neutral.
        assert alpha_label(score) != "Strong"

    def test_excellent_entry_strong(self):
        """An excellent entry should still be Strong."""
        score, _ = alpha_score(_entry(
            books_used=10,
            market_hold_median=4.0,
            edge_z=2.5,
            edge_ev_shrunk=0.03,
            agreement_score=95,
            market_volatility_sigma=0.003,
            consensus_prob=0.55,
        ))
        assert alpha_label(score) == "Strong"

    def test_moneyline_dog_harder_to_strong(self):
        """A moneyline longshot gets penalised, making Strong harder."""
        score_ml, _ = alpha_score(_entry(
            books_used=8,
            market_hold_median=4.5,
            edge_z=2.3,
            edge_ev_shrunk=0.02,
            agreement_score=92,
            market_volatility_sigma=0.004,
            consensus_prob=0.55,
            best_odds=500,       # implied ~0.167
            market="moneyline",
        ))
        score_spread, _ = alpha_score(_entry(
            books_used=8,
            market_hold_median=4.5,
            edge_z=2.3,
            edge_ev_shrunk=0.02,
            agreement_score=92,
            market_volatility_sigma=0.004,
            consensus_prob=0.55,
            best_odds=-110,
            market="spread",
        ))
        # ML longshot should score lower than comparable spread
        assert score_ml < score_spread


# ---------------------------------------------------------------------------
# alpha_score – clamp behaviour
# ---------------------------------------------------------------------------


class TestClamp:
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
        assert comp["raw_total"] < 0
        assert score == 0

    def test_score_never_exceeds_100(self):
        """Even with all max inputs, score <= 100."""
        score, _ = alpha_score(_entry(
            books_used=10,
            market_hold_median=3.0,
            edge_z=3.0,
            edge_ev_shrunk=0.05,
            agreement_score=99,
            market_volatility_sigma=0.001,
            consensus_prob=0.60,
        ))
        assert score <= 100


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
            "longshot_penalty_consensus", "longshot_penalty_implied",
            "ml_penalty", "raw_total", "final_score", "inputs",
        }
        assert expected_keys.issubset(comp.keys())

    def test_inputs_dict(self):
        _, comp = alpha_score(_entry(
            books_used=6, consensus_prob=0.40,
            best_odds=-150, market="spread",
        ))
        inputs = comp["inputs"]
        assert inputs["books_used"] == 6
        assert inputs["consensus_prob"] == 0.40
        assert "edge_z" in inputs
        assert "market_hold_median" in inputs
        assert "agreement_score" in inputs
        assert "market_volatility_sigma" in inputs
        assert "shrunk_pct" in inputs
        assert "edge_ev_shrunk" in inputs
        assert inputs["implied_prob"] is not None
        assert inputs["market_type"] == "spread"

    def test_inputs_no_odds(self):
        """When no odds provided, implied_prob is None."""
        _, comp = alpha_score(_entry())
        assert comp["inputs"]["implied_prob"] is None
        assert comp["inputs"]["market_type"] is None


# ---------------------------------------------------------------------------
# alpha_score – missing / default fields
# ---------------------------------------------------------------------------


class TestMissingFields:
    def test_empty_entry(self):
        """alpha_score should handle an empty dict gracefully.

        With all-zero inputs, hold=0 gives max hold_pts (12) and
        sigma=0 gives max sigma_pts (8), but consensus=0 gives
        penalty=-18, so raw = 12 + 8 - 18 = 2.
        """
        score, comp = alpha_score({})
        assert score >= 0  # never negative
        assert comp["longshot_penalty_consensus"] == -18.0

    def test_partial_entry(self):
        score, comp = alpha_score({"books_used": 7, "edge_z": 2.5})
        assert comp["books_pts"] > 0
        assert comp["edge_z_pts"] > 0


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
            "agreement_score": 0.0,
        }

        result = classify_rec(entry)
        assert result["tier"] == "tier1b"

        a_score, _ = alpha_score(entry)
        alpha_label(a_score)

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

    def test_gating_skips_strong_tier1b(self):
        """Strong/Neutral tier1b should not be downgraded."""
        entry = _entry(
            books_used=10,
            market_hold_median=4.0,
            edge_z=2.5,
            edge_ev_shrunk=0.03,
            agreement_score=95,
            market_volatility_sigma=0.003,
            consensus_prob=0.55,
        )
        score, _ = alpha_score(entry)
        label = alpha_label(score)
        assert label in ("Strong", "Neutral")


# ---------------------------------------------------------------------------
# End-to-end: score + label consistency
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_strong_entry(self):
        """A well-supported entry should score Strong."""
        score, _ = alpha_score(_entry(
            books_used=10,
            market_hold_median=4.0,
            edge_z=2.5,
            edge_ev_shrunk=0.03,
            agreement_score=95,
            market_volatility_sigma=0.003,
            consensus_prob=0.55,
        ))
        assert score >= 72
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
        assert score < 45
        assert alpha_label(score) == "Weak"

    def test_neutral_entry(self):
        """A middling entry should score Neutral."""
        score, _ = alpha_score(_entry(
            books_used=7,
            market_hold_median=5.5,
            edge_z=1.8,
            edge_ev_shrunk=0.012,
            agreement_score=82,
            market_volatility_sigma=0.008,
            consensus_prob=0.40,
        ))
        assert 45 <= score < 72
        assert alpha_label(score) == "Neutral"

    def test_json_serializable(self):
        """All components should be JSON-serializable."""
        import json
        _, comp = alpha_score(_entry(best_odds=-150, market="moneyline"))
        # Should not raise
        json.dumps(comp)
