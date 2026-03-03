"""Step 1 – CLV-driven selection engine.

Queries historical closed rec_snapshots to learn which (market, tier,
alpha_label) combinations historically produce CLV > 0.  Returns a
"CLV filter profile" that downstream pruning can use as a gate.
"""

from __future__ import annotations

# Default: a group must have beaten the close >= 55% of the time to pass.
DEFAULT_MIN_PCT_POSITIVE = 55.0
# Minimum closed samples for the group to be considered reliable.
DEFAULT_MIN_SAMPLE = 5

# Fallback key hierarchy (most specific → least specific).
_FALLBACK_LABELS = ("market_tier_alpha", "market_tier", "market", "global")


def _compute_bucket_stats(snaps: list[dict]) -> dict:
    """Compute CLV stats for a list of snapshots."""
    cnt = len(snaps)
    if cnt == 0:
        return {"pct_positive": 0.0, "avg_clv_implied": 0.0, "count": 0}
    positive = sum(
        1 for s in snaps if (s.get("clv_delta_implied") or 0) > 0
    )
    pct_pos = 100.0 * positive / cnt
    avg_clv = sum(s.get("clv_delta_implied") or 0 for s in snaps) / cnt
    return {
        "pct_positive": round(pct_pos, 1),
        "avg_clv_implied": round(avg_clv, 6),
        "count": cnt,
    }


def build_clv_filter_profile(
    store,
    *,
    min_pct_positive: float = DEFAULT_MIN_PCT_POSITIVE,
    min_sample: int = DEFAULT_MIN_SAMPLE,
) -> dict:
    """Build a CLV filter profile from historical closed rec_snapshots.

    Returns a nested dict keyed by ``(market, tier, alpha_label)`` with
    ``{"pct_positive": float, "avg_clv_implied": float, "count": int,
      "passes": bool, "fallback_source": str}``.

    When a bucket has fewer than *min_sample* snapshots, statistics are
    taken from a coarser grouping using this fallback chain:

        (market, tier, alpha) → (market, tier) → (market) → global
    """
    closed = store.get_closed_snapshots()
    if not closed:
        return {"groups": {}, "total_closed": 0}

    # ── Build buckets at every granularity level ──────────────────────
    fine_groups: dict[tuple, list[dict]] = {}          # (m, t, a)
    market_tier_groups: dict[tuple, list[dict]] = {}   # (m, t)
    market_groups: dict[tuple, list[dict]] = {}        # (m,)
    all_snaps: list[dict] = []

    for snap in closed:
        market = snap.get("market", "")
        tier = snap.get("tier", "")
        alpha = snap.get("alpha_label", "")

        fine_groups.setdefault((market, tier, alpha), []).append(snap)
        market_tier_groups.setdefault((market, tier), []).append(snap)
        market_groups.setdefault((market,), []).append(snap)
        all_snaps.append(snap)

    # Pre-compute coarse stats once so lookups are O(1).
    mt_stats = {k: _compute_bucket_stats(v) for k, v in market_tier_groups.items()}
    m_stats = {k: _compute_bucket_stats(v) for k, v in market_groups.items()}
    global_stats = _compute_bucket_stats(all_snaps)

    # ── Resolve each fine-grained bucket ──────────────────────────────
    profile: dict[tuple, dict] = {}
    for key, snaps in fine_groups.items():
        stats = _compute_bucket_stats(snaps)
        fallback_source = _FALLBACK_LABELS[0]  # "market_tier_alpha"

        if stats["count"] < min_sample:
            # Try (market, tier)
            mt_key = key[:2]
            candidate = mt_stats[mt_key]
            if candidate["count"] >= min_sample:
                stats = candidate
                fallback_source = _FALLBACK_LABELS[1]  # "market_tier"
            else:
                # Try (market,)
                m_key = key[:1]
                candidate = m_stats[m_key]
                if candidate["count"] >= min_sample:
                    stats = candidate
                    fallback_source = _FALLBACK_LABELS[2]  # "market"
                else:
                    stats = global_stats
                    fallback_source = _FALLBACK_LABELS[3]  # "global"

        passes = (
            stats["count"] >= min_sample
            and stats["pct_positive"] >= min_pct_positive
        )
        profile[key] = {
            **stats,
            "passes": passes,
            "fallback_source": fallback_source,
        }

    return {"groups": profile, "total_closed": len(closed)}


def passes_clv_filter(entry: dict, profile: dict) -> bool:
    """Check whether *entry* passes the CLV filter from *profile*.

    If no profile data exists for the entry's group, it fails by default
    (conservative: unknown group is not trusted).
    """
    groups = profile.get("groups", {})
    if not groups:
        # No historical data yet — let everything through.
        return True

    key = (
        entry.get("market", ""),
        entry.get("tier", ""),
        entry.get("alpha_label", ""),
    )
    group = groups.get(key)
    if group is None:
        return False
    return group["passes"]
