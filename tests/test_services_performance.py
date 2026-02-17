"""Tests for performance service orchestration."""

from line_tracker.services.performance_service import load_clv_df


class _Store:
    def get_all_clv(self):
        return [
            {
                "pick_odds_decimal": 1.95,
                "best_odds_close_decimal": 1.84,
                "consensus_prob_at_pick": 0.51,
                "consensus_prob_close": 0.55,
                "closed_at": "2026-01-20T12:00:00",
            },
        ]


def test_load_clv_df_builds_expected_columns():
    df = load_clv_df(_Store())
    assert not df.empty
    assert "clv_decimal" in df.columns
    assert "clv_prob" in df.columns
