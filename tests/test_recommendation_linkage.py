"""Tests for PR3.3: recommendation-to-bet linkage."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from line_tracker.core.math import build_recommendation_id
from line_tracker.db.migrate import ensure_latest, get_schema_version
from line_tracker.services import bet_service
from line_tracker.storage import LineStore


def _migrations_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "src" / "line_tracker" / "db" / "migrations"
    )


# ---------------------------------------------------------------------------
# build_recommendation_id tests
# ---------------------------------------------------------------------------


class TestBuildRecommendationId:
    """Determinism and input-format tests for build_recommendation_id."""

    def test_deterministic_same_dict(self):
        rec = {
            "market": "moneyline",
            "selection": "Team A",
            "line": None,
            "best_sportsbook": "DraftKings",
            "best_odds": -150,
        }
        id1 = build_recommendation_id(rec)
        id2 = build_recommendation_id(rec)
        assert id1 == id2
        assert isinstance(id1, str)
        assert len(id1) == 16

    def test_deterministic_equivalent_dicts(self):
        rec_a = {
            "market": "spread",
            "selection": "Away",
            "line": 3.5,
            "best_sportsbook": "FanDuel",
            "best_odds": 110,
        }
        rec_b = dict(rec_a)
        assert build_recommendation_id(rec_a) == build_recommendation_id(rec_b)

    def test_different_recs_produce_different_ids(self):
        rec_a = {
            "market": "moneyline",
            "selection": "Team A",
            "line": None,
            "best_sportsbook": "DraftKings",
            "best_odds": -150,
        }
        rec_b = {
            "market": "moneyline",
            "selection": "Team B",
            "line": None,
            "best_sportsbook": "FanDuel",
            "best_odds": 130,
        }
        assert build_recommendation_id(rec_a) != build_recommendation_id(rec_b)

    def test_works_with_object_attributes(self):
        """build_recommendation_id should handle objects with attributes."""

        class _FakeRec:
            market = "total"
            selection = "Over"
            line = 220.5
            best_sportsbook = "BetMGM"
            best_odds = -110

        rec = _FakeRec()
        id1 = build_recommendation_id(rec)
        id2 = build_recommendation_id(rec)
        assert id1 == id2
        assert len(id1) == 16

    def test_dict_and_object_with_same_values_match(self):
        rec_dict = {
            "market": "total",
            "selection": "Over",
            "line": 220.5,
            "best_sportsbook": "BetMGM",
            "best_odds": -110,
        }

        class _FakeRec:
            market = "total"
            selection = "Over"
            line = 220.5
            best_sportsbook = "BetMGM"
            best_odds = -110

        assert build_recommendation_id(rec_dict) == build_recommendation_id(_FakeRec())


# ---------------------------------------------------------------------------
# execution_delta sign logic
# ---------------------------------------------------------------------------


class TestExecutionDelta:
    """Verify execution_delta_decimal sign logic via submit_bet."""

    def test_positive_delta_when_odds_better_than_fair(self, monkeypatch):
        """When best_odds_decimal > 1/consensus_prob, delta should be positive."""
        captured = {}

        def _fake_persist(bet, store, recommendation_meta=None):
            captured["meta"] = recommendation_meta
            return "bet-42"

        monkeypatch.setattr(
            bet_service, "persist_bet_with_snapshot", _fake_persist,
        )

        # consensus_prob=0.50 → fair_decimal=2.0
        # best_odds=+120 → decimal=2.20
        # delta = 2.20 - 2.0 = +0.20
        rec = {
            "market": "moneyline",
            "selection": "Team X",
            "line": None,
            "best_sportsbook": "DraftKings",
            "best_odds": 120,
            "consensus_prob": 0.50,
            "quality_tier": "Strong",
            "edge_pct": 3.5,
        }
        bet_service.submit_bet(
            object(), object(),
            source_page="slate",
            recommendation=rec,
            rank=1,
        )
        meta = captured["meta"]
        assert meta is not None
        assert meta["execution_delta_decimal"] > 0

    def test_negative_delta_when_odds_worse_than_fair(self, monkeypatch):
        """When best_odds_decimal < 1/consensus_prob, delta should be negative."""
        captured = {}

        def _fake_persist(bet, store, recommendation_meta=None):
            captured["meta"] = recommendation_meta
            return "bet-43"

        monkeypatch.setattr(
            bet_service, "persist_bet_with_snapshot", _fake_persist,
        )

        # consensus_prob=0.60 → fair_decimal=1.6667
        # best_odds=-150 → decimal=1.6667
        # Actually -150 → 100/150 + 1 = 1.6667 (exact match, ~0 delta)
        # Use -200 → 100/200 + 1 = 1.50
        # delta = 1.50 - 1.6667 = -0.1667
        rec = {
            "market": "moneyline",
            "selection": "Team Y",
            "line": None,
            "best_sportsbook": "FanDuel",
            "best_odds": -200,
            "consensus_prob": 0.60,
            "quality_tier": "Moderate",
            "edge_pct": 1.0,
        }
        bet_service.submit_bet(
            object(), object(),
            source_page="shopping",
            recommendation=rec,
            rank=3,
        )
        meta = captured["meta"]
        assert meta is not None
        assert meta["execution_delta_decimal"] < 0

    def test_all_metadata_fields_populated(self, monkeypatch):
        captured = {}

        def _fake_persist(bet, store, recommendation_meta=None):
            captured["meta"] = recommendation_meta
            return "bet-44"

        monkeypatch.setattr(
            bet_service, "persist_bet_with_snapshot", _fake_persist,
        )

        rec = {
            "market": "spread",
            "selection": "Home",
            "line": -3.5,
            "best_sportsbook": "BetMGM",
            "best_odds": -110,
            "consensus_prob": 0.55,
            "quality_tier": "Elite",
            "edge_pct": 4.2,
        }
        bet_service.submit_bet(
            object(), object(),
            source_page="slate",
            recommendation=rec,
            rank=2,
        )
        meta = captured["meta"]
        assert meta["source_page"] == "slate"
        assert meta["rank_at_pick"] == 2
        assert meta["quality_tier_at_pick"] == "Elite"
        assert meta["edge_pct_at_pick"] == 4.2
        assert meta["consensus_prob_at_pick"] == 0.55
        assert isinstance(meta["recommendation_id"], str)
        assert len(meta["recommendation_id"]) == 16
        assert isinstance(meta["execution_delta_decimal"], float)

    def test_no_recommendation_only_source_page(self, monkeypatch):
        captured = {}

        def _fake_persist(bet, store, recommendation_meta=None):
            captured["meta"] = recommendation_meta
            return "bet-45"

        monkeypatch.setattr(
            bet_service, "persist_bet_with_snapshot", _fake_persist,
        )

        bet_service.submit_bet(
            object(), object(),
            source_page="detail",
        )
        meta = captured["meta"]
        assert meta == {"source_page": "detail"}

    def test_backward_compat_no_kwargs(self, monkeypatch):
        captured = {}

        def _fake_persist(bet, store, recommendation_meta=None):
            captured["meta"] = recommendation_meta
            return "bet-46"

        monkeypatch.setattr(
            bet_service, "persist_bet_with_snapshot", _fake_persist,
        )

        bet_service.submit_bet(object(), object())
        assert captured["meta"] is None


# ---------------------------------------------------------------------------
# Migration 0003 and schema version 3
# ---------------------------------------------------------------------------


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in info)


class TestMigration0003:
    def test_linestore_init_reaches_version_3(self, tmp_path):
        db = tmp_path / "linkage_a.db"
        with LineStore(db_path=db) as store:
            version = get_schema_version(store._conn)
            assert version == 11

    def test_new_columns_exist(self, tmp_path):
        db = tmp_path / "linkage_b.db"
        with LineStore(db_path=db) as store:
            for col in (
                "source_page",
                "recommendation_id",
                "rank_at_pick",
                "quality_tier_at_pick",
                "edge_pct_at_pick",
                "consensus_prob_at_pick",
                "execution_delta_decimal",
            ):
                assert _column_exists(store._conn, "bets", col), (
                    f"Column {col} missing from bets table"
                )

    def test_migration_idempotent(self, tmp_path):
        db = tmp_path / "linkage_c.db"
        migrations = _migrations_path()

        with sqlite3.connect(db) as conn:
            ensure_latest(conn, migrations)
            assert get_schema_version(conn) == 11
            ensure_latest(conn, migrations)
            assert get_schema_version(conn) == 11

    def test_upgrade_from_version_2(self, tmp_path):
        """Upgrade a DB at version 2 to 3 by applying only migration 0003."""
        db = tmp_path / "linkage_d.db"
        migrations = _migrations_path()

        with sqlite3.connect(db) as conn:
            # Apply up to version 2 manually
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version"
                " (version INTEGER NOT NULL)"
            )
            conn.execute("INSERT INTO schema_version(version) VALUES (2)")
            init_sql = (migrations / "0001_init.sql").read_text(encoding="utf-8")
            conn.executescript(init_sql)
            idx_sql = (migrations / "0002_indexes.sql").read_text(encoding="utf-8")
            conn.executescript(idx_sql)
            conn.commit()

            assert get_schema_version(conn) == 2
            ensure_latest(conn, migrations)
            assert get_schema_version(conn) == 11
            assert _column_exists(conn, "bets", "recommendation_id")


# ---------------------------------------------------------------------------
# End-to-end: metadata persists in DB
# ---------------------------------------------------------------------------


class TestMetadataPersistsInDB:
    def test_recommendation_metadata_round_trips(self, tmp_path):
        """Insert a bet with recommendation metadata and read it back."""
        db = tmp_path / "linkage_e2e.db"
        with LineStore(db_path=db) as store:
            bet_row = {
                "bet_id": "test-bet-1",
                "created_at": "2025-01-01T00:00:00Z",
                "sportsbook": "DraftKings",
                "stake": 100.0,
                "total_odds_american": -110,
                "total_odds_decimal": 1.9091,
                "potential_payout": 190.91,
                "profit": 90.91,
                "status": "active",
                "settled_at": None,
                "outcome": None,
                "source_page": "slate",
                "recommendation_id": "abc123def4567890",
                "rank_at_pick": 1,
                "quality_tier_at_pick": "Elite",
                "edge_pct_at_pick": 3.5,
                "consensus_prob_at_pick": 0.55,
                "execution_delta_decimal": 0.123456,
            }
            store.bets_repo.insert_bet(bet_row)
            store._conn.commit()

            rows = store.get_bets()
            assert len(rows) == 1
            row = rows[0]
            assert row["source_page"] == "slate"
            assert row["recommendation_id"] == "abc123def4567890"
            assert row["rank_at_pick"] == 1
            assert row["quality_tier_at_pick"] == "Elite"
            assert abs(row["edge_pct_at_pick"] - 3.5) < 0.001
            assert abs(row["consensus_prob_at_pick"] - 0.55) < 0.001
            assert abs(row["execution_delta_decimal"] - 0.123456) < 0.0001

    def test_null_metadata_backward_compat(self, tmp_path):
        """Bet without recommendation metadata should have NULLs."""
        db = tmp_path / "linkage_compat.db"
        with LineStore(db_path=db) as store:
            bet_row = {
                "bet_id": "test-bet-2",
                "created_at": "2025-01-01T00:00:00Z",
                "sportsbook": "FanDuel",
                "stake": 50.0,
                "total_odds_american": 150,
                "total_odds_decimal": 2.5,
                "potential_payout": 125.0,
                "profit": 75.0,
                "status": "active",
            }
            store.bets_repo.insert_bet(bet_row)
            store._conn.commit()

            rows = store.get_bets()
            assert len(rows) == 1
            row = rows[0]
            assert row["source_page"] is None
            assert row["recommendation_id"] is None
            assert row["rank_at_pick"] is None
