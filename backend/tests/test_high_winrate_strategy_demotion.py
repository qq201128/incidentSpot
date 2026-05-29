from __future__ import annotations

import sqlite3
from pathlib import Path

from app.services import high_winrate_strategy_demotion as demotion
from app.services.high_winrate_strategy_rotation import ensure_high_winrate_status_table
from app.services.rule_config import DURATION_TO_MINUTES
from app.services.strategy_registry import HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY


def test_promotion_enables_simulation_slot(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "promotion.db"
    _init_db(db_path)
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))

    result = demotion.promote_high_winrate_strategy("btcusdt", "10m")

    row = _slot(db_path, "10m")
    assert result["status"] == demotion.STATUS_PAPER_LIVE_COLLECTING
    assert result["tradable"] is False
    assert row["enabled"] == 1
    assert row["live_trading_enabled"] == 0


def test_paper_live_passed_after_live_samples_hit_target(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "active.db"
    _init_db(db_path)
    _insert_slot(db_path, "10m", enabled=1, live=1)
    _insert_predictions(db_path, "10m", [True] * demotion.ACTIVE_SAMPLE_COUNT)
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(demotion, "high_winrate_candidate_rule", lambda *_args: None)

    result = demotion.evaluate_high_winrate_demotion("BTCUSDT", "10m")

    row = _slot(db_path, "10m")
    assert result["status"] == demotion.STATUS_PAPER_LIVE_PASSED
    assert result["reason"] == "stable_paper_live_target_met"
    assert result["sampleCount"] == demotion.ACTIVE_SAMPLE_COUNT
    assert result["requiredSampleCount"] == demotion.ACTIVE_SAMPLE_COUNT
    assert result["tradable"] is False
    assert row["enabled"] == 1
    assert row["live_trading_enabled"] == 0


def test_insufficient_samples_stays_paper_live_collecting(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "collecting.db"
    _init_db(db_path)
    _insert_slot(db_path, "10m", enabled=1, live=1)
    _insert_predictions(db_path, "10m", [True])
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(demotion, "high_winrate_candidate_rule", lambda *_args: None)

    result = demotion.evaluate_high_winrate_demotion("BTCUSDT", "10m")

    row = _slot(db_path, "10m")
    assert result["status"] == demotion.STATUS_PAPER_LIVE_COLLECTING
    assert result["reason"] == "insufficient_settled_samples"
    assert row["live_trading_enabled"] == 0


def test_demotion_disables_live_but_keeps_simulation_on_loss_streak(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "loss-streak.db"
    _init_db(db_path)
    _insert_slot(db_path, "10m", enabled=1, live=1)
    _insert_predictions(db_path, "10m", [False, False, False, False, False])
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(demotion, "high_winrate_candidate_rule", lambda *_args: None)

    result = demotion.evaluate_high_winrate_demotion("BTCUSDT", "10m")

    row = _slot(db_path, "10m")
    assert result["status"] == demotion.STATUS_DEMOTED
    assert result["reason"] == "consecutive_losses"
    assert row["enabled"] == 1
    assert row["live_trading_enabled"] == 0


def test_demotion_below_live_target_keeps_collecting_predictions(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "below-target.db"
    _init_db(db_path)
    _insert_slot(db_path, "30m", enabled=1, live=1)
    _insert_predictions(db_path, "30m", ([True] * 3 + [False] * 2) * 6)
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(demotion, "high_winrate_candidate_rule", lambda *_args: None)

    result = demotion.evaluate_high_winrate_demotion("BTCUSDT", "30m")

    row = _slot(db_path, "30m")
    assert result["status"] == demotion.STATUS_DEMOTED
    assert result["reason"] == "paper_live_win_rate_below_target"
    assert row["enabled"] == 1
    assert row["live_trading_enabled"] == 0


def test_demotion_rejects_recent_rolling_instability(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "unstable-recent.db"
    _init_db(db_path)
    _insert_slot(db_path, "10m", enabled=1, live=1)
    stable_history = [True] * 20
    unstable_recent = [True, False, False, True, False, False, True, False, True, False]
    _insert_predictions(db_path, "10m", stable_history + unstable_recent)
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(demotion, "high_winrate_candidate_rule", lambda *_args: None)

    result = demotion.evaluate_high_winrate_demotion("BTCUSDT", "10m")

    row = _slot(db_path, "10m")
    assert result["status"] == demotion.STATUS_DEMOTED
    assert result["reason"] == "rolling_window_win_rate_below_target"
    assert result["paperStability"]["rollingWindows"][0]["winRate"] == 0.4
    assert row["live_trading_enabled"] == 0


def test_failed_top1_rotates_to_top2(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "rotate-top2.db"
    _init_db(db_path)
    _insert_slot(db_path, "10m", enabled=1, live=1)
    _insert_predictions(db_path, "10m", [False] * demotion.ACTIVE_SAMPLE_COUNT, rule="goal_combo__top1")
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(demotion, "high_winrate_candidate_rule", _cached_goal_rule)

    result = demotion.evaluate_high_winrate_demotion("BTCUSDT", "10m")

    row = _slot(db_path, "10m")
    assert result["status"] == demotion.STATUS_PAPER_LIVE_COLLECTING
    assert result["reason"] == "rotated_after_candidate_failed"
    assert result["activeRank"] == 2
    assert result["activeRule"] == "goal_combo__top2"
    assert result["previousCandidate"]["rank"] == 1
    assert row["live_trading_enabled"] == 0


def test_failed_top2_rotates_to_top3(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "rotate-top3.db"
    _init_db(db_path)
    _insert_slot(db_path, "10m", enabled=1, live=1)
    _insert_status(db_path, "10m", demotion.STATUS_PAPER_LIVE_COLLECTING, "rotated", active_rank=2, failed_ranks=[1])
    _insert_predictions(db_path, "10m", [False] * demotion.ACTIVE_SAMPLE_COUNT, rule="goal_combo__top2")
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(demotion, "high_winrate_candidate_rule", _cached_goal_rule)

    result = demotion.evaluate_high_winrate_demotion("BTCUSDT", "10m")

    assert result["status"] == demotion.STATUS_PAPER_LIVE_COLLECTING
    assert result["activeRank"] == 3
    assert result["activeRule"] == "goal_combo__top3"
    assert result["failedRanks"] == [1, 2]


def test_failed_top3_refreshes_goal_ranking(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "refresh-after-top3.db"
    refreshed = []
    _init_db(db_path)
    _insert_slot(db_path, "10m", enabled=1, live=1)
    _insert_status(db_path, "10m", demotion.STATUS_PAPER_LIVE_COLLECTING, "rotated", active_rank=3, failed_ranks=[1, 2])
    _insert_predictions(db_path, "10m", [False] * demotion.ACTIVE_SAMPLE_COUNT, rule="goal_combo__top3")
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(demotion, "high_winrate_candidate_rule", _cached_goal_rule)

    def refresh(symbol: str, duration: str) -> dict:
        refreshed.append((symbol, duration))
        demotion.promote_high_winrate_strategy(symbol, duration)
        return {
            "updatedAt": "now",
            "ranking": [{"factorName": "goal_combo__top1"}],
            "promotion": {"status": "active"},
        }

    monkeypatch.setattr(demotion, "refresh_high_winrate_goal", refresh)

    result = demotion.evaluate_high_winrate_demotion("BTCUSDT", "10m", allow_goal_refresh=True)

    assert refreshed == [("BTCUSDT", "10m")]
    assert result["status"] == demotion.STATUS_PAPER_LIVE_COLLECTING
    assert result["reason"] == demotion.REASON_OFFLINE_PROMOTION
    assert result["activeRank"] == 1


def test_failed_top3_records_empty_refresh_result(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "refresh-empty.db"
    _init_db(db_path)
    _insert_slot(db_path, "10m", enabled=1, live=1)
    _insert_status(db_path, "10m", demotion.STATUS_PAPER_LIVE_COLLECTING, "rotated", active_rank=3, failed_ranks=[1, 2])
    _insert_predictions(db_path, "10m", [False] * demotion.ACTIVE_SAMPLE_COUNT, rule="goal_combo__top3")
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(demotion, "high_winrate_candidate_rule", _cached_goal_rule)
    monkeypatch.setattr(
        demotion,
        "refresh_high_winrate_goal",
        lambda *_args: {
            "updatedAt": "now",
            "ranking": [],
            "rankingFailure": {"stage": "combo_threshold_gates", "reason": "no_combo_met_target_gates"},
            "validationGate": {"failureReason": "all_combos_rejected_by_validation"},
        },
    )

    result = demotion.evaluate_high_winrate_demotion("BTCUSDT", "10m", allow_goal_refresh=True)

    assert result["status"] == demotion.STATUS_DEMOTED
    assert result["reason"] == demotion.RANKING_REFRESH_FAILED_REASON
    assert result["refreshReport"]["rankingTotal"] == 0
    assert result["refreshReport"]["rankingFailure"]["reason"] == "no_combo_met_target_gates"


def test_failed_top3_defers_goal_refresh_by_default(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "refresh-pending.db"
    refreshed = []
    _init_db(db_path)
    _insert_slot(db_path, "10m", enabled=1, live=1)
    _insert_status(db_path, "10m", demotion.STATUS_PAPER_LIVE_COLLECTING, "rotated", active_rank=3, failed_ranks=[1, 2])
    _insert_predictions(db_path, "10m", [False] * demotion.ACTIVE_SAMPLE_COUNT, rule="goal_combo__top3")
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(demotion, "high_winrate_candidate_rule", _cached_goal_rule)
    monkeypatch.setattr(
        demotion,
        "refresh_high_winrate_goal",
        lambda *args: refreshed.append(args) or {"updatedAt": "now", "ranking": []},
    )

    result = demotion.evaluate_high_winrate_demotion("BTCUSDT", "10m")

    assert refreshed == []
    assert result["status"] == demotion.STATUS_DEMOTED
    assert result["reason"] == demotion.RANKING_REFRESH_PENDING_REASON
    assert result["pendingGoalRefresh"] is True


def test_paused_status_requires_new_promotion_to_clear(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "sticky-paused.db"
    _init_db(db_path)
    _insert_slot(db_path, "10m", enabled=1, live=0)
    _insert_status(db_path, "10m", demotion.STATUS_PAUSED, "manual_test_pause")
    _insert_predictions(db_path, "10m", [True] * demotion.ACTIVE_SAMPLE_COUNT)
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))

    result = demotion.evaluate_high_winrate_demotion("BTCUSDT", "10m")

    assert result["status"] == demotion.STATUS_PAUSED
    assert result["reason"] == "manual_test_pause"


def _init_db(path: Path) -> None:
    conn = _connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE auto_trade_strategies (
              strategy_key TEXT NOT NULL,
              symbol TEXT NOT NULL,
              duration TEXT NOT NULL,
              enabled INTEGER NOT NULL,
              live_trading_enabled INTEGER NOT NULL,
              duration_minutes INTEGER NOT NULL,
              qty REAL NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (strategy_key, symbol, duration)
            );
            CREATE TABLE predictions (
              strategy_key TEXT NOT NULL,
              symbol TEXT NOT NULL,
              duration TEXT NOT NULL,
              open_time INTEGER NOT NULL,
              prediction_correct INTEGER,
              actual_return REAL,
              high_winrate_rule TEXT,
              settled_at TEXT
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def _insert_slot(path: Path, duration: str, *, enabled: int, live: int) -> None:
    conn = _connect(path)
    try:
        conn.execute(
            """
            INSERT INTO auto_trade_strategies
            VALUES(?, 'BTCUSDT', ?, ?, ?, ?, 5.0, 'now')
            """,
            (
                HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
                duration,
                enabled,
                live,
                int(DURATION_TO_MINUTES[duration]),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_predictions(
    path: Path,
    duration: str,
    outcomes: list[bool],
    *,
    rule: str = "goal_combo__test",
) -> None:
    conn = _connect(path)
    try:
        for index, correct in enumerate(outcomes):
            conn.execute(
                """
                INSERT INTO predictions
                VALUES(?, 'BTCUSDT', ?, ?, ?, ?, ?, 'done')
                """,
                (
                    HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
                    duration,
                    index,
                    int(correct),
                    0.01 if correct else -0.01,
                    rule,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _insert_status(
    path: Path,
    duration: str,
    status: str,
    reason: str,
    *,
    active_rank: int = 1,
    failed_ranks: list[int] | None = None,
) -> None:
    conn = _connect(path)
    try:
        ensure_high_winrate_status_table(conn)
        details = f'{{"activeRank": {active_rank}, "failedRanks": {failed_ranks or []}}}'
        conn.execute(
            """
            INSERT INTO high_winrate_strategy_status
            VALUES(?, 'BTCUSDT', ?, ?, ?, ?, 0, NULL, NULL, 0, 'now', 'now')
            """,
            (HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, duration, status, reason, details),
        )
        conn.commit()
    finally:
        conn.close()


def _slot(path: Path, duration: str) -> sqlite3.Row:
    conn = _connect(path)
    try:
        return conn.execute(
            """
            SELECT * FROM auto_trade_strategies
            WHERE strategy_key = ? AND symbol = ? AND duration = ?
            """,
            (HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, "BTCUSDT", duration),
        ).fetchone()
    finally:
        conn.close()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _cached_goal_rule(_symbol: str, _duration: str, rank: int) -> str:
    return f"goal_combo__top{rank}"
