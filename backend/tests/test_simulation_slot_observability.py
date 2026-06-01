from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.api import auto_trade as auto_trade_api
from app.db import session
from app.services import (
    factor_combination_cache_service,
    factor_cache_metadata,
    factor_ranking_cache_service,
    high_winrate_combo_cache_service,
    paper_live_failure_store,
    qualified_factor_simulation_slots,
    simulation_slot_observability_service,
    simulation_slot_runtime,
)
from app.services.factor_candidate_signal_keys import factor_candidate_signal_key
from app.services.factor_combo_simulation_keys import simulation_strategy_key_for_factor_name


def test_sync_report_exposes_single_combo_rejections_runtime_and_failures(monkeypatch, tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)
    _init_db(db_path, monkeypatch)
    single_key = factor_candidate_signal_key("ret_good")
    combo_key = simulation_strategy_key_for_factor_name("combo__good__carry")
    _seed_klines(db_path)
    _save_single_cache(db_path)
    _save_combo_cache(db_path)
    _seed_settled_event(db_path, combo_key)
    _seed_prediction_failure(db_path, single_key)

    report = qualified_factor_simulation_slots.sync_qualified_simulation_slots("btcusdt", "10m", qty=7.5)

    assert report["singleFactorSlots"] == 1
    assert report["comboFactorSlots"] == 1
    assert report["enabledSlots"] == 2
    assert report["thresholds"]["minWinRate"] > 0
    assert report["updatedAt"]
    by_key = {row["strategyKey"]: row for row in report["items"] if row.get("strategyKey")}
    assert by_key[single_key]["gateStatus"] == "enabled"
    assert by_key[single_key]["slot"]["qty"] == 7.5
    assert by_key[single_key]["latestFailure"]["reason"] == "factor candidate signal missing column"
    assert by_key[combo_key]["gateStatus"] == "enabled"
    assert by_key[combo_key]["latestEvent"]["externalStatus"] == "SIMULATED"
    assert by_key[combo_key]["latestEvent"]["settlementPnl"] == 3.25
    rejected = {row["factorName"]: row["rejectionReason"] for row in report["items"]}
    assert rejected["ret_weak"] == "win_rate_below_min"
    assert rejected["ret_no_pf"] == "profit_factor_missing"
    assert rejected["ret_small_sample"] == "sample_count_below_min"
    assert rejected["combo__weak__carry"] == "profit_factor_below_min"


def test_simulation_slots_api_returns_cache_unavailable(monkeypatch, tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)
    _init_db(db_path, monkeypatch)

    payload = auto_trade_api.read_simulation_slots(symbol="BTCUSDT", duration="10m")

    reasons = {row["rejectionReason"] for row in payload["items"]}
    assert "cache_unavailable" in reasons
    assert payload["singleFactorSlots"] == 0
    assert payload["comboFactorSlots"] == 0
    assert payload["enabledSlots"] == 0


def test_strategy_payloads_include_simulation_status(monkeypatch, tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)
    _init_db(db_path, monkeypatch)
    _seed_klines(db_path)
    _save_single_cache(db_path)
    qualified_factor_simulation_slots.sync_qualified_simulation_slots("BTCUSDT", "10m")

    payload = auto_trade_api.read_strategies()

    slot = next(row for row in payload["strategies"] if row["strategyKey"] == factor_candidate_signal_key("ret_good"))
    assert slot["simulationStatus"]["gateStatus"] == "enabled"
    assert slot["simulationStatus"]["candidateType"] == "single_factor"


def _db_path(tmp_path: Path) -> Path:
    return tmp_path / "simulation_slots.db"


def _init_db(db_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(session, "DB_PATH", db_path)
    for module in (
        factor_ranking_cache_service,
        factor_combination_cache_service,
        high_winrate_combo_cache_service,
        paper_live_failure_store,
        qualified_factor_simulation_slots,
        simulation_slot_runtime,
        factor_cache_metadata,
    ):
        monkeypatch.setattr(module, "get_conn", lambda db_path=db_path: _connect(db_path))
    session.init_db()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _seed_klines(db_path: Path) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO klines(symbol, interval, open_time, open, high, low, close, volume, close_time)
            VALUES('BTCUSDT', '10m', 1000, 1, 2, 1, 2, 10, 1600)
            """
        )
        conn.commit()
    finally:
        conn.close()


def _save_single_cache(db_path: Path) -> None:
    _insert_cache(
        db_path,
        "factor_ranking_cache",
        [
            {"factorName": "ret_good", "winRate": 0.71, "profitFactor": 1.4, "totalPeriods": 140},
            {"factorName": "ret_weak", "winRate": 0.5, "profitFactor": 1.4, "totalPeriods": 140},
            {"factorName": "ret_no_pf", "winRate": 0.71, "totalPeriods": 140},
            {"factorName": "ret_small_sample", "winRate": 0.71, "profitFactor": 1.4, "totalPeriods": 5},
        ],
    )


def _save_combo_cache(db_path: Path) -> None:
    _insert_cache(
        db_path,
        "factor_combo_ranking_cache",
        [
            {"factorName": "combo__good__carry", "winRate": 0.72, "profitFactor": 1.5, "totalPeriods": 140},
            {"factorName": "combo__weak__carry", "winRate": 0.72, "profitFactor": 0.8, "totalPeriods": 140},
        ],
    )


def _insert_cache(db_path: Path, table: str, ranking: list[dict]) -> None:
    payload = json.dumps({"ranking": ranking, "cacheMeta": _cache_meta()}, ensure_ascii=False)
    conn = _connect(db_path)
    try:
        if table == "factor_ranking_cache":
            _insert_factor_cache(conn, table, ranking, payload)
        else:
            _insert_combo_cache(conn, table, ranking, payload)
        conn.commit()
    finally:
        conn.close()


def _insert_factor_cache(conn: sqlite3.Connection, table: str, ranking: list[dict], payload: str) -> None:
    conn.execute(
        f"""
        INSERT INTO {table}(symbol, duration, updated_at, total, payload)
        VALUES('BTCUSDT', '10m', '2026-01-01T00:00:00+00:00', ?, ?)
        """,
        (len(ranking), payload),
    )


def _insert_combo_cache(conn: sqlite3.Connection, table: str, ranking: list[dict], payload: str) -> None:
    conn.execute(
        f"""
        INSERT INTO {table}(symbol, duration, updated_at, total, payload, search_config)
        VALUES('BTCUSDT', '10m', '2026-01-01T00:00:00+00:00', ?, ?, '{{}}')
        """,
        (len(ranking), payload),
    )


def _cache_meta() -> dict:
    return {
        "schemaVersion": 2,
        "symbol": "BTCUSDT",
        "duration": "10m",
        "marketData": {"rowCount": 1, "maxOpenTime": 1000},
    }


def _seed_settled_event(db_path: Path, strategy_key: str) -> None:
    conn = _connect(db_path)
    try:
        event_id = conn.execute(
            """
            INSERT INTO events(strategy_key, symbol, title, event_interval, rule_type, strike_value,
              start_time, end_time, status, result, ai_high_winrate_rule)
            VALUES(?, 'BTCUSDT', 'combo sim', '10m', 'ABOVE', 100,
              '2026-01-01T00:00:00+00:00', '2026-01-01T00:10:00+00:00', 'SETTLED', 'YES', 'combo__good__carry')
            """,
            (strategy_key,),
        ).lastrowid
        order_id = conn.execute(
            """
            INSERT INTO orders(event_id, side, price, qty, status, created_at, external_status)
            VALUES(?, 'BUY', 0.65, 5, 'OPEN', '2026-01-01T00:00:00+00:00', 'SIMULATED')
            """,
            (event_id,),
        ).lastrowid
        conn.execute(
            "INSERT INTO settlements(event_id, order_id, pnl, settled_at) VALUES(?, ?, 3.25, '2026-01-01T00:11:00+00:00')",
            (event_id, order_id),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_prediction_failure(db_path: Path, strategy_key: str) -> None:
    conn = _connect(db_path)
    try:
        paper_live_failure_store.ensure_prediction_failure_table(conn)
        conn.execute(
            """
            INSERT INTO paper_live_prediction_failures(
              candidate_key, strategy_key, symbol, duration, stage, reason, details_json, created_at
            )
            VALUES(?, ?, 'BTCUSDT', '10m', 'factor_candidate_signal', ?, '{}', '2026-01-01T00:02:00+00:00')
            """,
            (strategy_key, strategy_key, "factor candidate signal missing column"),
        )
        conn.commit()
    finally:
        conn.close()
