from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.db import session
from app.services import (
    factor_cache_metadata,
    factor_combination_cache_service,
    factor_ranking_cache_service,
    high_winrate_combo_cache_service,
    paper_live_failure_store,
    qualified_factor_simulation_slots,
    simulation_slot_runtime,
)
from app.services.factor_candidate_signal_keys import factor_candidate_signal_key
from app.services.factor_combo_simulation_keys import simulation_strategy_key_for_factor_name
from app.services.factor_simulation_trace_service import factor_simulation_trace


@pytest.fixture()
def trace_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    db_path = tmp_path / "trace.db"
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
    _seed_klines(db_path)
    return db_path


def test_single_factor_trace_exposes_full_simulation_lifecycle(trace_db: Path) -> None:
    _save_cache(trace_db, "factor_ranking_cache", [{"factorName": "ret_good", **_good_metrics()}])
    key = factor_candidate_signal_key("ret_good")
    qualified_factor_simulation_slots.sync_qualified_simulation_slots("BTCUSDT", "10m", qty=6)
    _seed_prediction(trace_db, key, "ret_good")
    _seed_settled_event(trace_db, key, "ret_good", pnl=2.5)

    payload = factor_simulation_trace("btcusdt", "10m", factor_name="ret_good", strategy_key=None)

    assert payload["kind"] == "single_factor"
    assert payload["found"] is True
    assert payload["status"] == "simulation_enabled"
    assert payload["ranking"]["rank"] == 1
    assert payload["gate"]["status"] == "enabled"
    assert payload["simulationSlot"]["enabled"] is True
    assert payload["latestPrediction"]["highWinrateRule"] == "ret_good"
    assert payload["order"]["externalStatus"] == "SIMULATED"
    assert payload["settlement"]["pnl"] == 2.5
    assert payload["observation"]["metrics"]["sampleCount"] == 1


def test_combo_trace_exposes_ranking_slot_event_and_observation(trace_db: Path) -> None:
    factor = "combo__good__carry"
    key = simulation_strategy_key_for_factor_name(factor)
    _save_cache(trace_db, "factor_combo_ranking_cache", [{"factorName": factor, **_good_metrics()}])
    qualified_factor_simulation_slots.sync_qualified_simulation_slots("BTCUSDT", "10m", qty=4)
    _seed_prediction(trace_db, key, factor)
    _seed_settled_event(trace_db, key, factor, pnl=-1.5)

    payload = factor_simulation_trace("BTCUSDT", "10m", factor_name=factor, strategy_key=None)

    assert payload["kind"] == "factor_combo"
    assert payload["strategyKey"] == key
    assert payload["ranking"]["source"] == "factor_combo_ranking_cache"
    assert payload["order"]["externalStatus"] == "SIMULATED"
    assert payload["settlement"]["status"] == "settled"
    assert payload["observation"]["status"] == "collecting"


def test_trace_reports_no_simulation_event_explicitly(trace_db: Path) -> None:
    _save_cache(trace_db, "factor_ranking_cache", [{"factorName": "ret_good", **_good_metrics()}])
    qualified_factor_simulation_slots.sync_qualified_simulation_slots("BTCUSDT", "10m")

    payload = factor_simulation_trace("BTCUSDT", "10m", factor_name="ret_good", strategy_key=None)

    assert payload["found"] is True
    assert "no_simulation_events" in payload["issues"]
    assert payload["order"]["reason"] == "no_simulation_events"
    assert payload["settlement"]["status"] == "none"


def test_trace_reports_cache_missing_without_fake_success(trace_db: Path) -> None:
    payload = factor_simulation_trace("BTCUSDT", "10m", factor_name="ret_missing", strategy_key=None)

    assert payload["found"] is False
    assert payload["status"] == "not_found"
    assert "cache_missing" in payload["issues"]
    assert payload["ranking"]["status"] == "unavailable"


def test_trace_reports_watchlist_status(monkeypatch: pytest.MonkeyPatch, trace_db: Path) -> None:
    _save_cache(trace_db, "factor_ranking_cache", [{"factorName": "ret_good", **_good_metrics()}])
    key = factor_candidate_signal_key("ret_good")
    qualified_factor_simulation_slots.sync_qualified_simulation_slots("BTCUSDT", "10m")
    monkeypatch.setattr(
        "app.services.factor_simulation_trace_service.evaluate_factor_candidate_event_demotion",
        lambda *_args: {"evaluations": [{"strategyKey": key, "status": "demoted", "reason": "consecutive_losses", "metrics": {"sampleCount": 30}}]},
    )

    payload = factor_simulation_trace("BTCUSDT", "10m", factor_name="ret_good", strategy_key=None)

    assert payload["status"] == "watchlist"
    assert "demotion_watchlist" in payload["issues"]
    assert payload["observation"]["reason"] == "consecutive_losses"


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


def _save_cache(db_path: Path, table: str, ranking: list[dict]) -> None:
    payload = json.dumps({"ranking": ranking, "cacheMeta": _cache_meta()}, ensure_ascii=False)
    conn = _connect(db_path)
    try:
        if table == "factor_ranking_cache":
            sql = "INSERT INTO factor_ranking_cache(symbol, duration, updated_at, total, payload) VALUES(?, ?, ?, ?, ?)"
            conn.execute(sql, ("BTCUSDT", "10m", "2026-01-01T00:00:00+00:00", len(ranking), payload))
        else:
            sql = f"INSERT INTO {table}(symbol, duration, updated_at, total, search_config, payload) VALUES(?, ?, ?, ?, ?, ?)"
            conn.execute(sql, ("BTCUSDT", "10m", "2026-01-01T00:00:00+00:00", len(ranking), "{}", payload))
        conn.commit()
    finally:
        conn.close()


def _cache_meta() -> dict:
    return {"schemaVersion": 2, "symbol": "BTCUSDT", "duration": "10m", "marketData": {"rowCount": 1, "maxOpenTime": 1000}}


def _good_metrics() -> dict:
    return {"winRate": 0.72, "profitFactor": 1.4, "totalPeriods": 140, "factorScore": 88}


def _seed_prediction(db_path: Path, strategy_key: str, factor_name: str) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO predictions(signal_key, strategy_key, symbol, duration, open_time, direction, probability_up,
              confidence, certainty_label, trade_quality_passed, high_winrate_rule, created_at)
            VALUES(?, ?, 'BTCUSDT', '10m', 1000, 'up', 0.72, 0.72, 'FACTOR', 1, ?, '2026-01-01T00:00:00+00:00')
            """,
            (strategy_key, strategy_key, factor_name),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_settled_event(db_path: Path, strategy_key: str, factor_name: str, *, pnl: float) -> None:
    conn = _connect(db_path)
    try:
        event_id = conn.execute(
            """
            INSERT INTO events(strategy_key, symbol, title, event_interval, rule_type, strike_value,
              start_time, end_time, status, result, ai_high_winrate_rule, market_regime_gate_passed)
            VALUES(?, 'BTCUSDT', 'sim', '10m', 'ABOVE', 100,
              '2026-01-01T00:00:00+00:00', '2026-01-01T00:10:00+00:00', 'SETTLED', 'YES', ?, 1)
            """,
            (strategy_key, factor_name),
        ).lastrowid
        order_id = conn.execute(
            """
            INSERT INTO orders(event_id, side, price, qty, status, created_at, external_status)
            VALUES(?, 'BUY', 0.65, 5, 'OPEN', '2026-01-01T00:00:00+00:00', 'SIMULATED')
            """,
            (event_id,),
        ).lastrowid
        conn.execute(
            "INSERT INTO settlements(event_id, order_id, pnl, settled_at) VALUES(?, ?, ?, '2026-01-01T00:11:00+00:00')",
            (event_id, order_id, pnl),
        )
        conn.commit()
    finally:
        conn.close()
