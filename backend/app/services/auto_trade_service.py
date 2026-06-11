from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.background_loop_status import record_loop_failure, record_loop_start, record_loop_stopped, record_loop_success
from app.services.auto_trade_execution import create_trade_from_prediction
from app.services.auto_trade_types import AutoTradeSettings
from app.services.auto_trade_settings_payloads import (
    AUTO_TRADE_SLOT_DURATIONS,
    DEFAULT_DURATION,
    DEFAULT_SYMBOL,
    default_settings as _default_settings,
    ensemble_ranker_settings as _ensemble_ranker_settings,
    payload_durations as _payload_durations,
    settings_from_row as _settings_from_row,
    strategy_payload as _strategy_payload,
    write_settings as _write_settings,
)
from app.services.auto_trade_strategy_observability_payloads import with_simulation_status
from app.services.auto_trade_settings_validation import validated_auto_trade_settings
from app.services.factor_candidate_signal_keys import is_factor_candidate_signal_key
from app.services.factor_combo_simulation_keys import is_batch_combo_simulation_strategy
from app.services.position_guard import has_open_position
from app.services.kline_timing import current_rule_entry_open_time_for_duration
from app.services.market_regime_trade_gate import evaluate_market_regime_trade_gate
from app.services.prediction_policy import trade_confidence_threshold_for_duration, trade_policy_payload
from app.services.runtime_symbols import configured_runtime_symbols
from app.services.strategy_registry import (
    DEFAULT_STRATEGY_KEY,
    strategy_entry_grace_ms,
    strategy_definition,
    strategy_payloads,
    strategy_uses_trade_policy_gates,
)

logger = logging.getLogger("uvicorn.error")

MS_PER_SECOND = 1000
LOOP_NAME = "auto_trade"


async def auto_trade_loop(stop_event: asyncio.Event, poll_seconds: int = 1) -> None:
    logger.info("auto trade loop: running every %ss", poll_seconds)
    record_loop_start(LOOP_NAME, {"pollSeconds": poll_seconds})
    if stop_event.is_set():
        record_loop_stopped(LOOP_NAME, "stop_before_first_tick")
        return
    while not stop_event.is_set():
        started = asyncio.get_running_loop().time()
        try:
            results = await asyncio.to_thread(run_auto_trade_once)
            record_loop_success(LOOP_NAME, {"placedCount": len(results)})
            if results:
                logger.info("auto trade placed %s strategy order(s)", len(results))
        except Exception as exc:
            record_loop_failure(LOOP_NAME, exc)
            logger.exception("auto trade failed")
        if await _sleep_until_next_tick(stop_event, started, poll_seconds):
            record_loop_stopped(LOOP_NAME, "stop_between_ticks")
            return


def list_auto_trade_settings() -> list[AutoTradeSettings]:
    """每个可交易执行项 × 交易对 × 结算周期一条配置。"""
    conn = get_conn()
    try:
        rows = conn.execute("SELECT * FROM auto_trade_strategies").fetchall()
        by_slot = _settings_by_slot(rows)
        symbols = configured_runtime_symbols()
        static_keys = _strategy_slot_keys(symbols)
        result = _strategy_slot_settings(by_slot, static_keys)
        result.extend(_dynamic_simulation_settings(rows, set(static_keys)))
        result.extend(_ensemble_ranker_settings(conn, by_slot, symbols))
        return result
    finally:
        conn.close()


def list_auto_trade_strategy_payloads() -> list[dict[str, Any]]:
    payloads = [_strategy_payload(settings) for settings in list_auto_trade_settings()]
    return with_simulation_status(payloads)


def get_auto_trade_settings(strategy_key: str = DEFAULT_STRATEGY_KEY, symbol: str = DEFAULT_SYMBOL) -> AutoTradeSettings:
    strategy = strategy_definition(strategy_key)
    sym = symbol.strip().upper()
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM auto_trade_strategies WHERE strategy_key = ? AND symbol = ? AND duration = ?",
            (strategy.key, sym, DEFAULT_DURATION),
        ).fetchone()
        if row is not None:
            return _settings_from_row(row)
        settings = _default_settings(strategy.key, DEFAULT_DURATION, sym)
        _write_settings(conn, settings)
        conn.commit()
        return settings
    finally:
        conn.close()


def update_auto_trade_settings(settings: AutoTradeSettings) -> AutoTradeSettings:
    validated = _validated_settings(settings)
    conn = get_conn()
    try:
        _write_settings(conn, validated)
        conn.commit()
        return validated
    finally:
        conn.close()


def run_auto_trade_once() -> list[dict[str, Any]]:
    results = []
    for settings in list_auto_trade_settings():
        result = _run_strategy_once(settings)
        if result is not None:
            results.append(result)
    return results


def _run_strategy_once(settings: AutoTradeSettings) -> dict[str, Any] | None:
    if not settings.enabled or _has_open_position(settings.symbol, settings.strategy_key, settings.duration):
        return None
    prediction = _latest_prediction_row(settings)
    if prediction is None:
        return None
    if not _prediction_matches_current_kline_bucket(prediction, settings):
        return None
    if not _is_fresh_prediction(prediction, settings):
        return None
    if not _is_prediction_tradable(prediction, settings):
        return None
    regime_decision = evaluate_market_regime_trade_gate(
        symbol=settings.symbol,
        duration=settings.duration,
        open_time=int(prediction["open_time"]),
        direction=str(prediction["direction"]),
    )
    if not regime_decision.allowed:
        return None
    result = _create_trade(settings, prediction)
    logger.info(
        "auto trade placed strategy=%s event=%s order=%s",
        settings.strategy_key,
        result.get("eventId"),
        result.get("orderId"),
    )
    return result


def _create_trade(settings: AutoTradeSettings, prediction: dict[str, Any]) -> dict:
    return create_trade_from_prediction(settings, prediction)


def _prediction_matches_current_kline_bucket(
    prediction: dict[str, Any], settings: AutoTradeSettings
) -> bool:
    """预测记录的 open_time 须与当前 UTC 周期桶起点一致，避免误用其它桶的缓存预测。"""
    now_ms = int(datetime.now(timezone.utc).timestamp() * MS_PER_SECOND)
    bucket = int(current_rule_entry_open_time_for_duration(settings.duration, now_ms))
    pred_ot = int(prediction["open_time"])
    if pred_ot == bucket:
        return True
    logger.debug(
        "auto trade skip bucket mismatch strategy=%s symbol=%s duration=%s pred_open=%s current_bucket=%s",
        settings.strategy_key,
        settings.symbol,
        settings.duration,
        pred_ot,
        bucket,
    )
    return False


def _is_prediction_tradable(prediction: dict[str, Any], settings: AutoTradeSettings) -> bool:
    if not strategy_uses_trade_policy_gates(settings.strategy_key):
        return _as_bool(prediction.get("trade_quality_passed")) is True
    probability = float(prediction["probability_up"])
    threshold = trade_confidence_threshold_for_duration(settings.duration)
    policy = trade_policy_payload(settings.duration, strategy_key=settings.strategy_key)
    if not _production_target_passed(policy):
        return False
    confidence_passed = max(probability, 1 - probability) >= threshold
    if bool(policy.get("highWinrateGateEnabled")):
        return confidence_passed and _as_bool(prediction.get("high_winrate_gate_passed")) is True
    score_min = float(policy.get("tradeQualityScoreMin") or 0)
    score = float(prediction.get("trade_quality_score") or 0)
    return confidence_passed and _as_bool(prediction.get("trade_quality_passed")) and score >= score_min


def _latest_prediction_row(settings: AutoTradeSettings) -> dict[str, Any] | None:
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT *
            FROM predictions
            WHERE strategy_key = ? AND symbol = ? AND duration = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (settings.strategy_key, settings.symbol, settings.duration),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _is_fresh_prediction(prediction: dict[str, Any], settings: AutoTradeSettings) -> bool:
    entry_open_time = int(prediction["open_time"])
    now_ms = int(datetime.now(timezone.utc).timestamp() * MS_PER_SECOND)
    current_bucket = int(current_rule_entry_open_time_for_duration(settings.duration, now_ms))
    if entry_open_time == current_bucket:
        return True
    now_ms = int(datetime.now(timezone.utc).timestamp() * MS_PER_SECOND)
    age_ms = now_ms - entry_open_time
    return 0 <= age_ms <= fresh_prediction_ms_for_strategy(settings.strategy_key)


def fresh_prediction_ms_for_strategy(strategy_key: str) -> int:
    return strategy_entry_grace_ms(strategy_key)


def _has_open_position(symbol: str, strategy_key: str, duration: str) -> bool:
    conn = get_conn()
    try:
        return has_open_position(
            conn,
            symbol,
            strategy_key,
            event_interval=duration,
            require_market_regime_gate_passed=True,
        )
    finally:
        conn.close()


def _settings_by_slot(rows) -> dict[tuple[str, str, str], Any]:
    return {(str(r["strategy_key"]), str(r["symbol"]).upper(), str(r["duration"])): r for r in rows}


def _strategy_slot_keys(symbols: tuple[str, ...]) -> list[tuple[str, str, str]]:
    return [
        (str(payload["key"]), symbol, dur)
        for payload in strategy_payloads()
        for symbol in symbols
        for dur in _payload_durations(payload)
    ]


def _strategy_slot_settings(
    by_slot: dict[tuple[str, str, str], Any],
    slot_keys: list[tuple[str, str, str]],
) -> list[AutoTradeSettings]:
    return [_slot_settings(by_slot, slot_key) for slot_key in slot_keys]


def _slot_settings(
    by_slot: dict[tuple[str, str, str], Any],
    slot_key: tuple[str, str, str],
) -> AutoTradeSettings:
    strategy_key, symbol, duration = slot_key
    row = by_slot.get(slot_key)
    return _settings_from_row(row) if row is not None else _default_settings(strategy_key, duration, symbol)


def _dynamic_simulation_settings(rows: list[Any], static_keys: set[tuple[str, str, str]]) -> list[AutoTradeSettings]:
    settings = [
        _settings_from_row(row)
        for row in rows
        if _dynamic_simulation_slot_key(row) not in static_keys
        and _is_dynamic_simulation_strategy(str(row["strategy_key"]))
    ]
    return sorted(settings, key=lambda item: (item.strategy_key, item.symbol, item.duration_minutes))


def _dynamic_simulation_slot_key(row: Any) -> tuple[str, str, str]:
    return (str(row["strategy_key"]), str(row["symbol"]).upper(), str(row["duration"]))


def _is_dynamic_simulation_strategy(strategy_key: str) -> bool:
    return is_batch_combo_simulation_strategy(strategy_key) or is_factor_candidate_signal_key(strategy_key)


def _find_auto_trade_settings(key: str, symbol: str, duration: str) -> AutoTradeSettings | None:
    for settings in list_auto_trade_settings():
        if settings.strategy_key == key and settings.symbol == symbol and settings.duration == duration:
            return settings
    return None


def _validated_settings(settings: AutoTradeSettings) -> AutoTradeSettings:
    return validated_auto_trade_settings(settings)


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _production_target_passed(policy: dict[str, Any]) -> bool:
    target = policy.get("productionTarget") or {}
    return target.get("passed") is True


async def _sleep_until_next_tick(stop_event: asyncio.Event, started: float, poll_seconds: int) -> bool:
    elapsed = asyncio.get_running_loop().time() - started
    wait_seconds = max(float(poll_seconds) - elapsed, 0.0)
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=wait_seconds)
        return True
    except TimeoutError:
        return False
