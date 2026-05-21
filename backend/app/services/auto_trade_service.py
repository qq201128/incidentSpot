from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.auto_trade_execution import create_trade_from_prediction
from app.services.auto_trade_types import AutoTradeSettings
from app.services.factor_combo_simulation_keys import is_batch_combo_simulation_strategy
from app.services.position_guard import has_open_position
from app.services.kline_timing import current_rule_entry_open_time_for_duration
from app.services.prediction_policy import trade_confidence_threshold_for_duration, trade_policy_payload
from app.services.rule_config import DURATION_TO_MINUTES
from app.services.high_winrate_strategy_demotion import STATUS_TRADABLE, high_winrate_demotion_status
from app.services.strategy_registry import (
    DEFAULT_STRATEGY_KEY,
    HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
    strategy_entry_grace_ms,
    strategy_definition,
    strategy_payloads,
    strategy_supports_duration,
    strategy_uses_trade_policy_gates,
)
from app.services.model_family_config import is_model_family_shadow_strategy

logger = logging.getLogger("uvicorn.error")

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_DURATION = "10m"
DEFAULT_DURATION_MINUTES = 10
DEFAULT_QTY = 5.0
SUPPORTED_AUTO_DURATIONS = frozenset({"10m", "30m", "60m", "1d"})
AUTO_TRADE_SLOT_DURATIONS: tuple[str, ...] = ("10m", "30m", "60m", "1d")
MS_PER_SECOND = 1000


async def auto_trade_loop(stop_event: asyncio.Event, poll_seconds: int = 1) -> None:
    logger.info("auto trade loop: running every %ss", poll_seconds)
    while not stop_event.is_set():
        started = asyncio.get_running_loop().time()
        try:
            results = await asyncio.to_thread(run_auto_trade_once)
            if results:
                logger.info("auto trade placed %s strategy order(s)", len(results))
        except Exception:
            logger.exception("auto trade failed")
        await _sleep_until_next_tick(stop_event, started, poll_seconds)


def list_auto_trade_settings() -> list[AutoTradeSettings]:
    """每个可交易策略 × 每个结算周期一条配置（可同时开启多周期）。"""
    conn = get_conn()
    try:
        rows = conn.execute("SELECT * FROM auto_trade_strategies").fetchall()
        by_pair = {(str(r["strategy_key"]), str(r["duration"])): r for r in rows}
        result: list[AutoTradeSettings] = []
        for payload in strategy_payloads():
            key = str(payload["key"])
            for dur in _payload_durations(payload):
                row = by_pair.get((key, dur))
                if row is not None:
                    result.append(_settings_from_row(row))
                else:
                    result.append(_default_settings(key, dur))
        return result
    finally:
        conn.close()


def list_auto_trade_strategy_payloads() -> list[dict[str, Any]]:
    return [_strategy_payload(settings) for settings in list_auto_trade_settings()]


def get_auto_trade_settings(strategy_key: str = DEFAULT_STRATEGY_KEY) -> AutoTradeSettings:
    strategy = strategy_definition(strategy_key)
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM auto_trade_strategies WHERE strategy_key = ? AND duration = ?",
            (strategy.key, DEFAULT_DURATION),
        ).fetchone()
        if row is not None:
            return _settings_from_row(row)
        settings = _default_settings(strategy.key, DEFAULT_DURATION)
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
    if _blocked_by_high_winrate_live_state(settings):
        return False
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


def _blocked_by_high_winrate_live_state(settings: AutoTradeSettings) -> bool:
    if settings.strategy_key != HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY:
        return False
    if not settings.live_trading_enabled:
        return False
    status = high_winrate_demotion_status(settings.symbol, settings.duration)
    return status.get("status") != STATUS_TRADABLE


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
        return has_open_position(conn, symbol, strategy_key, event_interval=duration)
    finally:
        conn.close()


def _validated_settings(settings: AutoTradeSettings) -> AutoTradeSettings:
    strategy = strategy_definition(settings.strategy_key)
    if settings.enabled and not strategy.tradable:
        raise ValueError(strategy.disabled_reason or f"strategy is not tradable: {strategy.key}")
    if is_model_family_shadow_strategy(strategy.key) and settings.live_trading_enabled:
        raise ValueError("model family shadow strategy supports simulation only; live trading must stay disabled")
    if is_batch_combo_simulation_strategy(strategy.key) and settings.live_trading_enabled:
        raise ValueError("batch factor combo strategy supports simulation only; live trading must stay disabled")
    symbol = settings.symbol.strip().upper()
    if len(symbol) < 6:
        raise ValueError("symbol must contain at least 6 characters")
    if settings.enabled and settings.duration not in SUPPORTED_AUTO_DURATIONS:
        raise ValueError(
            "backend auto trade duration must be one of "
            + ", ".join(sorted(SUPPORTED_AUTO_DURATIONS))
        )
    if settings.enabled and not strategy_supports_duration(settings.strategy_key, settings.duration):
        raise ValueError(
            f"strategy {strategy.key} does not support duration {settings.duration}, "
            f"supported: {', '.join(sorted(strategy.supported_durations))}"
        )
    if settings.duration_minutes <= 0:
        raise ValueError("durationMinutes must be > 0")
    if settings.qty <= 0:
        raise ValueError("qty must be > 0")
    canonical_minutes = DURATION_TO_MINUTES[settings.duration]
    return AutoTradeSettings(
        strategy_key=strategy.key,
        enabled=settings.enabled,
        symbol=symbol,
        duration=settings.duration,
        duration_minutes=canonical_minutes,
        qty=settings.qty,
        live_trading_enabled=settings.live_trading_enabled,
    )


def _write_settings(conn: Any, settings: AutoTradeSettings) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO auto_trade_strategies(
          strategy_key, duration, enabled, live_trading_enabled, symbol, duration_minutes, qty, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            settings.strategy_key,
            settings.duration,
            int(settings.enabled),
            int(settings.live_trading_enabled),
            settings.symbol,
            settings.duration_minutes,
            settings.qty,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _settings_from_row(row: Any) -> AutoTradeSettings:
    return AutoTradeSettings(
        strategy_key=_row_strategy_key(row),
        enabled=bool(row["enabled"]),
        symbol=str(row["symbol"]).upper(),
        duration=str(row["duration"]),
        duration_minutes=int(row["duration_minutes"]),
        qty=float(row["qty"]),
        live_trading_enabled=bool(row["live_trading_enabled"]),
    )


def _default_settings(strategy_key: str = DEFAULT_STRATEGY_KEY, duration: str = DEFAULT_DURATION) -> AutoTradeSettings:
    minutes = int(DURATION_TO_MINUTES.get(duration, DEFAULT_DURATION_MINUTES))
    return AutoTradeSettings(
        strategy_key=strategy_key,
        enabled=False,
        symbol=DEFAULT_SYMBOL,
        duration=duration,
        duration_minutes=minutes,
        qty=DEFAULT_QTY,
        live_trading_enabled=False,
    )


def _strategy_payload(settings: AutoTradeSettings) -> dict[str, Any]:
    strategy = strategy_definition(settings.strategy_key)
    return {
        **settings.to_response(),
        "name": strategy.name,
        "description": strategy.description,
        "requiresVegasConfirmation": strategy.requires_vegas_confirmation,
        "requiresHighWinrateGate": strategy.requires_high_winrate_gate,
        "requiresTradeQualityGate": strategy.requires_trade_quality_gate,
        "signalSource": strategy.signal_source,
        "ruleNames": strategy.rule_names,
        "tradable": strategy.tradable,
        "disabledReason": strategy.disabled_reason,
        "backtestSummary": strategy.backtest_summary,
        "minDailyTrades": strategy.min_daily_trades,
        "requiresKlineFeatures": strategy.requires_kline_features,
        "usesTradePolicyGates": strategy.uses_trade_policy_gates,
        "entryGraceMs": strategy.entry_grace_ms,
        "supportedDurations": sorted(strategy.supported_durations),
        "demotion": _demotion_payload(settings),
    }


def _demotion_payload(settings: AutoTradeSettings) -> dict[str, Any] | None:
    if settings.strategy_key != HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY:
        return None
    return high_winrate_demotion_status(settings.symbol, settings.duration)


def _payload_durations(payload: dict[str, Any]) -> list[str]:
    durations = payload.get("supportedDurations") or AUTO_TRADE_SLOT_DURATIONS
    return [duration for duration in AUTO_TRADE_SLOT_DURATIONS if duration in set(durations)]


def _row_strategy_key(row: Any) -> str:
    return str(row["strategy_key"] or DEFAULT_STRATEGY_KEY)


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _production_target_passed(policy: dict[str, Any]) -> bool:
    target = policy.get("productionTarget") or {}
    return target.get("passed") is True


async def _sleep_until_next_tick(stop_event: asyncio.Event, started: float, poll_seconds: int) -> None:
    elapsed = asyncio.get_running_loop().time() - started
    wait_seconds = max(float(poll_seconds) - elapsed, 0.0)
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=wait_seconds)
    except TimeoutError:
        return
