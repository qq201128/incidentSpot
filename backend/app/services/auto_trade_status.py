from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.auto_trade_service import fresh_prediction_ms_for_strategy, list_auto_trade_settings
from app.services.auto_trade_types import AutoTradeSettings
from app.services.kline_timing import current_rule_entry_open_time_for_duration
from app.services.position_guard import has_open_position
from app.services.prediction_policy import trade_confidence_threshold_for_duration, trade_policy_payload
from app.services.strategy_registry import DEFAULT_STRATEGY_KEY, strategy_uses_trade_policy_gates

MS_PER_SECOND = 1000


def get_auto_trade_status() -> dict[str, Any]:
    strategies = [_strategy_status(settings) for settings in list_auto_trade_settings()]
    default_status = _default_status(strategies)
    return {**default_status, "strategies": strategies}


def _strategy_status(settings: AutoTradeSettings) -> dict[str, Any]:
    prediction = _latest_prediction(settings)
    status = {
        "settings": settings.to_response(),
        "openPosition": _has_open_position(settings),
        "latestPrediction": _prediction_status(prediction, settings),
    }
    return {**status, "reason": _reason(status)}


def _default_status(strategies: list[dict[str, Any]]) -> dict[str, Any]:
    for status in strategies:
        settings = status["settings"]
        if settings["strategyKey"] == DEFAULT_STRATEGY_KEY and settings.get("duration") == "10m":
            return status
    return strategies[0] if strategies else {}


def _prediction_status(
    prediction: dict[str, Any] | None,
    settings: AutoTradeSettings,
) -> dict[str, Any] | None:
    if prediction is None:
        return None
    probability_up = float(prediction["probability_up"])
    policy = trade_policy_payload(settings.duration, strategy_key=settings.strategy_key)
    return {
        "id": prediction["id"],
        "strategyKey": settings.strategy_key,
        "createdAt": prediction["created_at"],
        "ageMs": _prediction_age_ms(prediction),
        "freshPredictionMs": fresh_prediction_ms_for_strategy(settings.strategy_key),
        "fresh": _prediction_is_fresh(prediction, settings),
        "direction": prediction["direction"],
        "probabilityUp": probability_up,
        "bestProbability": max(probability_up, 1 - probability_up),
        "confidenceThreshold": trade_confidence_threshold_for_duration(settings.duration),
        "qualityScore": prediction.get("trade_quality_score"),
        "qualityScoreMin": policy.get("tradeQualityScoreMin"),
        "qualityPassed": _as_bool(prediction.get("trade_quality_passed")),
        "highWinrateGate": prediction.get("high_winrate_gate"),
        "highWinrateGatePassed": _as_bool(prediction.get("high_winrate_gate_passed")),
        "highWinrateGateValue": prediction.get("high_winrate_gate_value"),
        "highWinrateGateEnabled": policy.get("highWinrateGateEnabled"),
        "productionTarget": policy.get("productionTarget"),
        "tradePolicyGatesEnabled": strategy_uses_trade_policy_gates(settings.strategy_key),
    }


def _reason(status: dict[str, Any]) -> str:
    if not status["settings"]["enabled"]:
        return "disabled"
    if status["openPosition"]:
        return "waiting_open_position_settled"
    prediction = status["latestPrediction"]
    if prediction is None:
        return "waiting_prediction"
    if not prediction["fresh"]:
        return "waiting_fresh_prediction"
    if not prediction["tradePolicyGatesEnabled"]:
        return _ungated_strategy_reason(prediction)
    if prediction["bestProbability"] < prediction["confidenceThreshold"]:
        return "confidence_below_threshold"
    if not _production_target_passed(prediction):
        return "production_target_failed"
    if prediction["highWinrateGateEnabled"]:
        return _high_winrate_reason(prediction)
    if not prediction["qualityPassed"]:
        return "quality_gate_failed"
    return "ready_to_place_order"


def _latest_prediction(settings: AutoTradeSettings) -> dict[str, Any] | None:
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


def _has_open_position(settings: AutoTradeSettings) -> bool:
    conn = get_conn()
    try:
        return has_open_position(
            conn,
            settings.symbol,
            settings.strategy_key,
            event_interval=settings.duration,
        )
    finally:
        conn.close()


def _prediction_age_ms(prediction: dict[str, Any]) -> int:
    now_ms = int(datetime.now(timezone.utc).timestamp() * MS_PER_SECOND)
    return max(now_ms - int(prediction["open_time"]), 0)


def _prediction_is_fresh(prediction: dict[str, Any], settings: AutoTradeSettings) -> bool:
    now_ms = int(datetime.now(timezone.utc).timestamp() * MS_PER_SECOND)
    entry_open_time = int(prediction["open_time"])
    current_bucket = current_rule_entry_open_time_for_duration(settings.duration, now_ms)
    if entry_open_time == current_bucket:
        return True
    age_ms = now_ms - entry_open_time
    return 0 <= age_ms <= fresh_prediction_ms_for_strategy(settings.strategy_key)


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _production_target_passed(prediction: dict[str, Any]) -> bool:
    target = prediction.get("productionTarget") or {}
    return target.get("passed") is True


def _high_winrate_reason(prediction: dict[str, Any]) -> str:
    if not prediction["highWinrateGatePassed"]:
        return "high_winrate_gate_failed"
    return "ready_to_place_order"


def _ungated_strategy_reason(prediction: dict[str, Any]) -> str:
    if prediction["qualityPassed"]:
        return "ready_to_place_order"
    return "signal_condition_not_met"
