from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from app.services.auto_predict_loop_status import (
    record_auto_predict_current_task,
    record_auto_predict_current_task_done,
    record_auto_predict_current_task_progress,
)
from app.services.auto_trade_types import AutoTradeSettings
from app.services.paper_live_candidate_service import log_prediction_failure

logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class ModelFamilyShadowContext:
    family: str
    collection: Any


@dataclass(frozen=True)
class ModelFamilyShadowOutcome:
    family: str
    result: dict[str, Any] | None
    saved: bool


@dataclass(frozen=True)
class ModelPredictionFailureContext:
    settings: AutoTradeSettings
    family: str
    entry_open_time: int
    exc: Exception
    status: dict[str, Any]


async def save_one_model_family_shadow_prediction(
    context: ModelFamilyShadowContext,
    *,
    create_simulation_trade: bool = True,
) -> ModelFamilyShadowOutcome:
    family = context.family
    collection = context.collection
    settings = collection.settings
    task = _model_shadow_task(settings, family, collection.entry_open_time)
    timings: dict[str, Any] = {}
    task_id = record_auto_predict_current_task({**task, "operation": "status_check"})
    started = time.perf_counter()
    try:
        outcome = await _save_one_model_family_shadow_prediction(
            context,
            timings,
            task_id,
            create_simulation_trade=create_simulation_trade,
        )
    except Exception as exc:
        timings["totalSeconds"] = _elapsed(started)
        _record_model_shadow_done(timings, "failed", exc, task_id)
        logger.exception(
            "model family shadow failed family=%s symbol=%s duration=%s entry=%s",
            family,
            settings.symbol,
            settings.duration,
            collection.entry_open_time,
        )
        raise
    timings["totalSeconds"] = _elapsed(started)
    _record_model_shadow_done(timings, str(timings.get("outcome") or "completed"), None, task_id)
    _log_model_shadow_timing(task, timings)
    return outcome


async def _save_one_model_family_shadow_prediction(
    context: ModelFamilyShadowContext,
    timings: dict[str, Any],
    task_id: str,
    *,
    create_simulation_trade: bool,
) -> ModelFamilyShadowOutcome:
    status = await _model_shadow_status(context, timings)
    if not status.get("shadowPredictionReady"):
        timings["outcome"] = "skipped_not_ready"
        _record_model_shadow_skip(context, status)
        return ModelFamilyShadowOutcome(context.family, None, False)
    result = await _model_shadow_prediction(context, status, timings, task_id)
    saved = await _save_model_shadow_result(
        context,
        result,
        timings,
        task_id,
        create_simulation_trade=create_simulation_trade,
    )
    timings["outcome"] = "saved" if saved else "already_exists"
    return ModelFamilyShadowOutcome(context.family, result, bool(saved))


async def _model_shadow_status(
    context: ModelFamilyShadowContext,
    timings: dict[str, Any],
) -> dict[str, Any]:
    family = context.family
    collection = context.collection
    deps = collection.deps
    settings = collection.settings
    status_loader = deps["lstm_model_status"] if family == "lstm" else deps["model_family_status"]
    status_args = (settings.symbol, settings.duration) if family == "lstm" else (family, settings.symbol, settings.duration)
    return await _timed("statusSeconds", timings, asyncio.to_thread(status_loader, *status_args))


async def _model_shadow_prediction(
    context: ModelFamilyShadowContext,
    status: dict[str, Any],
    timings: dict[str, Any],
    task_id: str,
) -> dict[str, Any]:
    family = context.family
    collection = context.collection
    deps = collection.deps
    settings = collection.settings
    predictor = deps["predict_lstm_shadow_prediction"] if family == "lstm" else deps["predict_model_family_shadow_prediction"]
    args = (settings.symbol, settings.duration) if family == "lstm" else (family, settings.symbol, settings.duration)
    record_auto_predict_current_task_progress(task_id, operation="prediction")
    try:
        prediction_kwargs = {"entry_open_time": collection.entry_open_time}
        if family != "lstm":
            prediction_kwargs["timings"] = timings
            prediction_kwargs["cycle_context"] = collection.deps.get("prediction_cycle_context")
        result = await _timed(
            "predictionSeconds",
            timings,
            asyncio.to_thread(predictor, *args, **prediction_kwargs),
        )
    except Exception as exc:
        _log_model_prediction_failure(ModelPredictionFailureContext(settings, family, collection.entry_open_time, exc, status))
        raise
    return result


async def _save_model_shadow_result(
    context: ModelFamilyShadowContext,
    result: dict[str, Any],
    timings: dict[str, Any],
    task_id: str,
    *,
    create_simulation_trade: bool,
) -> bool:
    collection = context.collection
    deps = collection.deps
    settings = collection.settings
    record_auto_predict_current_task_progress(task_id, operation="save_prediction")
    saved = await _timed("saveSeconds", timings, deps["save_prediction"](result, collection.write_lock))
    if saved and create_simulation_trade:
        record_auto_predict_current_task_progress(task_id, operation="simulation_trade")
        await _timed(
            "tradeSeconds",
            timings,
            save_model_family_shadow_trade(
                settings,
                _shadow_trade_payload(result),
                deps["create_batch_combo_simulation_trade"],
            ),
        )
    return bool(saved)


async def save_model_family_shadow_trade_for_outcome(
    parent: AutoTradeSettings,
    outcome: ModelFamilyShadowOutcome,
    create_trade: Callable[[AutoTradeSettings, dict], Any],
) -> None:
    if outcome.saved and outcome.result is not None:
        await save_model_family_shadow_trade(parent, _shadow_trade_payload(outcome.result), create_trade)


async def save_model_family_shadow_trade(
    parent: AutoTradeSettings,
    result: dict,
    create_trade: Callable[[AutoTradeSettings, dict], Any],
) -> None:
    if result.get("trade_quality_passed"):
        await asyncio.to_thread(create_trade, parent, result)


async def _timed(label: str, timings: dict[str, Any], awaitable):
    started = time.perf_counter()
    try:
        return await awaitable
    finally:
        timings[label] = _elapsed(started)


def _elapsed(started: float) -> float:
    return round(max(time.perf_counter() - started, 0.0), 6)


def _model_shadow_task(settings: AutoTradeSettings, family: str, entry_open_time: int) -> dict[str, Any]:
    return {
        "currentStage": "model_family_shadow_prediction",
        "currentFamily": family,
        "symbol": settings.symbol.upper(),
        "duration": settings.duration,
        "entryOpenTime": int(entry_open_time),
    }


def _record_model_shadow_done(
    timings: dict[str, Any],
    outcome: str,
    exc: Exception | None,
    task_id: str,
) -> None:
    payload = {**timings, "outcome": outcome}
    if exc is not None:
        payload.update({"error": str(exc), "exceptionType": type(exc).__name__})
    record_auto_predict_current_task_done(payload, task_id)


def _record_model_shadow_skip(context: ModelFamilyShadowContext, status: dict[str, Any]) -> None:
    family = context.family
    collection = context.collection
    settings = collection.settings
    _log_model_skip((settings, family, status, collection.entry_open_time))
    collection.deps["log_model_family_shadow_skip"](settings, family, status, role="sidecar")


def _log_model_shadow_timing(task: dict[str, Any], timings: dict[str, Any]) -> None:
    logger.info(
        "model family shadow timing family=%s symbol=%s duration=%s entry=%s timings=%s",
        task["currentFamily"],
        task["symbol"],
        task["duration"],
        task["entryOpenTime"],
        timings,
    )


def _shadow_trade_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {**result, "trade_quality_passed": True}


def _log_model_skip(context: tuple[AutoTradeSettings, str, dict[str, Any], int]) -> None:
    settings, family, status, entry_open_time = context
    reason = str(status.get("shadowPredictionBlockedReason") or status.get("reason") or "model_shadow_not_ready")
    log_prediction_failure(
        candidate_key=f"{family}:{settings.symbol.upper()}:{settings.duration}",
        strategy_key=settings.strategy_key,
        symbol=settings.symbol,
        duration=settings.duration,
        stage="model_shadow_readiness",
        reason=reason,
        details={"family": family, "entryOpenTime": int(entry_open_time), "status": status.get("status")},
    )


def _log_model_prediction_failure(context: ModelPredictionFailureContext) -> None:
    log_prediction_failure(
        candidate_key=f"{context.family}:{context.settings.symbol.upper()}:{context.settings.duration}",
        strategy_key=context.settings.strategy_key,
        symbol=context.settings.symbol,
        duration=context.settings.duration,
        stage="model_shadow_prediction",
        reason=str(context.exc),
        details={
            "family": context.family,
            "entryOpenTime": int(context.entry_open_time),
            "exceptionType": type(context.exc).__name__,
            "modelVersion": context.status.get("modelVersion"),
            "status": context.status.get("status"),
        },
    )
