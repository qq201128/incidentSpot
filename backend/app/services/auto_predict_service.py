from __future__ import annotations
import asyncio
import logging
from app.services.auto_predict_config import predict_initial_delay_seconds
from app.services.auto_trade_service import get_auto_trade_settings, list_auto_trade_settings
from app.services.auto_trade_types import AutoTradeSettings
from app.services.factor_combo_simulation_keys import simulation_strategy_key_for_factor_name
from app.services.factor_combo_batch_predictions import eligible_factor_combo_rows, predict_eligible_factor_combo_rows
from app.services.factor_combo_batch_simulation_service import create_batch_combo_simulation_trade
from app.services.factor_candidate_signal_keys import is_factor_candidate_signal_key
from app.services.factor_candidate_signal_service import factor_candidate_signal_keys, predict_factor_candidate_signals
from app.services.ensemble_judge_constants import ENSEMBLE_RANKER_STRATEGY_KEY
from app.services.event_final_decision_service import (
    EVENT_FINAL_DECISION_STRATEGY_KEY,
    event_final_decision_exists,
    predict_event_final_decision,
)
from app.services.ensemble_ranker_prediction_service import predict_ensemble_ranker_prediction
from app.services.kline_timing import current_rule_entry_open_time_for_duration, seconds_until_next_rule_entry_for_duration, utc_now_ms
from app.services.model_family_config import MODEL_FAMILIES, is_model_family_shadow_strategy, parse_model_family_strategy
from app.services.model_family_shadow_backfill_service import backfill_model_family_shadow_predictions, missing_model_family_shadow_entry_times
from app.services.prediction_cache_service import prediction_exists, prediction_response, save_prediction
from app.services.auto_predict_loop_status import record_auto_predict_cycle_failure, record_auto_predict_cycle_success, record_auto_predict_loop_start, record_auto_predict_loop_stopped
from app.services.rule_signal_service import predict_rule_direction
from app.services.strategy_registry import DEFAULT_STRATEGY_KEY, FACTOR_COMBO_STRATEGY_KEY, strategy_entry_grace_ms, strategy_supports_duration
from app.services.strategy_prediction_readiness import strategy_prediction_readiness
from app.services.model_family_prediction_service import predict_model_family_shadow_prediction
from app.services.model_family_prediction_runtime import PredictionCycleContext
from app.services.model_family_status_service import model_family_status
from app.services import auto_predict_targets as target_helpers
from app.services import auto_predict_candidate_collection as collection_helpers
from app.services import auto_predict_cycle_summary as cycle_helpers
from app.services import auto_predict_due_entries as due_entry_helpers
from app.services import auto_predict_input_deps as input_deps
from app.services import auto_predict_shadow as shadow_helpers
from app.services import auto_predict_runtime as runtime_helpers
logger = logging.getLogger("uvicorn.error")
_SUBSCRIBERS: dict[tuple[str, str, str], set] = {}
DEFAULT_PREDICT_SECONDS = 1
DEFAULT_DURATION = "10m"
FinalCandidateBatch = tuple[collection_helpers.CandidateCollectionContext, list[collection_helpers.ModelFamilyShadowOutcome]]


def _predict_initial_delay_seconds() -> float:
    return predict_initial_delay_seconds()


async def auto_predict_loop(stop_event: asyncio.Event, poll_seconds: int = DEFAULT_PREDICT_SECONDS) -> None:
    try:
        initial = _predict_initial_delay_seconds()
    except Exception as exc:
        record_auto_predict_cycle_failure(exc, float(poll_seconds))
        logger.exception("auto prediction startup failed")
        raise
    record_auto_predict_loop_start(initial_delay=initial, poll_seconds=poll_seconds)
    logger.info("predict loop: initial_delay=%ss poll=%ss", initial, poll_seconds)
    if initial > 0:
        if await runtime_helpers.sleep_for(stop_event, initial):
            record_auto_predict_loop_stopped("stop_during_initial_delay")
            return
    if stop_event.is_set():
        record_auto_predict_loop_stopped("stop_before_first_cycle")
        return
    logger.info("predict loop: running during each enabled strategy's kline entry window")
    while not stop_event.is_set():
        try:
            targets = await asyncio.to_thread(_prediction_targets)
            cycle_details = await _predict_due_entries(targets)
            wait_seconds = _next_predict_wait(targets, poll_seconds)
            record_auto_predict_cycle_success(len(targets), wait_seconds, cycle_details=cycle_details)
        except Exception as exc:
            logger.exception("auto prediction failed")
            wait_seconds = float(poll_seconds)
            record_auto_predict_cycle_failure(exc, wait_seconds)
        if await runtime_helpers.sleep_for(stop_event, wait_seconds):
            record_auto_predict_loop_stopped("stop_between_cycles")
            return
async def _predict_due_entries(targets: list[AutoTradeSettings]) -> dict[str, Any]:
    return await due_entry_helpers.predict_due_entries(targets, _predict_due_entry_deps())


def _predict_due_entry_deps() -> due_entry_helpers.PredictDueEntryDeps:
    return due_entry_helpers.PredictDueEntryDeps(
        ready_due_prediction_targets=_ready_due_prediction_targets,
        candidate_collection_targets=_candidate_collection_targets,
        prediction_cycle_targets=cycle_helpers.prediction_cycle_targets,
        record_prediction_progress=cycle_helpers.record_prediction_progress,
        prediction_cycle_summary=cycle_helpers.prediction_cycle_summary,
        prepare_prediction_inputs=_prepare_prediction_inputs,
        run_prediction_batch=_run_prediction_batch,
        run_candidate_collection_batch=_run_candidate_collection_batch,
        run_final_decision_required_candidate_batch=_run_final_decision_required_candidate_batch,
        run_observe_only_candidate_collection_batch=_run_observe_only_candidate_collection_batch,
        backfill_lstm_shadow_predictions=_backfill_lstm_shadow_predictions,
        logger=logger,
    )
async def _prepare_prediction_inputs(settings_list: list[AutoTradeSettings]) -> None:
    await runtime_helpers.prepare_prediction_inputs(settings_list, input_deps.runtime_deps())
async def _run_prediction_batch(settings_list: list[AutoTradeSettings]) -> None:
    await runtime_helpers.run_prediction_batch(settings_list, _run_prediction)
async def _run_prediction(settings: AutoTradeSettings, *, write_lock: asyncio.Lock) -> None:
    entry_open_time = current_rule_entry_open_time_for_duration(settings.duration)
    result = await _predict_strategy_result(settings, entry_open_time)
    if result is None:
        logger.info("predict: %s %s entry=%s -> SKIP", settings.symbol, settings.duration, entry_open_time)
        return
    if not await _save_prediction(result, write_lock):
        return
    await _broadcast(prediction_response(result))
    logger.info(
        "predict: %s %s entry=%s -> %s (conf=%.4f quality=%.4f qualityPassed=%s)",
        settings.symbol,
        settings.duration,
        entry_open_time,
        result["direction"],
        result["confidence"],
        result["trade_quality_score"],
        result["trade_quality_passed"],
    )
async def _run_candidate_collection_batch(settings_list: list[AutoTradeSettings]) -> None:
    await runtime_helpers.run_candidate_collection_batch(settings_list, _save_candidate_collection_predictions, logger)
async def _run_final_decision_required_candidate_batch(settings_list: list[AutoTradeSettings]) -> list[FinalCandidateBatch]:
    batches: list[FinalCandidateBatch] = []

    async def save_required(settings: AutoTradeSettings, *, write_lock: asyncio.Lock) -> None:
        context = _candidate_collection_context(settings, current_rule_entry_open_time_for_duration(settings.duration), write_lock)
        outcomes = await collection_helpers.save_final_decision_required_predictions(context)
        batches.append((context, outcomes))

    await runtime_helpers.run_candidate_collection_batch(settings_list, save_required, logger)
    return batches
async def _run_observe_only_candidate_collection_batch(
    settings_list: list[AutoTradeSettings],
    final_candidate_batches: list[FinalCandidateBatch],
) -> None:
    for context, outcomes in final_candidate_batches:
        await collection_helpers.save_model_family_shadow_simulation_trades(context, outcomes)
    await runtime_helpers.run_candidate_collection_batch(settings_list, _save_observe_only_candidate_collection_predictions, logger)
async def _predict_strategy_result(settings: AutoTradeSettings, entry_open_time: int) -> dict | None:
    if is_factor_candidate_signal_key(settings.strategy_key):
        return await asyncio.to_thread(
            _predict_live_factor_candidate_signal,
            settings,
            entry_open_time,
        )
    if settings.strategy_key == ENSEMBLE_RANKER_STRATEGY_KEY:
        return await asyncio.to_thread(
            predict_ensemble_ranker_prediction,
            settings.symbol,
            settings.duration,
            entry_open_time=entry_open_time,
        )
    if settings.strategy_key == EVENT_FINAL_DECISION_STRATEGY_KEY:
        return await asyncio.to_thread(
            predict_event_final_decision,
            settings.symbol,
            settings.duration,
            entry_open_time=entry_open_time,
        )
    return await asyncio.to_thread(
        predict_rule_direction,
        settings.symbol,
        settings.duration,
        entry_open_time=entry_open_time,
        strategy_key=settings.strategy_key,
    )
def _predict_live_factor_candidate_signal(settings: AutoTradeSettings, entry_open_time: int) -> dict:
    predictions = predict_factor_candidate_signals(
        settings.symbol,
        settings.duration,
        entry_open_time=entry_open_time,
        entry_grace_ms=strategy_entry_grace_ms(settings.strategy_key),
    )
    for prediction in predictions:
        if _is_settings_signal_prediction(prediction, settings):
            return prediction
    raise ValueError(f"live factor candidate prediction not produced: {settings.strategy_key}")


def _is_settings_signal_prediction(prediction: dict[str, Any], settings: AutoTradeSettings) -> bool:
    return prediction.get("strategy_key") == settings.strategy_key or prediction.get("signal_key") == settings.strategy_key


async def _save_candidate_collection_predictions(settings: AutoTradeSettings, *, write_lock: asyncio.Lock) -> None:
    entry_open_time = current_rule_entry_open_time_for_duration(settings.duration)
    await collection_helpers.save_candidate_collection_predictions(
        _candidate_collection_context(settings, entry_open_time, write_lock)
    )
async def _save_observe_only_candidate_collection_predictions(settings: AutoTradeSettings, *, write_lock: asyncio.Lock) -> None:
    entry_open_time = current_rule_entry_open_time_for_duration(settings.duration)
    await collection_helpers.save_observe_only_candidate_collection_predictions(
        _candidate_collection_context(settings, entry_open_time, write_lock)
    )
async def _save_lstm_shadow_prediction(settings, entry_open_time, write_lock) -> None:
    await _save_one_model_family_shadow_prediction(
        "lstm",
        {"settings": settings, "entry_open_time": entry_open_time, "write_lock": write_lock},
    )
async def _save_one_model_family_shadow_prediction(family, context) -> None:
    collection_context = _candidate_collection_context(
        context["settings"],
        context["entry_open_time"],
        context["write_lock"],
    )
    await collection_helpers.save_one_model_family_shadow_prediction(
        collection_helpers.ModelFamilyShadowContext(family, collection_context)
    )
def _candidate_collection_context(
    settings: AutoTradeSettings,
    entry_open_time: int,
    write_lock: asyncio.Lock,
) -> collection_helpers.CandidateCollectionContext:
    return collection_helpers.CandidateCollectionContext(settings, entry_open_time, write_lock, _collection_deps())
def _collection_deps() -> dict[str, Any]:
    return {"predict_eligible_factor_combo_rows": predict_eligible_factor_combo_rows, "predict_factor_candidate_signals": predict_factor_candidate_signals, "create_batch_combo_simulation_trade": create_batch_combo_simulation_trade, "save_prediction": _save_prediction, "lstm_model_status": lstm_model_status, "model_family_status": model_family_status, "predict_lstm_shadow_prediction": predict_lstm_shadow_prediction, "predict_model_family_shadow_prediction": predict_model_family_shadow_prediction, "prediction_cycle_context": PredictionCycleContext(), "log_model_family_shadow_skip": _log_model_family_shadow_skip}
async def _save_prediction(result: dict, write_lock: asyncio.Lock, *, allow_existing: bool = False) -> bool:
    async with write_lock:
        return await asyncio.to_thread(save_prediction, result, allow_existing=allow_existing)
def _prediction_targets() -> list[AutoTradeSettings]:
    return target_helpers.prediction_targets(list_auto_trade_settings(), get_default_settings=get_auto_trade_settings, readiness=strategy_prediction_readiness, supports_duration=strategy_supports_duration, logger=logger)
def _enabled_prediction_targets(settings: list[AutoTradeSettings]) -> list[AutoTradeSettings]:
    return target_helpers.enabled_prediction_targets(settings, strategy_supports_duration)
def _due_prediction_targets(targets: list[AutoTradeSettings]) -> list[AutoTradeSettings]:
    return [settings for settings in targets if _should_predict_entry(settings)]
def _ready_due_prediction_targets(targets: list[AutoTradeSettings]) -> target_helpers.ReadyPredictionTargetResult:
    return target_helpers.ready_due_prediction_target_result(
        targets,
        due_targets=_due_prediction_targets,
        readiness=strategy_prediction_readiness,
        logger=logger,
        attempt_recovery=False,
    )
def _candidate_collection_targets(targets: list[AutoTradeSettings]) -> list[AutoTradeSettings]:
    return target_helpers.candidate_collection_targets(targets, current_bucket=current_rule_entry_open_time_for_duration, collection_settings=_collection_settings, collection_due=_candidate_collection_due, supports_duration=strategy_supports_duration)
def _unique_collection_settings(targets: list[AutoTradeSettings]) -> list[AutoTradeSettings]:
    return target_helpers.unique_collection_settings(targets, _collection_settings, strategy_supports_duration)
def _collection_settings(settings: AutoTradeSettings) -> AutoTradeSettings:
    return target_helpers.collection_settings(settings)
def _candidate_collection_due(settings: AutoTradeSettings, bucket: int) -> bool:
    return (
        _factor_candidate_signal_due(settings, bucket)
        or _factor_combo_shadow_due(settings, bucket)
        or _ready_any_model_family_shadow_due(settings, bucket)
    )
def _should_predict_entry(settings: AutoTradeSettings) -> bool:
    if is_factor_candidate_signal_key(settings.strategy_key):
        return _live_factor_candidate_signal_due(settings)
    bucket = current_rule_entry_open_time_for_duration(settings.duration)
    if settings.strategy_key == EVENT_FINAL_DECISION_STRATEGY_KEY:
        return not event_final_decision_exists(settings.symbol, settings.duration, bucket)
    return target_helpers.should_predict_entry(settings, current_bucket=current_rule_entry_open_time_for_duration, prediction_exists=prediction_exists, ready_any_shadow_due=_ready_any_model_family_shadow_due, ready_family_strategy_due=_ready_model_family_strategy_due)
def _factor_combo_shadow_due(settings: AutoTradeSettings, bucket: int) -> bool:
    return target_helpers.factor_combo_shadow_due(
        settings,
        bucket,
        eligible_rows=eligible_factor_combo_rows,
        prediction_exists=prediction_exists,
        simulation_key=simulation_strategy_key_for_factor_name,
    )
def _factor_candidate_signal_due(settings: AutoTradeSettings, bucket: int) -> bool:
    return target_helpers.factor_candidate_signal_due(settings, bucket, signal_keys=factor_candidate_signal_keys, prediction_exists=prediction_exists)
def _ready_any_model_family_shadow_due(settings: AutoTradeSettings, bucket: int) -> bool:
    return any(_ready_model_family_shadow_due(family, settings, bucket, role="sidecar") for family in MODEL_FAMILIES)
def _ready_model_family_shadow_due(family: str, settings: AutoTradeSettings, bucket: int, *, role: str) -> bool:
    if family == "lstm":
        status = lstm_model_status(settings.symbol, settings.duration)
    else:
        status = model_family_status(family, settings.symbol, settings.duration)
    if status.get("shadowPredictionReady"):
        if family == "lstm":
            return bool(missing_lstm_shadow_entry_times(settings.symbol, settings.duration, bucket))
        return bool(missing_model_family_shadow_entry_times(family, settings.symbol, settings.duration, bucket))
    _log_model_family_shadow_skip(settings, family, status, role=role)
    return False
def _ready_model_family_strategy_due(settings: AutoTradeSettings, bucket: int) -> bool:
    return target_helpers.ready_model_family_strategy_due(
        settings,
        bucket,
        lambda family, item, entry: _ready_model_family_shadow_due(family, item, entry, role="primary"),
    )
def _live_factor_candidate_signal_due(settings: AutoTradeSettings) -> bool:
    if not settings.live_trading_enabled:
        return False
    bucket = current_rule_entry_open_time_for_duration(settings.duration)
    return not prediction_exists(
        strategy_key=settings.strategy_key,
        symbol=settings.symbol,
        duration=settings.duration,
        open_time=bucket,
    )


def _ready_lstm_shadow_due(settings: AutoTradeSettings, bucket: int) -> bool:
    return _ready_model_family_shadow_due("lstm", settings, bucket, role="sidecar")
def _ready_lstm_strategy_due(settings: AutoTradeSettings, bucket: int) -> bool:
    return _ready_model_family_shadow_due("lstm", settings, bucket, role="primary")
async def _backfill_lstm_shadow_predictions(settings_list: list[AutoTradeSettings]) -> None:
    await shadow_helpers.backfill_shadow_predictions(settings_list, _shadow_deps())
def _ready_model_family_shadow_backfill_targets(settings_list: list[AutoTradeSettings]) -> list[tuple[str, str, str]]:
    return shadow_helpers.ready_shadow_backfill_targets(settings_list, _shadow_deps())
def _unique_model_family_shadow_targets(settings_list: list[AutoTradeSettings]) -> list[tuple[str, str, str]]:
    return target_helpers.unique_model_family_shadow_targets(settings_list)
def _log_model_family_shadow_skip(settings: AutoTradeSettings, family: str, status: dict, *, role: str) -> None:
    shadow_helpers.log_model_family_shadow_skip(
        logger,
        settings=settings,
        family=family,
        status=status,
        role=role,
    )
def _shadow_deps() -> dict[str, Any]:
    return {"ready_targets": _ready_model_family_shadow_backfill_targets, "unique_targets": _unique_model_family_shadow_targets, "backfill": backfill_model_family_shadow_predictions, "current_entry": current_rule_entry_open_time_for_duration, "lstm_status": lstm_model_status, "family_status": model_family_status, "logger": logger}
def lstm_model_status(symbol: str, duration: str) -> dict:
    return model_family_status("lstm", symbol, duration)
def predict_lstm_shadow_prediction(symbol: str, duration: str, *, entry_open_time: int | None = None) -> dict:
    return predict_model_family_shadow_prediction("lstm", symbol, duration, entry_open_time=entry_open_time)
def missing_lstm_shadow_entry_times(symbol: str, duration: str, current_entry_open_time: int) -> tuple[int, ...]:
    return missing_model_family_shadow_entry_times("lstm", symbol, duration, current_entry_open_time)
def _next_predict_wait(targets: list[AutoTradeSettings], poll_seconds: int) -> float:
    if not targets:
        return float(poll_seconds)
    now_ms = utc_now_ms()
    min_wait = float("inf")
    for settings in targets:
        bucket = current_rule_entry_open_time_for_duration(settings.duration, now_ms)
        if _should_predict_entry(settings) or _candidate_collection_due(_collection_settings(settings), bucket):
            min_wait = min(min_wait, float(poll_seconds))
        else:
            min_wait = min(min_wait, seconds_until_next_rule_entry_for_duration(settings.duration, now_ms))
    return float(min_wait) if min_wait != float("inf") else float(poll_seconds)
async def _broadcast(result: dict) -> None:
    await runtime_helpers.broadcast(result, _SUBSCRIBERS, DEFAULT_STRATEGY_KEY)
