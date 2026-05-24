from __future__ import annotations

import asyncio
import logging
import os

from app.services.auto_trade_service import get_auto_trade_settings, list_auto_trade_settings
from app.services.auto_trade_types import AutoTradeSettings
from app.services.factor_combo_simulation_keys import (
    simulation_strategy_key_for_factor_name,
)
from app.services.factor_combo_batch_predictions import (
    eligible_factor_combo_rows,
    predict_eligible_factor_combo_rows,
)
from app.services.factor_combo_batch_simulation_service import create_batch_combo_simulation_trade
from app.services.factor_candidate_signal_service import (
    factor_candidate_signal_keys,
    predict_factor_candidate_signals,
)
from app.services.forward_validation_service import settle_due_predictions
from app.services.ensemble_judge_constants import ENSEMBLE_RANKER_STRATEGY_KEY
from app.services.ensemble_judge_service import refresh_ensemble_judge
from app.services.ensemble_ranker_prediction_service import predict_ensemble_ranker_prediction
from app.services.kline_timing import (
    MS_PER_MINUTE,
    current_rule_entry_open_time_for_duration,
    seconds_until_next_rule_entry_for_duration,
    utc_now_ms,
)
from app.services.kline_prediction_refresh import refresh_prediction_klines
from app.services.model_family_config import (
    MODEL_FAMILIES,
    is_model_family_shadow_strategy,
    parse_model_family_strategy,
)
from app.services.model_family_shadow_backfill_service import (
    backfill_model_family_shadow_predictions,
    missing_model_family_shadow_entry_times,
)
from app.services.prediction_cache_service import (
    prediction_exists,
    prediction_response,
    save_prediction,
)
from app.services.rule_config import DURATION_TO_MINUTES
from app.services.rule_signal_service import predict_rule_direction
from app.services.strategy_registry import (
    DEFAULT_STRATEGY_KEY,
    FACTOR_COMBO_STRATEGY_KEY,
    strategy_entry_grace_ms,
    strategy_supports_duration,
)
from app.services.strategy_prediction_readiness import strategy_prediction_readiness
from app.services.model_family_prediction_service import (
    predict_model_family_shadow_prediction,
)
from app.services.model_family_status_service import (
    model_family_status,
)

logger = logging.getLogger("uvicorn.error")
_SUBSCRIBERS: dict[tuple[str, str, str], set] = {}

DEFAULT_PREDICT_SECONDS = 1
DEFAULT_DURATION = "10m"


def _predict_initial_delay_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("PREDICT_INITIAL_DELAY_SECONDS", "8")))
    except ValueError:
        return 8.0


async def auto_predict_loop(stop_event: asyncio.Event, poll_seconds: int = DEFAULT_PREDICT_SECONDS) -> None:
    initial = _predict_initial_delay_seconds()
    logger.info("predict loop: initial_delay=%ss poll=%ss", initial, poll_seconds)
    if initial > 0:
        await _sleep_for(stop_event, initial)
    logger.info("predict loop: running during each enabled strategy's kline entry window")
    while not stop_event.is_set():
        try:
            targets = await asyncio.to_thread(_prediction_targets)
            await _predict_due_entries(targets)
            wait_seconds = _next_predict_wait(targets, poll_seconds)
        except Exception:
            logger.exception("auto prediction failed")
            wait_seconds = float(poll_seconds)
        await _sleep_for(stop_event, wait_seconds)


async def _predict_due_entries(targets: list[AutoTradeSettings]) -> None:
    due_targets = await asyncio.to_thread(_ready_due_prediction_targets, targets)
    collection_targets = await asyncio.to_thread(_candidate_collection_targets, targets)
    active_targets = _merged_targets(due_targets, collection_targets)
    if not active_targets:
        return
    await _prepare_prediction_inputs(active_targets)
    if collection_targets:
        await _run_candidate_collection_batch(collection_targets)
    prediction_error = None
    if due_targets:
        try:
            await _run_prediction_batch(due_targets)
        except Exception as exc:
            logger.exception("primary prediction batch failed; candidate collection will still run")
            prediction_error = exc
    await _backfill_lstm_shadow_predictions(active_targets)
    if prediction_error is not None:
        raise prediction_error


async def _prepare_prediction_inputs(settings_list: list[AutoTradeSettings]) -> None:
    # 每次预测前拉取最新 1m 和策略周期 K；结算用 1m，因子预测用策略周期 K。
    by_symbol: dict[str, list[int]] = {}
    by_symbol_duration: dict[tuple[str, str], list[int]] = {}
    for settings in settings_list:
        sym = settings.symbol.upper()
        by_symbol.setdefault(sym, []).append(current_rule_entry_open_time_for_duration(settings.duration))
        key = (sym, settings.duration)
        by_symbol_duration.setdefault(key, []).append(current_rule_entry_open_time_for_duration(settings.duration))
    if by_symbol:
        await asyncio.gather(
            *(
                asyncio.to_thread(_refresh_1m_prediction_input, symbol, max(buckets))
                for symbol, buckets in by_symbol.items()
                if buckets
            )
        )
    if by_symbol_duration:
        await asyncio.gather(
            *(
                asyncio.to_thread(_refresh_duration_prediction_input, symbol, duration, max(buckets))
                for (symbol, duration), buckets in by_symbol_duration.items()
                if buckets
            )
        )
    await asyncio.to_thread(_run_prediction_db_side_effects, settings_list)


async def _run_prediction_batch(settings_list: list[AutoTradeSettings]) -> None:
    write_lock = asyncio.Lock()
    results = await asyncio.gather(
        *(_run_prediction(settings, write_lock=write_lock) for settings in settings_list),
        return_exceptions=True,
    )
    _raise_prediction_failures(settings_list, results)


async def _run_prediction(
    settings: AutoTradeSettings,
    *,
    write_lock: asyncio.Lock,
) -> None:
    entry_open_time = current_rule_entry_open_time_for_duration(settings.duration)
    result = await _predict_strategy_result(settings, entry_open_time)
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
    write_lock = asyncio.Lock()
    results = await asyncio.gather(
        *(_save_candidate_collection_predictions(settings, write_lock=write_lock) for settings in settings_list),
        return_exceptions=True,
    )
    failures = [
        (settings, result)
        for settings, result in zip(settings_list, results)
        if isinstance(result, Exception)
    ]
    for settings, exc in failures:
        logger.error(
            "candidate collection failed strategy=%s symbol=%s duration=%s",
            settings.strategy_key,
            settings.symbol,
            settings.duration,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    if failures:
        failed = ", ".join(f"{item.symbol}:{item.duration}" for item, _exc in failures)
        raise RuntimeError(f"candidate collection failed for: {failed}") from failures[0][1]


async def _predict_strategy_result(settings: AutoTradeSettings, entry_open_time: int) -> dict:
    if settings.strategy_key == ENSEMBLE_RANKER_STRATEGY_KEY:
        return await asyncio.to_thread(
            predict_ensemble_ranker_prediction,
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


async def _save_candidate_collection_predictions(
    settings: AutoTradeSettings,
    *,
    write_lock: asyncio.Lock,
) -> None:
    entry_open_time = current_rule_entry_open_time_for_duration(settings.duration)
    await _save_factor_candidate_signals(settings, entry_open_time, write_lock)
    await _save_factor_combo_shadow_predictions(settings, entry_open_time, write_lock)


async def _save_factor_combo_shadow_predictions(
    settings: AutoTradeSettings,
    entry_open_time: int,
    write_lock: asyncio.Lock,
) -> None:
    if settings.strategy_key != FACTOR_COMBO_STRATEGY_KEY:
        return
    results = await asyncio.to_thread(
        predict_eligible_factor_combo_rows,
        settings.symbol,
        settings.duration,
        entry_open_time=entry_open_time,
        entry_grace_ms=strategy_entry_grace_ms(settings.strategy_key),
    )
    for result in results:
        saved = await _save_prediction(result, write_lock)
        if saved:
            await asyncio.to_thread(create_batch_combo_simulation_trade, settings, result)


async def _save_model_family_shadow_predictions(
    settings: AutoTradeSettings,
    entry_open_time: int,
    write_lock: asyncio.Lock,
) -> None:
    if settings.strategy_key != FACTOR_COMBO_STRATEGY_KEY:
        return
    for family in MODEL_FAMILIES:
        if family == "lstm":
            status = await asyncio.to_thread(lstm_model_status, settings.symbol, settings.duration)
        else:
            status = await asyncio.to_thread(model_family_status, family, settings.symbol, settings.duration)
        if not status.get("shadowPredictionReady"):
            _log_model_family_shadow_skip(settings, family, status, role="sidecar")
            continue
        result = await asyncio.to_thread(
            predict_lstm_shadow_prediction if family == "lstm" else predict_model_family_shadow_prediction,
            *((settings.symbol, settings.duration) if family == "lstm" else (family, settings.symbol, settings.duration)),
            entry_open_time=entry_open_time,
        )
        await _save_prediction(result, write_lock)


async def _save_factor_candidate_signals(
    settings: AutoTradeSettings,
    entry_open_time: int,
    write_lock: asyncio.Lock,
) -> None:
    if not strategy_supports_duration(settings.strategy_key, settings.duration):
        return
    results = await asyncio.to_thread(
        predict_factor_candidate_signals,
        settings.symbol,
        settings.duration,
        entry_open_time=entry_open_time,
        entry_grace_ms=strategy_entry_grace_ms(settings.strategy_key),
    )
    for result in results:
        await _save_prediction(result, write_lock)


async def _save_lstm_shadow_prediction(settings, entry_open_time, write_lock) -> None:
    await _save_one_model_family_shadow_prediction("lstm", settings, entry_open_time, write_lock)


async def _save_one_model_family_shadow_prediction(family, settings, entry_open_time, write_lock) -> None:
    status_loader = lstm_model_status if family == "lstm" else model_family_status
    status_args = (settings.symbol, settings.duration) if family == "lstm" else (family, settings.symbol, settings.duration)
    status = await asyncio.to_thread(status_loader, *status_args)
    if not status.get("shadowPredictionReady"):
        _log_model_family_shadow_skip(settings, family, status, role="sidecar")
        return
    result = await asyncio.to_thread(
        predict_lstm_shadow_prediction if family == "lstm" else predict_model_family_shadow_prediction,
        *((settings.symbol, settings.duration) if family == "lstm" else (family, settings.symbol, settings.duration)),
        entry_open_time=entry_open_time,
    )
    await _save_prediction(result, write_lock)


async def _save_prediction(
    result: dict,
    write_lock: asyncio.Lock,
    *,
    allow_existing: bool = False,
) -> bool:
    async with write_lock:
        return await asyncio.to_thread(save_prediction, result, allow_existing=allow_existing)


def _raise_prediction_failures(settings_list: list[AutoTradeSettings], results: list[object]) -> None:
    failures = _prediction_failures(settings_list, results)
    for settings, exc in failures:
        logger.error(
            "predict strategy failed strategy=%s",
            settings.strategy_key,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    if failures:
        failed_keys = ", ".join(settings.strategy_key for settings, _exc in failures)
        raise RuntimeError(f"auto prediction failed for strategies: {failed_keys}") from failures[0][1]


def _prediction_failures(
    settings_list: list[AutoTradeSettings],
    results: list[object],
) -> list[tuple[AutoTradeSettings, Exception]]:
    return [
        (settings, result)
        for settings, result in zip(settings_list, results)
        if isinstance(result, Exception)
    ]


def _prediction_targets() -> list[AutoTradeSettings]:
    settings = list_auto_trade_settings()
    enabled = _enabled_prediction_targets(settings)
    if enabled:
        return enabled
    if any(item.enabled for item in settings):
        return []
    return _default_prediction_targets()


def _default_prediction_targets() -> list[AutoTradeSettings]:
    default = get_auto_trade_settings(DEFAULT_STRATEGY_KEY)
    readiness = strategy_prediction_readiness(
        default.strategy_key,
        default.symbol,
        default.duration,
        attempt_recovery=True,
    )
    if readiness.ready:
        return [default]
    logger.warning(
        "default predict target skipped strategy=%s symbol=%s duration=%s reason=%s recoveryAttempted=%s recoveryStatus=%s diagnostics=%s",
        default.strategy_key,
        default.symbol,
        default.duration,
        readiness.reason,
        readiness.recovery_attempted,
        readiness.recovery_status,
        readiness.diagnostics,
    )
    return []


def _enabled_prediction_targets(settings: list[AutoTradeSettings]) -> list[AutoTradeSettings]:
    return [
        item
        for item in settings
        if item.enabled and strategy_supports_duration(item.strategy_key, item.duration)
    ]


def _due_prediction_targets(targets: list[AutoTradeSettings]) -> list[AutoTradeSettings]:
    return [settings for settings in targets if _should_predict_entry(settings)]


def _ready_due_prediction_targets(targets: list[AutoTradeSettings]) -> list[AutoTradeSettings]:
    ready = []
    for settings in _due_prediction_targets(targets):
        readiness = strategy_prediction_readiness(
            settings.strategy_key,
            settings.symbol,
            settings.duration,
            attempt_recovery=False,
        )
        if readiness.ready:
            ready.append(settings)
            continue
        logger.warning(
            "predict due target skipped strategy=%s symbol=%s duration=%s reason=%s recoveryAttempted=%s recoveryStatus=%s diagnostics=%s",
            settings.strategy_key,
            settings.symbol,
            settings.duration,
            readiness.reason,
            readiness.recovery_attempted,
            readiness.recovery_status,
            readiness.diagnostics,
        )
    return ready


def _candidate_collection_targets(targets: list[AutoTradeSettings]) -> list[AutoTradeSettings]:
    collection = []
    for settings in _unique_collection_settings(targets):
        bucket = current_rule_entry_open_time_for_duration(settings.duration)
        if _candidate_collection_due(settings, bucket):
            collection.append(settings)
    return collection


def _unique_collection_settings(targets: list[AutoTradeSettings]) -> list[AutoTradeSettings]:
    selected: dict[tuple[str, str], AutoTradeSettings] = {}
    for settings in targets:
        if not strategy_supports_duration(settings.strategy_key, settings.duration):
            continue
        key = (settings.symbol.upper(), settings.duration)
        selected[key] = _collection_settings(settings)
    return list(selected.values())


def _collection_settings(settings: AutoTradeSettings) -> AutoTradeSettings:
    return AutoTradeSettings(
        strategy_key=FACTOR_COMBO_STRATEGY_KEY,
        enabled=True,
        symbol=settings.symbol.upper(),
        duration=settings.duration,
        duration_minutes=settings.duration_minutes,
        qty=settings.qty,
        live_trading_enabled=False,
    )


def _candidate_collection_due(settings: AutoTradeSettings, bucket: int) -> bool:
    return _factor_candidate_signal_due(settings, bucket) or _factor_combo_shadow_due(settings, bucket)


def _should_predict_entry(settings: AutoTradeSettings) -> bool:
    bucket = current_rule_entry_open_time_for_duration(settings.duration)
    if _current_bucket_prediction_exists(settings, bucket):
        return _ready_any_model_family_shadow_due(settings, bucket)
    if is_model_family_shadow_strategy(settings.strategy_key):
        return _ready_model_family_strategy_due(settings, bucket)
    return True


def _current_bucket_prediction_exists(settings: AutoTradeSettings, bucket: int) -> bool:
    return prediction_exists(
        strategy_key=settings.strategy_key,
        symbol=settings.symbol,
        duration=settings.duration,
        open_time=bucket,
    )


def _factor_combo_shadow_due(settings: AutoTradeSettings, bucket: int) -> bool:
    for row in eligible_factor_combo_rows(settings.symbol, settings.duration):
        if not prediction_exists(
            strategy_key=simulation_strategy_key_for_factor_name(str(row["factorName"])),
            symbol=settings.symbol,
            duration=settings.duration,
            open_time=bucket,
        ):
            return True
    return False


def _factor_candidate_signal_due(settings: AutoTradeSettings, bucket: int) -> bool:
    for signal_key in factor_candidate_signal_keys(settings.symbol, settings.duration):
        if not prediction_exists(
            signal_key=signal_key,
            symbol=settings.symbol,
            duration=settings.duration,
            open_time=bucket,
        ):
            return True
    return False


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
    parsed = parse_model_family_strategy(settings.strategy_key)
    if parsed is None:
        return False
    family, _duration = parsed
    return _ready_model_family_shadow_due(family, settings, bucket, role="primary")


def _ready_lstm_shadow_due(settings: AutoTradeSettings, bucket: int) -> bool:
    return _ready_model_family_shadow_due("lstm", settings, bucket, role="sidecar")


def _ready_lstm_strategy_due(settings: AutoTradeSettings, bucket: int) -> bool:
    return _ready_model_family_shadow_due("lstm", settings, bucket, role="primary")


async def _backfill_lstm_shadow_predictions(settings_list: list[AutoTradeSettings]) -> None:
    targets = _ready_model_family_shadow_backfill_targets(settings_list)
    if not targets:
        return
    summaries = await asyncio.gather(
        *(
            asyncio.to_thread(
                backfill_model_family_shadow_predictions,
                family,
                symbol,
                duration,
                current_rule_entry_open_time_for_duration(duration),
            )
            for family, symbol, duration in targets
        ),
        return_exceptions=True,
    )
    for summary in summaries:
        if isinstance(summary, Exception):
            logger.exception("model family shadow backfill failed")
            continue
        if summary["savedCount"]:
            logger.info("predict: model family shadow backfill summary=%s", summary)


def _ready_model_family_shadow_backfill_targets(settings_list: list[AutoTradeSettings]) -> list[tuple[str, str, str]]:
    ready = []
    for family, symbol, duration in _unique_model_family_shadow_targets(settings_list):
        status = lstm_model_status(symbol, duration) if family == "lstm" else model_family_status(family, symbol, duration)
        if status.get("shadowPredictionReady"):
            ready.append((family, symbol, duration))
    return ready


def _unique_lstm_shadow_targets(settings_list: list[AutoTradeSettings]) -> list[tuple[str, str]]:
    return [(symbol, duration) for _family, symbol, duration in _unique_model_family_shadow_targets(settings_list) if _family == "lstm"]


def _unique_model_family_shadow_targets(settings_list: list[AutoTradeSettings]) -> list[tuple[str, str, str]]:
    targets = set()
    for settings in settings_list:
        if settings.strategy_key == FACTOR_COMBO_STRATEGY_KEY:
            for family in MODEL_FAMILIES:
                targets.add((family, settings.symbol.upper(), settings.duration))
            continue
        parsed = parse_model_family_strategy(settings.strategy_key)
        if parsed is not None:
            family, _duration = parsed
            targets.add((family, settings.symbol.upper(), settings.duration))
    return sorted(targets)


def _log_model_family_shadow_skip(
    settings: AutoTradeSettings,
    family: str,
    status: dict,
    *,
    role: str,
) -> None:
    logger.info(
        "predict: %s shadow skipped role=%s for %s %s reason=%s dependency=%s status=%s artifacts=%s combo=%s",
        family,
        role,
        settings.symbol,
        settings.duration,
        status.get("shadowPredictionBlockedReason"),
        status.get("dependencyStatus"),
        status.get("status"),
        status.get("artifactsReady"),
        status.get("comboSnapshotReason"),
    )


def _log_lstm_shadow_skip(settings: AutoTradeSettings, status: dict, *, role: str) -> None:
    _log_model_family_shadow_skip(settings, "lstm", status, role=role)


def lstm_model_status(symbol: str, duration: str) -> dict:
    return model_family_status("lstm", symbol, duration)


def predict_lstm_shadow_prediction(symbol: str, duration: str, *, entry_open_time: int | None = None) -> dict:
    return predict_model_family_shadow_prediction("lstm", symbol, duration, entry_open_time=entry_open_time)


def missing_lstm_shadow_entry_times(symbol: str, duration: str, current_entry_open_time: int) -> tuple[int, ...]:
    return missing_model_family_shadow_entry_times("lstm", symbol, duration, current_entry_open_time)


def _run_prediction_db_side_effects(settings_list: list[AutoTradeSettings]) -> None:
    for symbol, duration in _unique_symbol_durations(settings_list):
        settle_due_predictions(symbol, duration)
        refresh_ensemble_judge(symbol, duration)


def _unique_symbol_durations(settings_list: list[AutoTradeSettings]) -> list[tuple[str, str]]:
    return sorted({(settings.symbol.upper(), settings.duration) for settings in settings_list})


def _merged_targets(
    first: list[AutoTradeSettings],
    second: list[AutoTradeSettings],
) -> list[AutoTradeSettings]:
    merged = {}
    for settings in [*first, *second]:
        merged[(settings.strategy_key, settings.symbol.upper(), settings.duration)] = settings
    return list(merged.values())


def _refresh_1m_prediction_input(symbol: str, entry_open_time: int) -> None:
    refresh_prediction_klines(symbol, "1m", entry_open_time - MS_PER_MINUTE)


def _refresh_duration_prediction_input(symbol: str, duration: str, entry_open_time: int) -> None:
    refresh_prediction_klines(symbol, duration, entry_open_time - _duration_ms(duration))


def _duration_ms(duration: str) -> int:
    return DURATION_TO_MINUTES[duration] * MS_PER_MINUTE


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


async def _sleep_for(stop_event: asyncio.Event, wait_seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=wait_seconds)
    except TimeoutError:
        return


async def _broadcast(result: dict) -> None:
    key = (
        result["symbol"].upper(),
        result["duration"],
        result.get("strategyKey") or DEFAULT_STRATEGY_KEY,
    )
    websockets = _SUBSCRIBERS.get(key, set())
    dead = set()
    for ws in websockets:
        try:
            await ws.send_json(result)
        except Exception:
            dead.add(ws)
    if dead:
        websockets -= dead

def subscribe(
    ws,
    symbol: str,
    duration: str,
    strategy_key: str = DEFAULT_STRATEGY_KEY,
) -> None:
    _SUBSCRIBERS.setdefault((symbol.upper(), duration, strategy_key), set()).add(ws)


def unsubscribe(
    ws,
    symbol: str,
    duration: str,
    strategy_key: str = DEFAULT_STRATEGY_KEY,
) -> None:
    s = _SUBSCRIBERS.get((symbol.upper(), duration, strategy_key))
    if s:
        s.discard(ws)
