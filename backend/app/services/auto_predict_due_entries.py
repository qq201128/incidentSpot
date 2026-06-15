from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.services.auto_trade_types import AutoTradeSettings
from app.services.event_final_decision_service import EVENT_FINAL_DECISION_STRATEGY_KEY


@dataclass(frozen=True)
class PredictDueEntryDeps:
    ready_due_prediction_targets: Callable[[list[AutoTradeSettings]], Any]
    candidate_collection_targets: Callable[[list[AutoTradeSettings]], list[AutoTradeSettings]]
    prediction_cycle_targets: Callable[..., Any]
    record_prediction_progress: Callable[[Any, str], None]
    prediction_cycle_summary: Callable[[Any], dict[str, Any]]
    prepare_prediction_inputs: Callable[[list[AutoTradeSettings]], Awaitable[None]]
    run_prediction_batch: Callable[[list[AutoTradeSettings]], Awaitable[None]]
    run_candidate_collection_batch: Callable[[list[AutoTradeSettings]], Awaitable[None]]
    run_final_decision_required_candidate_batch: Callable[[list[AutoTradeSettings]], Awaitable[list[Any]]]
    run_observe_only_candidate_collection_batch: Callable[[list[AutoTradeSettings], list[Any]], Awaitable[None]]
    backfill_lstm_shadow_predictions: Callable[[list[AutoTradeSettings]], Awaitable[None]]
    logger: Any


async def predict_due_entries(targets: list[AutoTradeSettings], deps: PredictDueEntryDeps) -> dict[str, Any]:
    due_result = await asyncio.to_thread(deps.ready_due_prediction_targets, targets)
    due_targets = list(due_result.ready)
    skipped_targets = list(due_result.skipped)
    collection_targets = await asyncio.to_thread(deps.candidate_collection_targets, targets)
    cycle_targets = deps.prediction_cycle_targets(targets, due_targets, collection_targets, skipped_targets=skipped_targets)
    primary_due_targets, final_due_targets = _split_final_decision_targets(due_targets)
    live_due_targets = [target for target in primary_due_targets if target.live_trading_enabled]
    active_targets = list(cycle_targets.active)
    non_blocking_errors: list[Exception] = []
    deps.record_prediction_progress(cycle_targets, "inputs_preparing")
    if not active_targets:
        return deps.prediction_cycle_summary(cycle_targets)
    await deps.prepare_prediction_inputs(active_targets)
    prediction_error = await _run_primary_predictions(primary_due_targets, cycle_targets, deps)
    final_batches, collection_error = await _run_required_final_candidates(final_due_targets, collection_targets, cycle_targets, deps)
    if collection_error is not None:
        non_blocking_errors.append(collection_error)
    final_error = await _run_final_predictions([] if collection_error else final_due_targets, cycle_targets, deps)
    shadow_error = await _run_observe_only_work(collection_targets, final_batches, cycle_targets, active_targets, deps)
    if shadow_error is not None:
        non_blocking_errors.append(shadow_error)
    _raise_prediction_errors(prediction_error, final_error)
    if non_blocking_errors and not live_due_targets:
        raise non_blocking_errors[0]
    return _cycle_summary_with_errors(deps.prediction_cycle_summary(cycle_targets), non_blocking_errors)


async def _run_primary_predictions(settings_list: list[AutoTradeSettings], cycle_targets: Any, deps: PredictDueEntryDeps) -> Exception | None:
    deps.record_prediction_progress(cycle_targets, "primary_prediction_running")
    if not settings_list:
        deps.record_prediction_progress(cycle_targets, "primary_prediction_done")
        return None
    try:
        await deps.run_prediction_batch(settings_list)
    except Exception as exc:
        deps.logger.exception("primary prediction batch failed; candidate collection will still run")
        deps.record_prediction_progress(cycle_targets, "primary_prediction_done")
        return exc
    deps.record_prediction_progress(cycle_targets, "primary_prediction_done")
    return None


async def _run_required_final_candidates(
    final_due_targets: list[AutoTradeSettings],
    collection_targets: list[AutoTradeSettings],
    cycle_targets: Any,
    deps: PredictDueEntryDeps,
) -> tuple[list[Any], Exception | None]:
    if final_due_targets and collection_targets:
        deps.record_prediction_progress(cycle_targets, "final_decision_candidates_running")
        try:
            return await deps.run_final_decision_required_candidate_batch(collection_targets), None
        except Exception as exc:
            deps.logger.exception("final decision candidate batch failed")
            return [], exc
    if collection_targets:
        deps.record_prediction_progress(cycle_targets, "candidate_collection_running")
        try:
            await deps.run_candidate_collection_batch(collection_targets)
        except Exception as exc:
            deps.logger.exception("candidate collection batch failed")
            return [], exc
    return [], None


async def _run_final_predictions(settings_list: list[AutoTradeSettings], cycle_targets: Any, deps: PredictDueEntryDeps) -> Exception | None:
    if not settings_list:
        return None
    deps.record_prediction_progress(cycle_targets, "final_decision_running")
    try:
        await deps.run_prediction_batch(settings_list)
    except Exception as exc:
        deps.logger.exception("final decision prediction batch failed")
        return exc
    return None


async def _run_observe_only_work(
    collection_targets: list[AutoTradeSettings],
    final_batches: list[Any],
    cycle_targets: Any,
    active_targets: list[AutoTradeSettings],
    deps: PredictDueEntryDeps,
) -> Exception | None:
    if final_batches:
        deps.record_prediction_progress(cycle_targets, "observe_only_candidate_collection_running")
        try:
            await deps.run_observe_only_candidate_collection_batch(collection_targets, final_batches)
        except Exception as exc:
            deps.logger.exception("observe-only candidate collection failed")
            return exc
    deps.record_prediction_progress(cycle_targets, "shadow_backfill_running")
    try:
        await deps.backfill_lstm_shadow_predictions(active_targets)
    except Exception as exc:
        deps.logger.exception("shadow backfill failed")
        return exc
    return None


def _split_final_decision_targets(
    targets: list[AutoTradeSettings],
) -> tuple[list[AutoTradeSettings], list[AutoTradeSettings]]:
    primary = []
    final = []
    for settings in targets:
        if settings.strategy_key == EVENT_FINAL_DECISION_STRATEGY_KEY:
            final.append(settings)
        else:
            primary.append(settings)
    return primary, final


def _raise_prediction_errors(prediction_error: Exception | None, final_error: Exception | None) -> None:
    if prediction_error is not None:
        raise prediction_error
    if final_error is not None:
        raise final_error


def _cycle_summary_with_errors(summary: dict[str, Any], errors: list[Exception]) -> dict[str, Any]:
    result = dict(summary)
    if errors:
        result["nonBlockingErrorCount"] = len(errors)
        result["nonBlockingErrors"] = [_error_payload(error) for error in errors]
    return result


def _error_payload(error: Exception) -> dict[str, Any]:
    details = getattr(error, "details", None)
    payload: dict[str, Any] = {
        "error": str(error),
        "exceptionType": type(error).__name__,
    }
    if details is not None:
        payload["details"] = details
    return payload
