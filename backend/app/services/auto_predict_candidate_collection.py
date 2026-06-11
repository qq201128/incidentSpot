from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.services.auto_trade_types import AutoTradeSettings
from app.services.auto_predict_model_shadow_collection import (
    ModelFamilyShadowContext,
    ModelFamilyShadowOutcome,
    save_model_family_shadow_trade_for_outcome,
    save_model_family_shadow_trade,
    save_one_model_family_shadow_prediction,
)
from app.services.model_family_config import MODEL_FAMILIES
from app.services.paper_live_candidate_service import log_prediction_failure
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY, strategy_entry_grace_ms, strategy_supports_duration

MODEL_FAMILY_SHADOW_CONCURRENCY = 3


@dataclass(frozen=True)
class CandidateCollectionContext:
    settings: AutoTradeSettings
    entry_open_time: int
    write_lock: asyncio.Lock
    deps: dict[str, Any]


@dataclass(frozen=True)
class PredictionFailureContext:
    settings: AutoTradeSettings
    entry_open_time: int
    exc: Exception
    stage: str


@dataclass(frozen=True)
class ModelFamilyShadowFailure:
    family: str
    exception: Exception


class ModelFamilyShadowBatchError(RuntimeError):
    def __init__(self, context: CandidateCollectionContext, failures: list[ModelFamilyShadowFailure]) -> None:
        self.failures = tuple(failures)
        self.details = [_model_family_shadow_failure_detail(context, failure) for failure in failures]
        families = ", ".join(failure.family for failure in failures)
        settings = context.settings
        super().__init__(f"model family shadow prediction failed for {settings.symbol}:{settings.duration}: {families}")


async def save_candidate_collection_predictions(context: CandidateCollectionContext) -> None:
    await save_factor_candidate_signals(context)
    await save_factor_combo_shadow_predictions(context)
    await save_model_family_shadow_predictions(context)


async def save_final_decision_required_predictions(
    context: CandidateCollectionContext,
) -> list[ModelFamilyShadowOutcome]:
    return await _save_model_family_shadow_predictions(context, create_simulation_trades=False)


async def save_observe_only_candidate_collection_predictions(context: CandidateCollectionContext) -> None:
    await save_factor_candidate_signals(context)
    await save_factor_combo_shadow_predictions(context)


async def save_model_family_shadow_predictions(
    context: CandidateCollectionContext,
) -> list[ModelFamilyShadowOutcome]:
    return await _save_model_family_shadow_predictions(context, create_simulation_trades=True)


async def _save_model_family_shadow_predictions(
    context: CandidateCollectionContext,
    *,
    create_simulation_trades: bool,
) -> list[ModelFamilyShadowOutcome]:
    if context.settings.strategy_key != FACTOR_COMBO_STRATEGY_KEY:
        return []
    outcomes, failures = await _run_model_family_shadow_predictions(context, MODEL_FAMILIES, create_simulation_trades)
    if failures:
        raise ModelFamilyShadowBatchError(context, failures) from failures[0].exception
    return outcomes


async def _run_model_family_shadow_predictions(
    context: CandidateCollectionContext,
    families: tuple[str, ...],
    create_simulation_trades: bool,
) -> tuple[list[ModelFamilyShadowOutcome], list[ModelFamilyShadowFailure]]:
    semaphore = asyncio.Semaphore(MODEL_FAMILY_SHADOW_CONCURRENCY)
    results = await asyncio.gather(
        *(
            _run_model_family_shadow_prediction(semaphore, context, family, create_simulation_trades)
            for family in families
        )
    )
    failures = [item for item in results if isinstance(item, ModelFamilyShadowFailure)]
    outcomes = [item for item in results if isinstance(item, ModelFamilyShadowOutcome)]
    return outcomes, failures


async def _run_model_family_shadow_prediction(
    semaphore: asyncio.Semaphore,
    context: CandidateCollectionContext,
    family: str,
    create_simulation_trade: bool,
) -> ModelFamilyShadowOutcome | ModelFamilyShadowFailure:
    async with semaphore:
        try:
            return await save_one_model_family_shadow_prediction(
                ModelFamilyShadowContext(family, context),
                create_simulation_trade=create_simulation_trade,
            )
        except Exception as exc:
            return ModelFamilyShadowFailure(family, exc)


async def save_model_family_shadow_simulation_trades(
    context: CandidateCollectionContext,
    outcomes: list[ModelFamilyShadowOutcome],
) -> None:
    for outcome in outcomes:
        await save_model_family_shadow_trade_for_outcome(
            context.settings,
            outcome,
            context.deps["create_batch_combo_simulation_trade"],
        )


async def save_factor_combo_shadow_predictions(context: CandidateCollectionContext) -> None:
    settings = context.settings
    if settings.strategy_key != FACTOR_COMBO_STRATEGY_KEY:
        return
    try:
        results = await asyncio.to_thread(
            context.deps["predict_eligible_factor_combo_rows"],
            settings.symbol,
            settings.duration,
            entry_open_time=context.entry_open_time,
            entry_grace_ms=strategy_entry_grace_ms(settings.strategy_key),
        )
    except Exception as exc:
        _log_collection_failure(PredictionFailureContext(settings, context.entry_open_time, exc, "factor_combo_shadow_prediction"))
        raise
    for result in results:
        await _save_prediction_and_simulation_trade(context, result)


async def save_factor_candidate_signals(context: CandidateCollectionContext) -> None:
    settings = context.settings
    if not strategy_supports_duration(settings.strategy_key, settings.duration):
        return
    try:
        results = await asyncio.to_thread(
            context.deps["predict_factor_candidate_signals"],
            settings.symbol,
            settings.duration,
            entry_open_time=context.entry_open_time,
            entry_grace_ms=strategy_entry_grace_ms(settings.strategy_key),
        )
    except Exception as exc:
        _log_collection_failure(PredictionFailureContext(settings, context.entry_open_time, exc, "factor_candidate_signal_prediction"))
        raise
    for result in results:
        await _save_prediction_and_simulation_trade(context, result)


async def _save_prediction_and_simulation_trade(
    context: CandidateCollectionContext,
    result: dict[str, Any],
) -> None:
    saved = await context.deps["save_prediction"](result, context.write_lock)
    if saved:
        await asyncio.to_thread(context.deps["create_batch_combo_simulation_trade"], context.settings, result)


def _log_collection_failure(context: PredictionFailureContext) -> None:
    log_prediction_failure(
        candidate_key=context.settings.strategy_key,
        strategy_key=context.settings.strategy_key,
        symbol=context.settings.symbol,
        duration=context.settings.duration,
        stage=context.stage,
        reason=str(context.exc),
        details={"entryOpenTime": int(context.entry_open_time), "exceptionType": type(context.exc).__name__},
    )


def _model_family_shadow_failure_detail(
    context: CandidateCollectionContext,
    failure: ModelFamilyShadowFailure,
) -> dict[str, Any]:
    settings = context.settings
    return {
        "family": failure.family,
        "strategyKey": settings.strategy_key,
        "symbol": settings.symbol,
        "duration": settings.duration,
        "entryOpenTime": int(context.entry_open_time),
        "error": str(failure.exception),
        "exceptionType": type(failure.exception).__name__,
    }
