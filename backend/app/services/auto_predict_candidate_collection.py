from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from app.services.auto_trade_types import AutoTradeSettings
from app.services.model_family_config import MODEL_FAMILIES
from app.services.paper_live_candidate_service import log_prediction_failure
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY, strategy_entry_grace_ms, strategy_supports_duration


@dataclass(frozen=True)
class CandidateCollectionContext:
    settings: AutoTradeSettings
    entry_open_time: int
    write_lock: asyncio.Lock
    deps: dict[str, Any]


@dataclass(frozen=True)
class ModelFamilyShadowContext:
    family: str
    collection: CandidateCollectionContext


@dataclass(frozen=True)
class PredictionFailureContext:
    settings: AutoTradeSettings
    entry_open_time: int
    exc: Exception
    stage: str


@dataclass(frozen=True)
class ModelPredictionFailureContext:
    settings: AutoTradeSettings
    family: str
    entry_open_time: int
    exc: Exception
    status: dict[str, Any]


async def save_candidate_collection_predictions(context: CandidateCollectionContext) -> None:
    await save_factor_candidate_signals(context)
    await save_factor_combo_shadow_predictions(context)
    await save_model_family_shadow_predictions(context)


async def save_model_family_shadow_predictions(
    context: CandidateCollectionContext,
) -> None:
    if context.settings.strategy_key != FACTOR_COMBO_STRATEGY_KEY:
        return
    for family in MODEL_FAMILIES:
        await save_one_model_family_shadow_prediction(ModelFamilyShadowContext(family, context))


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


async def save_one_model_family_shadow_prediction(context: ModelFamilyShadowContext) -> None:
    family = context.family
    collection = context.collection
    deps = collection.deps
    settings = collection.settings
    status_loader = deps["lstm_model_status"] if family == "lstm" else deps["model_family_status"]
    status_args = (settings.symbol, settings.duration) if family == "lstm" else (family, settings.symbol, settings.duration)
    status = await asyncio.to_thread(status_loader, *status_args)
    if not status.get("shadowPredictionReady"):
        _log_model_skip((settings, family, status, collection.entry_open_time))
        deps["log_model_family_shadow_skip"](settings, family, status, role="sidecar")
        return
    predictor = deps["predict_lstm_shadow_prediction"] if family == "lstm" else deps["predict_model_family_shadow_prediction"]
    args = (settings.symbol, settings.duration) if family == "lstm" else (family, settings.symbol, settings.duration)
    try:
        result = await asyncio.to_thread(predictor, *args, entry_open_time=collection.entry_open_time)
    except Exception as exc:
        _log_model_prediction_failure(ModelPredictionFailureContext(settings, family, collection.entry_open_time, exc, status))
        raise
    saved = await deps["save_prediction"](result, collection.write_lock)
    if saved:
        # Shadow sidecar is observe-only: any shadow-ready prediction gets a SIM event,
        # not only models promoted to trade_active with validation gate passed.
        await save_model_family_shadow_trade(
            settings,
            _shadow_trade_payload(settings, result),
            deps["create_batch_combo_simulation_trade"],
        )


async def save_model_family_shadow_trade(
    parent: AutoTradeSettings,
    result: dict,
    create_trade: Callable[[AutoTradeSettings, dict], Any],
) -> None:
    if result.get("trade_quality_passed"):
        await asyncio.to_thread(create_trade, parent, result)


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


def _shadow_trade_payload(parent: AutoTradeSettings, result: dict[str, Any]) -> dict[str, Any]:
    return {
        **result,
        "trade_quality_passed": True,
    }


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
