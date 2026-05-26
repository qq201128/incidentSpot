from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from app.services.auto_trade_types import AutoTradeSettings
from app.services.model_family_config import MODEL_FAMILIES
from app.services.paper_live_candidate_service import log_prediction_failure
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY, strategy_entry_grace_ms, strategy_supports_duration


async def save_candidate_collection_predictions(settings: AutoTradeSettings, entry_open_time: int, write_lock: asyncio.Lock, deps: dict[str, Any]) -> None:
    await save_factor_candidate_signals(settings, entry_open_time, write_lock, deps)
    await save_factor_combo_shadow_predictions(settings, entry_open_time, write_lock, deps)


async def save_factor_combo_shadow_predictions(settings: AutoTradeSettings, entry_open_time: int, write_lock: asyncio.Lock, deps: dict[str, Any]) -> None:
    if settings.strategy_key != FACTOR_COMBO_STRATEGY_KEY:
        return
    try:
        results = await asyncio.to_thread(
            deps["predict_eligible_factor_combo_rows"],
            settings.symbol,
            settings.duration,
            entry_open_time=entry_open_time,
            entry_grace_ms=strategy_entry_grace_ms(settings.strategy_key),
        )
    except Exception as exc:
        _log_collection_failure(settings, "factor_combo_shadow_prediction", exc, entry_open_time)
        raise
    for result in results:
        await _save_prediction_and_simulation_trade(settings, result, write_lock, deps)


async def save_factor_candidate_signals(settings: AutoTradeSettings, entry_open_time: int, write_lock: asyncio.Lock, deps: dict[str, Any]) -> None:
    if not strategy_supports_duration(settings.strategy_key, settings.duration):
        return
    try:
        results = await asyncio.to_thread(
            deps["predict_factor_candidate_signals"],
            settings.symbol,
            settings.duration,
            entry_open_time=entry_open_time,
            entry_grace_ms=strategy_entry_grace_ms(settings.strategy_key),
        )
    except Exception as exc:
        _log_collection_failure(settings, "factor_candidate_signal_prediction", exc, entry_open_time)
        raise
    for result in results:
        await _save_prediction_and_simulation_trade(settings, result, write_lock, deps)


async def _save_prediction_and_simulation_trade(
    settings: AutoTradeSettings,
    result: dict[str, Any],
    write_lock: asyncio.Lock,
    deps: dict[str, Any],
) -> None:
    saved = await deps["save_prediction"](result, write_lock)
    if saved:
        await asyncio.to_thread(deps["create_batch_combo_simulation_trade"], settings, result)


async def save_one_model_family_shadow_prediction(
    family: str,
    settings: AutoTradeSettings,
    entry_open_time: int,
    write_lock: asyncio.Lock,
    deps: dict[str, Any],
) -> None:
    status_loader = deps["lstm_model_status"] if family == "lstm" else deps["model_family_status"]
    status_args = (settings.symbol, settings.duration) if family == "lstm" else (family, settings.symbol, settings.duration)
    status = await asyncio.to_thread(status_loader, *status_args)
    if not status.get("shadowPredictionReady"):
        _log_model_skip(settings, family, status, entry_open_time)
        deps["log_model_family_shadow_skip"](settings, family, status, role="sidecar")
        return
    predictor = deps["predict_lstm_shadow_prediction"] if family == "lstm" else deps["predict_model_family_shadow_prediction"]
    args = (settings.symbol, settings.duration) if family == "lstm" else (family, settings.symbol, settings.duration)
    result = await asyncio.to_thread(predictor, *args, entry_open_time=entry_open_time)
    saved = await deps["save_prediction"](result, write_lock)
    if saved:
        await save_model_family_shadow_trade(settings, result, deps["create_batch_combo_simulation_trade"])


async def save_model_family_shadow_trade(
    parent: AutoTradeSettings,
    result: dict,
    create_trade: Callable[[AutoTradeSettings, dict], Any],
) -> None:
    if result.get("trade_quality_passed"):
        await asyncio.to_thread(create_trade, parent, result)


def _log_collection_failure(
    settings: AutoTradeSettings,
    stage: str,
    exc: Exception,
    entry_open_time: int,
) -> None:
    log_prediction_failure(
        candidate_key=settings.strategy_key,
        strategy_key=settings.strategy_key,
        symbol=settings.symbol,
        duration=settings.duration,
        stage=stage,
        reason=str(exc),
        details={"entryOpenTime": int(entry_open_time), "exceptionType": type(exc).__name__},
    )


def _log_model_skip(
    settings: AutoTradeSettings,
    family: str,
    status: dict[str, Any],
    entry_open_time: int,
) -> None:
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
