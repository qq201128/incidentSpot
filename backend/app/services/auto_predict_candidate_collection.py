from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from app.services.auto_trade_types import AutoTradeSettings
from app.services.model_family_config import MODEL_FAMILIES
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY, strategy_entry_grace_ms, strategy_supports_duration


async def save_candidate_collection_predictions(settings: AutoTradeSettings, entry_open_time: int, write_lock: asyncio.Lock, deps: dict[str, Any]) -> None:
    await save_factor_candidate_signals(settings, entry_open_time, write_lock, deps)
    await save_factor_combo_shadow_predictions(settings, entry_open_time, write_lock, deps)


async def save_factor_combo_shadow_predictions(settings: AutoTradeSettings, entry_open_time: int, write_lock: asyncio.Lock, deps: dict[str, Any]) -> None:
    if settings.strategy_key != FACTOR_COMBO_STRATEGY_KEY:
        return
    results = await asyncio.to_thread(
        deps["predict_eligible_factor_combo_rows"],
        settings.symbol,
        settings.duration,
        entry_open_time=entry_open_time,
        entry_grace_ms=strategy_entry_grace_ms(settings.strategy_key),
    )
    for result in results:
        saved = await deps["save_prediction"](result, write_lock)
        if saved:
            await asyncio.to_thread(deps["create_batch_combo_simulation_trade"], settings, result)


async def save_factor_candidate_signals(settings: AutoTradeSettings, entry_open_time: int, write_lock: asyncio.Lock, deps: dict[str, Any]) -> None:
    if not strategy_supports_duration(settings.strategy_key, settings.duration):
        return
    results = await asyncio.to_thread(
        deps["predict_factor_candidate_signals"],
        settings.symbol,
        settings.duration,
        entry_open_time=entry_open_time,
        entry_grace_ms=strategy_entry_grace_ms(settings.strategy_key),
    )
    for result in results:
        await deps["save_prediction"](result, write_lock)


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
