from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.auto_trade_types import AutoTradeSettings
from app.services.auto_trade_service import (
    get_auto_trade_settings,
    list_auto_trade_strategy_payloads,
    update_auto_trade_settings,
)
from app.services.auto_trade_status import get_auto_trade_status
from app.services.simulation_slot_observability_service import simulation_slots_report
from app.services.strategy_registry import DEFAULT_STRATEGY_KEY

router = APIRouter(prefix="/api/auto-trade", tags=["auto-trade"])


class AutoTradeSettingsPayload(BaseModel):
    strategyKey: str | None = None
    enabled: bool
    liveTradingEnabled: bool = False
    symbol: str = Field(min_length=6)
    duration: str
    durationMinutes: int = Field(gt=0)
    qty: float = Field(gt=0)


@router.get("/settings")
def read_settings() -> dict:
    return get_auto_trade_settings().to_response()


@router.get("/status")
def read_status() -> dict:
    return get_auto_trade_status()


@router.get("/strategies")
def read_strategies() -> dict:
    return {"strategies": list_auto_trade_strategy_payloads()}


@router.get("/simulation-slots")
def read_simulation_slots(symbol: str = "BTCUSDT", duration: str = "10m") -> dict:
    try:
        return simulation_slots_report(symbol, duration)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/settings")
def update_settings(payload: AutoTradeSettingsPayload) -> dict:
    try:
        settings = _settings_from_payload(payload, DEFAULT_STRATEGY_KEY)
        return update_auto_trade_settings(settings).to_response()
    except ValueError as exc:
        _disable_invalid_auto_trade(payload, exc, DEFAULT_STRATEGY_KEY)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/strategies/{strategy_key}")
def update_strategy(strategy_key: str, payload: AutoTradeSettingsPayload) -> dict:
    try:
        settings = _settings_from_payload(payload, strategy_key)
        return update_auto_trade_settings(settings).to_response()
    except ValueError as exc:
        _disable_invalid_auto_trade(payload, exc, strategy_key)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _settings_from_payload(payload: AutoTradeSettingsPayload, strategy_key: str) -> AutoTradeSettings:
    if payload.liveTradingEnabled:
        raise ValueError("real trading is disabled in the current paper-live preparation phase")
    return AutoTradeSettings(
        strategy_key=payload.strategyKey or strategy_key,
        enabled=payload.enabled,
        symbol=payload.symbol,
        duration=payload.duration,
        duration_minutes=payload.durationMinutes,
        qty=payload.qty,
        live_trading_enabled=False,
    )


def _disable_invalid_auto_trade(
    payload: AutoTradeSettingsPayload,
    exc: ValueError,
    strategy_key: str,
) -> None:
    if "duration must be one of" not in str(exc):
        return
    update_auto_trade_settings(
        AutoTradeSettings(
            strategy_key=payload.strategyKey or strategy_key,
            enabled=False,
            symbol=payload.symbol,
            duration=payload.duration,
            duration_minutes=payload.durationMinutes,
            qty=payload.qty,
            live_trading_enabled=False,
        )
    )
