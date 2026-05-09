from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.rule_backtest_service import run_rule_backtest
from app.services.rule_config import RULE_DURATION

router = APIRouter(prefix="/api/rules", tags=["rules"])


@router.get("/backtest")
def backtest_rules(
    symbol: str = Query(..., min_length=6),
    duration: str = Query(RULE_DURATION),
    save: bool = Query(True),
    strategyKey: str | None = Query(None),
) -> dict:
    try:
        return run_rule_backtest(symbol, duration, save, strategy_key=strategyKey)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
