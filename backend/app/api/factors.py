from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.factor_backtest_service import run_factor_backtest, run_factor_ranking
from app.services.factor_registry import (
    get_factor,
    list_factor_categories,
    list_factor_payloads,
    factor_payload,
)

router = APIRouter(prefix="/api/factors", tags=["factors"])


@router.get("/list")
def list_factors(
    category: str | None = Query(None, description="Filter by category"),
) -> dict:
    try:
        factors = list_factor_payloads(category)
        categories = list_factor_categories()
        return {
            "factors": factors,
            "categories": categories,
            "total": len(factors),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/detail/{factor_name}")
def get_factor_detail(factor_name: str) -> dict:
    factor = get_factor(factor_name)
    if factor is None:
        raise HTTPException(status_code=404, detail=f"Factor not found: {factor_name}")
    return factor_payload(factor)


@router.get("/backtest/{factor_name}")
def backtest_factor(
    factor_name: str,
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
) -> dict:
    try:
        return run_factor_backtest(factor_name, symbol, duration)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/ranking")
def factor_ranking(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    category: str | None = Query(None),
) -> dict:
    try:
        ranking = run_factor_ranking(symbol, duration, category)
        return {
            "symbol": symbol.upper(),
            "duration": duration,
            "category": category,
            "ranking": ranking,
            "total": len(ranking),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/categories")
def get_categories() -> dict:
    return {"categories": list_factor_categories()}
