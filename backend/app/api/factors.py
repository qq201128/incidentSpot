from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.services.factor_backtest_batch_service import run_all_factor_backtests
from app.services.factor_backtest_service import run_factor_backtest
from app.services.factor_cache_metadata import cache_is_usable
from app.services.factor_ranking_background import refresh_symbol_rankings
from app.services.factor_ranking_cache_service import (
    factor_ranking_precomputed_symbols,
    get_cached_ranking,
)
from app.services.factor_catalog import (
    get_factor_payload_by_name,
    list_combo_factor_payloads,
    list_single_factor_categories,
    list_single_factor_payloads,
)
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

router = APIRouter(prefix="/api/factors", tags=["factors"])
logger = logging.getLogger("uvicorn.error")


def _filter_ranking_by_category(ranking: list[dict], category: str | None) -> list[dict]:
    if not category:
        return ranking
    return [row for row in ranking if row.get("category") == category]


def _ranking_sort_key(row: dict) -> tuple[float, float]:
    return (float(row.get("factorScore") or 0.0), abs(float(row.get("ir") or 0.0)))


@router.get("/list")
def list_factors(
    category: str | None = None,
) -> dict:
    try:
        factors = list_single_factor_payloads(category)
        combo_factors = list_combo_factor_payloads()
        categories = list_single_factor_categories()
        return {
            "factors": factors,
            "comboFactors": combo_factors,
            "categories": categories,
            "total": len(factors),
            "comboTotal": len(combo_factors),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/detail/{factor_name}")
def get_factor_detail(factor_name: str) -> dict:
    payload = get_factor_payload_by_name(factor_name)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Factor not found: {factor_name}")
    return payload


@router.get("/backtest/all")
def backtest_all_factors(symbol: str = Query(..., min_length=6)) -> dict:
    try:
        return run_all_factor_backtests(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise HTTPException(status_code=400, detail=f"unsupported duration: {duration}")

    sym_u = symbol.upper()
    precomputed = factor_ranking_precomputed_symbols()
    cached = get_cached_ranking(sym_u, duration)

    if cached is None:
        return {
            "symbol": sym_u,
            "duration": duration,
            "category": category,
            "ranking": [],
            "total": 0,
            "updatedAt": None,
            "source": "none",
            "precomputedSymbols": precomputed,
            "hint": "排名由后台定时写入缓存；当前交易对/周期尚无数据。可将该交易对加入 FACTOR_RANKING_SYMBOLS 或使用 POST /ranking/refresh 排队重算。",
        }
    if not cache_is_usable(cached):
        return _stale_factor_ranking(sym_u, duration, category, cached, precomputed)

    full = list(cached["ranking"])
    filtered = _filter_ranking_by_category(full, category)
    filtered.sort(key=_ranking_sort_key, reverse=True)

    return {
        "symbol": sym_u,
        "duration": duration,
        "category": category,
        "ranking": filtered,
        "total": len(filtered),
        "updatedAt": cached["updatedAt"],
        "source": "cache",
        "precomputedSymbols": precomputed,
    }


def _background_refresh_rankings(symbol: str, duration: str | None) -> None:
    try:
        refresh_symbol_rankings(symbol, duration)
    except Exception:
        logger.exception("background factor ranking refresh failed: %s %s", symbol, duration)


@router.post("/ranking/refresh")
def factor_ranking_refresh(
    background_tasks: BackgroundTasks,
    symbol: str = Query(..., min_length=6),
    duration: str | None = Query(None, description="omit to refresh all supported durations"),
) -> dict:
    if duration is not None and duration not in SUPPORTED_RULE_DURATIONS:
        raise HTTPException(status_code=400, detail=f"unsupported duration: {duration}")
    sym_u = symbol.upper()
    background_tasks.add_task(_background_refresh_rankings, sym_u, duration)
    return {
        "ok": True,
        "symbol": sym_u,
        "duration": duration,
        "message": "已排队后台重算并写入缓存；完成后刷新页面或稍候再加载排名。",
    }


def _stale_factor_ranking(
    symbol: str,
    duration: str,
    category: str | None,
    cached: dict,
    precomputed: list[str],
) -> dict:
    return {
        "symbol": symbol,
        "duration": duration,
        "category": category,
        "ranking": [],
        "total": 0,
        "updatedAt": cached.get("updatedAt"),
        "source": "stale_cache",
        "staleRankingTotal": cached.get("total"),
        "cacheStatus": cached.get("cacheStatus"),
        "precomputedSymbols": precomputed,
        "hint": "因子排名缓存对应的历史数据已变化或缺少数据指纹；请刷新重算后再使用。",
    }


@router.get("/categories")
def get_categories() -> dict:
    return {"categories": list_single_factor_categories()}
