from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.services.factor_backtest_batch_service import run_all_factor_backtests
from app.services.factor_backtest_service import run_factor_backtest
from app.services.factor_combination_background import refresh_symbol_combination_rankings
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_combination_live_service import build_combination_signal_watchlist
from app.services.factor_combination_service import (
    DEFAULT_BASE_FACTOR_LIMIT,
    DEFAULT_RESULT_LIMIT,
    MIN_COMBO_SIZE,
    CombinationSearchConfig,
)
from app.services.factor_ranking_background import refresh_symbol_rankings
from app.services.factor_ranking_cache_service import (
    factor_ranking_precomputed_symbols,
    get_cached_ranking,
)
from app.services.factor_registry import (
    factor_payload,
    get_factor,
    list_factor_categories,
    list_factor_payloads,
)
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

router = APIRouter(prefix="/api/factors", tags=["factors"])
logger = logging.getLogger("uvicorn.error")
DEFAULT_COMBO_SIZE_QUERY = "2,3"
DEFAULT_COMBO_SIGNAL_LIMIT = 4


def _filter_ranking_by_category(ranking: list[dict], category: str | None) -> list[dict]:
    if not category:
        return ranking
    return [row for row in ranking if row.get("category") == category]


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


@router.get("/combinations/ranking")
def factor_combination_ranking(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
) -> dict:
    _validate_duration(duration)
    sym_u = symbol.upper()
    cached = get_cached_combination_ranking(sym_u, duration)
    if cached is None:
        return _empty_combination_ranking(sym_u, duration)
    return {**cached, "source": "cache"}


@router.get("/combinations/signals")
def factor_combination_signals(
    symbol: str = Query(..., min_length=6),
    limit: int = Query(DEFAULT_COMBO_SIGNAL_LIMIT, gt=0),
) -> dict:
    try:
        return build_combination_signal_watchlist(symbol.upper(), limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/combinations/refresh")
def factor_combination_refresh(
    background_tasks: BackgroundTasks,
    symbol: str = Query(..., min_length=6),
    duration: str | None = Query(None, description="omit to refresh all supported durations"),
    base_factor_limit: int = Query(DEFAULT_BASE_FACTOR_LIMIT, alias="baseFactorLimit"),
    combo_sizes: str = Query(DEFAULT_COMBO_SIZE_QUERY, alias="comboSizes"),
    result_limit: int = Query(DEFAULT_RESULT_LIMIT, alias="resultLimit"),
) -> dict:
    _validate_optional_duration(duration)
    sym_u = symbol.upper()
    config = _combination_config(base_factor_limit, combo_sizes, result_limit)
    background_tasks.add_task(_background_refresh_combo_rankings, sym_u, duration, config)
    return {
        "ok": True,
        "symbol": sym_u,
        "duration": duration,
        "searchConfig": _config_response(config),
        "message": "已排队后台重算多因子组合并写入缓存。",
    }


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

    full = list(cached["ranking"])
    filtered = _filter_ranking_by_category(full, category)
    filtered.sort(key=lambda x: abs(x.get("ir") or 0), reverse=True)

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


def _background_refresh_combo_rankings(
    symbol: str,
    duration: str | None,
    config: CombinationSearchConfig,
) -> None:
    try:
        refresh_symbol_combination_rankings(symbol, duration, config)
    except Exception:
        logger.exception("background factor combo refresh failed: %s %s", symbol, duration)


def _empty_combination_ranking(symbol: str, duration: str) -> dict:
    return {
        "symbol": symbol,
        "duration": duration,
        "ranking": [],
        "total": 0,
        "updatedAt": None,
        "source": "none",
        "precomputedSymbols": factor_ranking_precomputed_symbols(),
        "hint": "多因子组合排名尚无缓存；可使用 POST /api/factors/combinations/refresh 排队重算。",
    }


def _combination_config(
    base_factor_limit: int,
    combo_sizes: str,
    result_limit: int,
) -> CombinationSearchConfig:
    try:
        sizes = _parse_combo_sizes(combo_sizes)
        _validate_combo_config_values(base_factor_limit, sizes, result_limit)
        return CombinationSearchConfig(base_factor_limit, sizes, result_limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _parse_combo_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes:
        raise ValueError("comboSizes must contain at least one integer")
    return sizes


def _validate_combo_config_values(
    base_factor_limit: int,
    sizes: tuple[int, ...],
    result_limit: int,
) -> None:
    if base_factor_limit < max(sizes):
        raise ValueError("baseFactorLimit must be >= largest combo size")
    if result_limit <= 0 or any(size < MIN_COMBO_SIZE for size in sizes):
        raise ValueError("resultLimit must be > 0 and comboSizes must be >= 2")


def _config_response(config: CombinationSearchConfig) -> dict:
    return {
        "baseFactorLimit": config.base_factor_limit,
        "comboSizes": list(config.combo_sizes),
        "resultLimit": config.result_limit,
        "method": config.method,
    }


def _validate_duration(duration: str) -> None:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise HTTPException(status_code=400, detail=f"unsupported duration: {duration}")


def _validate_optional_duration(duration: str | None) -> None:
    if duration is not None:
        _validate_duration(duration)


@router.get("/categories")
def get_categories() -> dict:
    return {"categories": list_factor_categories()}
