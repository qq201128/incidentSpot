from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.services.factor_backtest_batch_service import run_all_factor_backtests
from app.services.factor_backtest_service import run_factor_backtest
from app.services.factor_cache_metadata import cache_is_usable
from app.services.background_loop_status import record_loop_failure, record_loop_success
from app.services.factor_catalog_summaries import (
    list_combo_factor_summaries,
    list_single_factor_summaries,
)
from app.services.factor_combo_metrics import combo_metrics_for_factor, combo_period_scores
from app.services.factor_ranking_background import refresh_symbol_rankings
from app.services.factor_ranking_cache_service import (
    factor_ranking_precomputed_symbols,
    get_cached_ranking,
)
from app.services.factor_ranking_page import DEFAULT_RANKING_PAGE_SIZE, build_ranking_page
from app.services.factor_catalog import (
    get_factor_payload_by_name,
    list_single_factor_categories,
)
from app.services.factor_page_service import (
    build_factor_alerts,
    build_factor_list_page,
    build_factor_overview,
    build_factor_page_bundle,
    build_factor_page_context,
    build_factor_period_scores,
    enrich_factor_summary,
    ranking_metrics_for_factor,
)
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

router = APIRouter(prefix="/api/factors", tags=["factors"])
logger = logging.getLogger("uvicorn.error")
BACKGROUND_REFRESH_LOOP = "factor_ranking"


def _filter_ranking_by_category(ranking: list[dict], category: str | None) -> list[dict]:
    if not category:
        return ranking
    return [row for row in ranking if row.get("category") == category]


def _ranking_sort_key(row: dict) -> tuple[float, float]:
    return (float(row.get("factorScore") or 0.0), abs(float(row.get("ir") or 0.0)))


def _query_str(value: object, default: str | None = None) -> str | None:
    return value if isinstance(value, str) else default


def _query_int(value: object, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


@router.get("/list")
def list_factors(
    *,
    category: str | None = None,
    kind: str = Query("single", pattern="^(single|combo)$"),
    q: str | None = Query(None, description="search name/formula/source"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
) -> dict:
    safe_category = _query_str(category)
    query = _query_str(q)
    try:
        return build_factor_list_page(
            category=safe_category,
            kind=_query_str(kind, "single") or "single",
            query=query,
            page=_query_int(page, 1),
            page_size=_query_int(page_size, 20),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/page")
def factor_page(
    *,
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    category: str | None = Query(None),
    kind: str = Query("single", pattern="^(single|combo)$"),
    q: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
) -> dict:
    safe_duration = _query_str(duration, "10m") or "10m"
    safe_category = _query_str(category)
    query = _query_str(q)
    if safe_duration not in SUPPORTED_RULE_DURATIONS:
        raise HTTPException(status_code=400, detail=f"unsupported duration: {safe_duration}")
    try:
        return build_factor_page_bundle(
            symbol,
            safe_duration,
            category=safe_category,
            kind=_query_str(kind, "single") or "single",
            query=query,
            page=_query_int(page, 1),
            page_size=_query_int(page_size, 20),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/overview")
def factor_overview(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    category: str | None = Query(None),
) -> dict:
    safe_duration = _query_str(duration, "10m") or "10m"
    if safe_duration not in SUPPORTED_RULE_DURATIONS:
        raise HTTPException(status_code=400, detail=f"unsupported duration: {safe_duration}")
    return build_factor_page_context(symbol, safe_duration, category=_query_str(category))


@router.get("/alerts")
def factor_alerts(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
) -> dict:
    safe_duration = _query_str(duration, "10m") or "10m"
    if safe_duration not in SUPPORTED_RULE_DURATIONS:
        raise HTTPException(status_code=400, detail=f"unsupported duration: {safe_duration}")
    sym_u = symbol.upper()
    return {
        "symbol": sym_u,
        "duration": safe_duration,
        "alerts": build_factor_alerts(sym_u, safe_duration),
    }


@router.get("/detail/{factor_name}/scores")
def factor_period_scores(
    factor_name: str,
    symbol: str = Query(..., min_length=6),
) -> dict:
    if _is_combo_factor(factor_name):
        return combo_period_scores(symbol, factor_name)
    return build_factor_period_scores(symbol, factor_name)


@router.get("/detail/{factor_name}")
def get_factor_detail(
    factor_name: str,
    symbol: str | None = Query(None, min_length=6),
    duration: str | None = Query(None),
) -> dict:
    payload = get_factor_payload_by_name(factor_name)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Factor not found: {factor_name}")
    enriched = enrich_factor_summary(payload)
    safe_symbol = _query_str(symbol)
    safe_duration = _query_str(duration)
    if safe_symbol and safe_duration:
        if safe_duration not in SUPPORTED_RULE_DURATIONS:
            raise HTTPException(status_code=400, detail=f"unsupported duration: {safe_duration}")
        metrics = _metrics_for_factor(safe_symbol.upper(), safe_duration, factor_name)
        if metrics:
            enriched = enrich_factor_summary(payload, metrics)
    return enriched


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
    q: str | None = Query(None, description="search factor name/category/source"),
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_RANKING_PAGE_SIZE, alias="pageSize", ge=1, le=100),
) -> dict:
    safe_duration = _query_str(duration, "10m") or "10m"
    if safe_duration not in SUPPORTED_RULE_DURATIONS:
        raise HTTPException(status_code=400, detail=f"unsupported duration: {safe_duration}")

    sym_u = symbol.upper()
    safe_category = _query_str(category)
    query = _query_str(q)
    precomputed = factor_ranking_precomputed_symbols()
    safe_page = _query_int(page, 1)
    safe_page_size = _query_int(page_size, DEFAULT_RANKING_PAGE_SIZE)
    cached = get_cached_ranking(sym_u, safe_duration)

    if cached is None:
        return {
            "symbol": sym_u,
            "duration": safe_duration,
            "category": safe_category,
            "updatedAt": None,
            "source": "none",
            "precomputedSymbols": precomputed,
            "hint": "排名由后台定时写入缓存；当前交易对/周期尚无数据。可将该交易对加入 FACTOR_RANKING_SYMBOLS 或使用 POST /ranking/refresh 排队重算。",
            **build_ranking_page([], query, safe_page, safe_page_size),
        }
    if not cache_is_usable(cached):
        return _stale_factor_ranking(
            symbol=sym_u,
            duration=safe_duration,
            category=safe_category,
            query=query,
            page=safe_page,
            page_size=safe_page_size,
            cached=cached,
            precomputed=precomputed,
        )

    full = list(cached["ranking"])
    filtered = _filter_ranking_by_category(full, safe_category)
    filtered.sort(key=_ranking_sort_key, reverse=True)
    page_payload = build_ranking_page(filtered, query, safe_page, safe_page_size)

    return {
        "symbol": sym_u,
        "duration": safe_duration,
        "category": safe_category,
        "ranking": page_payload["ranking"],
        "updatedAt": cached["updatedAt"],
        "source": "cache",
        "precomputedSymbols": precomputed,
        "rankingDiagnostics": cached.get("rankingDiagnostics") or {},
        "rankingFailures": cached.get("rankingFailures") or [],
        **page_payload,
    }


def _background_refresh_rankings(symbol: str, duration: str | None) -> None:
    try:
        refresh_symbol_rankings(symbol, duration)
        record_loop_success(
            BACKGROUND_REFRESH_LOOP,
            {"stage": "manual_api_refresh", "symbol": symbol, "duration": duration},
        )
    except Exception as exc:
        record_loop_failure(
            BACKGROUND_REFRESH_LOOP,
            exc,
            {"stage": "manual_api_refresh", "symbol": symbol, "duration": duration},
        )
        logger.exception("background factor ranking refresh failed: %s %s", symbol, duration)
        raise


@router.post("/ranking/refresh")
def factor_ranking_refresh(
    background_tasks: BackgroundTasks,
    symbol: str = Query(..., min_length=6),
    duration: str | None = Query(None, description="omit to refresh all supported durations"),
) -> dict:
    safe_duration = _query_str(duration)
    if safe_duration is not None and safe_duration not in SUPPORTED_RULE_DURATIONS:
        raise HTTPException(status_code=400, detail=f"unsupported duration: {safe_duration}")
    sym_u = symbol.upper()
    background_tasks.add_task(_background_refresh_rankings, sym_u, safe_duration)
    return {
        "ok": True,
        "symbol": sym_u,
        "duration": safe_duration,
        "message": "已排队后台重算并写入缓存；完成后刷新页面或稍候再加载排名。",
    }


def _stale_factor_ranking(
    *,
    symbol: str,
    duration: str,
    category: str | None,
    query: str | None,
    page: int,
    page_size: int,
    cached: dict,
    precomputed: list[str],
) -> dict:
    stale_rows = _filter_ranking_by_category(list(cached.get("ranking") or []), category)
    return {
        "symbol": symbol,
        "duration": duration,
        "category": category,
        "updatedAt": cached.get("updatedAt"),
        "source": "stale_cache",
        "staleRankingTotal": cached.get("total"),
        "cacheStatus": cached.get("cacheStatus"),
        "precomputedSymbols": precomputed,
        "hint": "因子排名缓存对应的历史数据已变化或缺少数据指纹；请刷新重算后再使用。",
        **build_ranking_page(stale_rows, query, page, page_size),
    }


@router.get("/categories")
def get_categories() -> dict:
    return {"categories": list_single_factor_categories()}


def _metrics_for_factor(symbol: str, duration: str, factor_name: str) -> dict | None:
    if _is_combo_factor(factor_name):
        return combo_metrics_for_factor(symbol, duration, factor_name)
    return ranking_metrics_for_factor(symbol, duration, factor_name)


def _is_combo_factor(factor_name: str) -> bool:
    return factor_name.startswith("combo__") or factor_name.startswith("goal_combo__")
