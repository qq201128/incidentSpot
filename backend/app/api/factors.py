from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.services.factor_backtest_batch_service import run_all_factor_backtests
from app.services.factor_backtest_service import run_factor_backtest
from app.services.factor_catalog_summaries import (
    list_combo_factor_summaries,
    list_single_factor_summaries,
)
from app.services.factor_combo_metrics import cached_combo_row_for_display, combo_metrics_for_factor, combo_period_scores
from app.services.factor_ranking_api_payloads import background_refresh_rankings, factor_ranking_payload
from app.services.factor_ranking_page import DEFAULT_RANKING_PAGE_SIZE
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


def _query_str(value: object, default: str | None = None) -> str | None:
    return value if isinstance(value, str) else default


def _query_int(value: object, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


@router.get("/list")
def list_factors(
    *,
    category: str | None = None,
    kind: str = Query("single", pattern="^(single|combo)$"),
    symbol: str | None = Query(None, min_length=6),
    duration: str | None = Query(None),
    q: str | None = Query(None, description="search name/formula/source"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, alias="pageSize", ge=1, le=200),
) -> dict:
    safe_category = _query_str(category)
    safe_duration = _query_str(duration)
    if safe_duration is not None and safe_duration not in SUPPORTED_RULE_DURATIONS:
        raise HTTPException(status_code=400, detail=f"unsupported duration: {safe_duration}")
    query = _query_str(q)
    try:
        return build_factor_list_page(
            category=safe_category,
            kind=_query_str(kind, "single") or "single",
            symbol=_query_str(symbol),
            duration=safe_duration,
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
    page_size: int = Query(20, alias="pageSize", ge=1, le=200),
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
    safe_symbol = _query_str(symbol)
    safe_duration = _query_str(duration)
    if safe_symbol and safe_duration and _is_combo_factor(factor_name):
        if safe_duration not in SUPPORTED_RULE_DURATIONS:
            raise HTTPException(status_code=400, detail=f"unsupported duration: {safe_duration}")
        cached_combo = cached_combo_row_for_display(safe_symbol.upper(), safe_duration, factor_name)
        if cached_combo:
            return enrich_factor_summary(_combo_detail_payload(cached_combo), cached_combo)
    payload = get_factor_payload_by_name(factor_name)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Factor not found: {factor_name}")
    enriched = enrich_factor_summary(payload)
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
    safe_page = _query_int(page, 1)
    safe_page_size = _query_int(page_size, DEFAULT_RANKING_PAGE_SIZE)
    return factor_ranking_payload(
        symbol=sym_u,
        duration=safe_duration,
        category=safe_category,
        query=query,
        page=safe_page,
        page_size=safe_page_size,
    )


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
    background_tasks.add_task(background_refresh_rankings, sym_u, safe_duration)
    return {
        "ok": True,
        "symbol": sym_u,
        "duration": safe_duration,
        "message": "已排队后台重算并写入缓存；完成后刷新页面或稍候再加载排名。",
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


def _combo_detail_payload(row: dict) -> dict:
    name = str(row.get("factorName") or row.get("name") or "")
    display = str(row.get("factorDisplayName") or row.get("displayName") or row.get("description") or name)
    return {
        "name": name,
        "displayName": display,
        "description": display,
        "formula": str(row.get("formula") or name),
        "sourceFile": "mined_factor_library.json",
        "timeframes": [str(row.get("duration"))] if row.get("duration") else [],
        "direction": "higher_better",
    }
