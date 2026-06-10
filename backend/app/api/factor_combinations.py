from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.services.combination_ranking_page import build_combination_ranking_page
from app.services.factor_cache_metadata import cache_is_usable
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_combination_coverage_view import data_coverage_summary
from app.services.factor_combination_live_service import build_combination_signal_watchlist
from app.services.experiment_profiles import normalize_experiment_profile
from app.services.high_winrate_combo_cache_service import get_cached_high_winrate_combo_ranking
from app.services.high_winrate_combo_view import build_high_winrate_combo_view
from app.services.high_winrate_combo_view import regular_ranking_view
from app.services.paper_live_candidate_service import paper_live_candidate_report
from app.services.paper_live_report_cache import get_cached_paper_live_report
from app.services.paper_live_daily_loop_service import run_paper_live_daily_closed_loop
from app.services.factor_combination_refresh_api import (
    background_refresh_combo_rankings,
    combination_config,
    config_response,
)
from app.services.factor_combination_incremental_refresh import refresh_incremental_combination_cache
from app.services.factor_combination_ranking_view import (
    ranking_rows,
    ranking_visibility,
    regular_ranking_rows,
    stale_regular_rows,
)
from app.services.factor_ranking_cache_service import factor_ranking_precomputed_symbols
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

router = APIRouter(prefix="/api/factors/combinations", tags=["factors"])
DEFAULT_COMBO_TOP_PER_DURATION = 3
DEFAULT_COMBO_SIGNAL_LIMIT = 12
DEFAULT_RANKING_PAGE_SIZE = 6
MAX_RANKING_PAGE_SIZE = 100


def _query_str(value: object, default: str | None = None) -> str | None:
    return value if isinstance(value, str) else default


def _query_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _query_bool(value: object) -> bool:
    return value if isinstance(value, bool) else False


@router.get("/ranking")
def factor_combination_ranking(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    q: str | None = Query(None, description="search combo name/member/display name"),
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_RANKING_PAGE_SIZE, alias="pageSize", ge=1, le=MAX_RANKING_PAGE_SIZE),
) -> dict:
    safe_duration = _query_str(duration, "10m") or "10m"
    _validate_duration(safe_duration)
    sym_u = symbol.upper()
    query = _query_str(q)
    safe_page = _query_int(page) or 1
    safe_page_size = _query_int(page_size) or DEFAULT_RANKING_PAGE_SIZE
    cached = get_cached_combination_ranking(sym_u, safe_duration)
    high_winrate_view = _high_winrate_view(sym_u, safe_duration)
    if cached is None:
        return {
            **_empty_combination_ranking(sym_u, safe_duration),
            **build_combination_ranking_page([], query, safe_page, safe_page_size),
            **high_winrate_view,
        }
    if not cache_is_usable(cached):
        return {
            **_stale_combination_ranking(sym_u, safe_duration, cached),
            **build_combination_ranking_page(stale_regular_rows(cached), query, safe_page, safe_page_size),
            **high_winrate_view,
        }
    ranking = regular_ranking_view(regular_ranking_rows(cached))
    visibility = ranking_visibility(cached, ranking)
    page_payload = build_combination_ranking_page(ranking, query, safe_page, safe_page_size)
    return {
        **cached,
        "ranking": page_payload["ranking"],
        "total": page_payload["total"],
        "passedRankingTotal": len(ranking_rows(cached)),
        "evaluatedRankingTotal": visibility["evaluatedTotal"],
        "source": "cache",
        **page_payload,
        **visibility,
        **high_winrate_view,
    }


@router.get("/signals")
def factor_combination_signals(
    symbol: str = Query(..., min_length=6),
    limit: int | None = Query(None, gt=0),
    top_per_duration: int | None = Query(None, alias="topPerDuration", gt=0),
) -> dict:
    try:
        return build_combination_signal_watchlist(
            symbol.upper(),
            limit=_query_int(limit),
            top_per_duration=_query_int(top_per_duration),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/paper-live/candidates")
def paper_live_candidates(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
) -> dict:
    safe_duration = _query_str(duration, "10m") or "10m"
    _validate_duration(safe_duration)
    sym = symbol.upper()
    return get_cached_paper_live_report(sym, safe_duration, build=paper_live_candidate_report)


@router.post("/paper-live/daily-loop")
def paper_live_daily_loop(
    symbol: str | None = Query(None, min_length=6),
    duration: str | None = Query(None),
) -> dict:
    safe_symbol = _query_str(symbol)
    safe_duration = _query_str(duration)
    _validate_optional_duration(safe_duration)
    symbols = [safe_symbol.upper()] if safe_symbol else None
    durations = [safe_duration] if safe_duration else None
    return run_paper_live_daily_closed_loop(symbols=symbols, durations=durations)


@router.post("/refresh")
def factor_combination_refresh(
    background_tasks: BackgroundTasks,
    *,
    symbol: str = Query(..., min_length=6),
    duration: str | None = Query(None, description="omit to refresh all supported durations"),
    profile: str = Query("full"),
    base_factor_limit: int | None = Query(None, alias="baseFactorLimit"),
    combo_sizes: str | None = Query(None, alias="comboSizes"),
    result_limit: int | None = Query(None, alias="resultLimit"),
    incremental: bool = Query(False),
    batch_size: int = Query(120, alias="batchSize", ge=1, le=1000),
    lookback_days: int | None = Query(None, alias="lookbackDays", ge=30, le=2000),
) -> dict:
    safe_duration = _query_str(duration)
    _validate_optional_duration(safe_duration)
    sym_u = symbol.upper()
    safe_incremental = _query_bool(incremental)
    safe_batch_size = _query_int(batch_size) or 120
    if safe_incremental:
        report = refresh_incremental_combination_cache(
            sym_u,
            safe_duration or "10m",
            batch_size=safe_batch_size,
            lookback_days=_query_int(lookback_days) or 365,
        )
        return {
            "ok": True,
            "symbol": sym_u,
            "duration": safe_duration or "10m",
            "profile": "incremental",
            "rankingTotal": len(report.get("ranking") or []),
            "evaluatedRankingTotal": len(report.get("evaluatedRanking") or []),
            "message": "已增量刷新普通组合展示缓存。",
        }
    config = combination_config(
        profile=_query_str(profile, "full") or "full",
        base_factor_limit=_query_int(base_factor_limit),
        combo_sizes=_query_str(combo_sizes),
        result_limit=_query_int(result_limit),
        lookback_days=_query_int(lookback_days),
    )
    background_tasks.add_task(background_refresh_combo_rankings, sym_u, safe_duration, config)
    return {
        "ok": True,
        "symbol": sym_u,
        "duration": safe_duration,
        "profile": normalize_experiment_profile(_query_str(profile, "full") or "full"),
        "searchConfig": config_response(config),
        "message": "已排队后台重算多因子组合并写入缓存。",
    }


def _empty_combination_ranking(symbol: str, duration: str) -> dict:
    return {
        "symbol": symbol,
        "duration": duration,
        "ranking": [],
        "total": 0,
        "updatedAt": None,
        "source": "none",
        "rawTotal": 0,
        "regularTotal": 0,
        "nestedComboFilteredCount": 0,
        "precomputedSymbols": factor_ranking_precomputed_symbols(),
        "hint": "多因子组合排名尚无缓存；可使用 POST /api/factors/combinations/refresh 排队重算。",
    }


def _stale_combination_ranking(symbol: str, duration: str, cached: dict) -> dict:
    ranking = stale_regular_rows(cached)
    visibility = ranking_visibility(cached, ranking)
    return {
        "symbol": symbol,
        "duration": duration,
        "total": len(ranking),
        "passedRankingTotal": len(ranking_rows(cached)),
        "evaluatedRankingTotal": visibility["evaluatedTotal"],
        "updatedAt": cached.get("updatedAt"),
        "source": "stale_cache",
        "cacheStatus": cached.get("cacheStatus"),
        **visibility,
        "precomputedSymbols": factor_ranking_precomputed_symbols(),
        "hint": "多因子组合缓存对应的历史数据已变化或缺少数据指纹；当前展示旧缓存，请刷新重算后再用于实盘判断。",
    }


def _high_winrate_view(symbol: str, duration: str) -> dict:
    cached = get_cached_high_winrate_combo_ranking(symbol, duration)
    return {
        **build_high_winrate_combo_view(cached, duration),
        "dataCoverage": data_coverage_summary(symbol, duration),
    }


def _validate_duration(duration: str) -> None:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise HTTPException(status_code=400, detail=f"unsupported duration: {duration}")


def _validate_optional_duration(duration: str | None) -> None:
    if duration is not None:
        _validate_duration(duration)
