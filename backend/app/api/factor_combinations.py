from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.services.background_loop_status import record_loop_failure, record_loop_success
from app.services.factor_cache_metadata import cache_is_usable
from app.services.factor_combination_background import refresh_symbol_combination_rankings
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_combination_live_service import build_combination_signal_watchlist
from app.services.data_coverage_report import CoverageOptions, build_data_coverage_report
from app.services.experiment_profiles import combination_search_config_for_profile, normalize_experiment_profile
from app.services.high_winrate_combo_cache_service import get_cached_high_winrate_combo_ranking
from app.services.high_winrate_combo_view import build_high_winrate_combo_view
from app.services.high_winrate_combo_view import regular_ranking_view
from app.services.paper_live_candidate_service import paper_live_candidate_report
from app.services.paper_live_daily_loop_service import run_paper_live_daily_closed_loop
from app.services.factor_combination_service import MIN_COMBO_SIZE, CombinationSearchConfig
from app.services.factor_ranking_cache_service import factor_ranking_precomputed_symbols
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

router = APIRouter(prefix="/api/factors/combinations", tags=["factors"])
logger = logging.getLogger("uvicorn.error")
DEFAULT_COMBO_TOP_PER_DURATION = 3
DEFAULT_COMBO_SIGNAL_LIMIT = 12
DEFAULT_RANKING_PAGE_SIZE = 6
MAX_RANKING_PAGE_SIZE = 100
BACKGROUND_REFRESH_LOOP = "factor_combo_daily"


def _query_str(value: object, default: str | None = None) -> str | None:
    return value if isinstance(value, str) else default


def _query_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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
            **_paginated_ranking_payload([], query, safe_page, safe_page_size),
            **high_winrate_view,
        }
    if not cache_is_usable(cached):
        return {
            **_stale_combination_ranking(sym_u, safe_duration, cached),
            **_paginated_ranking_payload(_stale_regular_rows(cached), query, safe_page, safe_page_size),
            **high_winrate_view,
        }
    ranking = regular_ranking_view(_regular_ranking_rows(cached))
    visibility = _ranking_visibility(cached, ranking)
    page_payload = _paginated_ranking_payload(ranking, query, safe_page, safe_page_size)
    return {
        **cached,
        "ranking": page_payload["ranking"],
        "total": page_payload["total"],
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
    return paper_live_candidate_report(symbol.upper(), safe_duration)


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
) -> dict:
    safe_duration = _query_str(duration)
    _validate_optional_duration(safe_duration)
    sym_u = symbol.upper()
    config = _combination_config(
        profile=_query_str(profile, "full") or "full",
        base_factor_limit=_query_int(base_factor_limit),
        combo_sizes=_query_str(combo_sizes),
        result_limit=_query_int(result_limit),
    )
    background_tasks.add_task(_background_refresh_combo_rankings, sym_u, safe_duration, config)
    return {
        "ok": True,
        "symbol": sym_u,
        "duration": safe_duration,
        "profile": normalize_experiment_profile(_query_str(profile, "full") or "full"),
        "searchConfig": _config_response(config),
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
    ranking = _stale_regular_rows(cached)
    visibility = _ranking_visibility(cached, ranking)
    return {
        "symbol": symbol,
        "duration": duration,
        "total": len(ranking),
        "updatedAt": cached.get("updatedAt"),
        "source": "stale_cache",
        "cacheStatus": cached.get("cacheStatus"),
        **visibility,
        "precomputedSymbols": factor_ranking_precomputed_symbols(),
        "hint": "多因子组合缓存对应的历史数据已变化或缺少数据指纹；当前展示旧缓存，请刷新重算后再用于实盘判断。",
    }


def _stale_regular_rows(cached: dict) -> list[dict]:
    return regular_ranking_view(_regular_ranking_rows(cached))


def _high_winrate_view(symbol: str, duration: str) -> dict:
    cached = get_cached_high_winrate_combo_ranking(symbol, duration)
    return {
        **build_high_winrate_combo_view(cached, duration),
        "dataCoverage": _data_coverage_summary(symbol, duration),
    }


def _data_coverage_summary(symbol: str, duration: str) -> dict:
    report = build_data_coverage_report(CoverageOptions(symbol=symbol, interval=duration))
    return {
        "mainRange": report["mainRange"],
        "missingFeatureSources": _missing_feature_sources(report["tables"]),
    }


def _missing_feature_sources(tables: list[dict]) -> list[dict]:
    missing = []
    for table in tables:
        for row in table.get("rows") or []:
            if row.get("status") in {"healthy"}:
                continue
            missing.append({
                "table": table.get("table"),
                "status": row.get("status"),
                "coveragePct": row.get("coveragePct"),
                "missingReason": row.get("missingReason"),
            })
    return missing


def _regular_ranking_rows(cached: dict) -> list[dict]:
    return [row for row in _ranking_rows(cached) if not _has_combo_member(row)]


def _ranking_visibility(cached: dict, ranking: list[dict]) -> dict[str, int]:
    raw_rows = _ranking_rows(cached)
    return {
        "rawTotal": len(raw_rows),
        "regularTotal": len(ranking),
        "nestedComboFilteredCount": max(len(raw_rows) - len(ranking), 0),
    }


def _ranking_rows(cached: dict) -> list[dict]:
    ranking = cached.get("ranking")
    return list(ranking) if isinstance(ranking, list) else []


def _paginated_ranking_payload(
    ranking: list[dict],
    query: str | None,
    page: int,
    page_size: int,
) -> dict:
    filtered = _filter_ranking_rows(ranking, query)
    size = max(1, min(int(page_size), MAX_RANKING_PAGE_SIZE))
    page_count = max(1, (len(filtered) + size - 1) // size) if filtered else 1
    current = min(max(1, int(page)), page_count)
    start = (current - 1) * size
    return {
        "ranking": filtered[start:start + size],
        "total": len(filtered),
        "unfilteredTotal": len(ranking),
        "page": current,
        "pageSize": size,
        "pageCount": page_count,
        "query": (query or "").strip(),
    }


def _filter_ranking_rows(ranking: list[dict], query: str | None) -> list[dict]:
    q = (query or "").strip().lower()
    if not q:
        return list(ranking)
    return [row for row in ranking if _ranking_row_matches(row, q)]


def _ranking_row_matches(row: dict, query: str) -> bool:
    haystack = " ".join(
        [
            str(row.get("factorName") or ""),
            str(row.get("factorDisplayName") or ""),
            str(row.get("description") or ""),
            _member_search_text(row.get("members")),
        ]
    ).lower()
    return query in haystack


def _member_search_text(members: object) -> str:
    if not isinstance(members, list):
        return ""
    return " ".join(
        f"{member.get('name', '')} {member.get('displayName', '')}"
        for member in members
        if isinstance(member, dict)
    )


def _has_combo_member(row: dict) -> bool:
    members = row.get("members")
    if not isinstance(members, list):
        return False
    return any(_is_combo_name(member.get("name")) for member in members if isinstance(member, dict))


def _is_combo_name(name: object) -> bool:
    raw = str(name or "")
    return raw.startswith("combo__") or raw.startswith("goal_combo__")


def _combination_config(
    *,
    profile: str,
    base_factor_limit: int | None,
    combo_sizes: str | None,
    result_limit: int | None,
) -> CombinationSearchConfig:
    try:
        config = combination_search_config_for_profile(
            profile,
            base_factor_limit=base_factor_limit,
            combo_sizes=_parse_combo_sizes(combo_sizes) if combo_sizes is not None else None,
            result_limit=result_limit,
        )
        _validate_combo_config_values(config.base_factor_limit, config.combo_sizes, config.result_limit)
        return config
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _parse_combo_sizes(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
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
        "prefilterLimit": config.prefilter_limit,
        "beamWidth": config.beam_width,
        "parallelWorkers": config.parallel_workers,
        "method": config.method,
    }


def _validate_duration(duration: str) -> None:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise HTTPException(status_code=400, detail=f"unsupported duration: {duration}")


def _validate_optional_duration(duration: str | None) -> None:
    if duration is not None:
        _validate_duration(duration)


def _background_refresh_combo_rankings(
    symbol: str,
    duration: str | None,
    config: CombinationSearchConfig,
) -> None:
    try:
        refresh_symbol_combination_rankings(symbol, duration, config)
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
        logger.exception("background factor combo refresh failed: %s %s", symbol, duration)
        raise
