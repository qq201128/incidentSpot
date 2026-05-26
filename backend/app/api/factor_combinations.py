from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

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


@router.get("/ranking")
def factor_combination_ranking(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
) -> dict:
    _validate_duration(duration)
    sym_u = symbol.upper()
    cached = get_cached_combination_ranking(sym_u, duration)
    high_winrate_view = _high_winrate_view(sym_u, duration)
    if cached is None:
        return {**_empty_combination_ranking(sym_u, duration), **high_winrate_view}
    if not cache_is_usable(cached):
        return {**_stale_combination_ranking(sym_u, duration, cached), **high_winrate_view}
    ranking = regular_ranking_view(_regular_ranking_rows(cached))
    visibility = _ranking_visibility(cached, ranking)
    return {
        **cached,
        "ranking": ranking,
        "regularRanking": ranking,
        "total": len(ranking),
        "source": "cache",
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
            limit=limit,
            top_per_duration=top_per_duration,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/paper-live/candidates")
def paper_live_candidates(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
) -> dict:
    _validate_duration(duration)
    return paper_live_candidate_report(symbol.upper(), duration)


@router.post("/paper-live/daily-loop")
def paper_live_daily_loop(
    symbol: str | None = Query(None, min_length=6),
    duration: str | None = Query(None),
) -> dict:
    _validate_optional_duration(duration)
    symbols = [symbol.upper()] if symbol else None
    durations = [duration] if duration else None
    return run_paper_live_daily_closed_loop(symbols=symbols, durations=durations)


@router.post("/refresh")
def factor_combination_refresh(
    background_tasks: BackgroundTasks,
    symbol: str = Query(..., min_length=6),
    duration: str | None = Query(None, description="omit to refresh all supported durations"),
    profile: str = Query("full"),
    base_factor_limit: int | None = Query(None, alias="baseFactorLimit"),
    combo_sizes: str | None = Query(None, alias="comboSizes"),
    result_limit: int | None = Query(None, alias="resultLimit"),
) -> dict:
    _validate_optional_duration(duration)
    sym_u = symbol.upper()
    config = _combination_config(profile, base_factor_limit, combo_sizes, result_limit)
    background_tasks.add_task(_background_refresh_combo_rankings, sym_u, duration, config)
    return {
        "ok": True,
        "symbol": sym_u,
        "duration": duration,
        "profile": normalize_experiment_profile(profile),
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
    ranking = regular_ranking_view(_regular_ranking_rows(cached))
    visibility = _ranking_visibility(cached, ranking)
    return {
        "symbol": symbol,
        "duration": duration,
        "ranking": ranking,
        "regularRanking": ranking,
        "total": len(ranking),
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


def _has_combo_member(row: dict) -> bool:
    members = row.get("members")
    if not isinstance(members, list):
        return False
    return any(_is_combo_name(member.get("name")) for member in members if isinstance(member, dict))


def _is_combo_name(name: object) -> bool:
    raw = str(name or "")
    return raw.startswith("combo__") or raw.startswith("goal_combo__")


def _combination_config(
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
    except Exception:
        logger.exception("background factor combo refresh failed: %s %s", symbol, duration)
