from __future__ import annotations

import logging

from fastapi import HTTPException

from app.services.background_loop_status import record_loop_failure, record_loop_success
from app.services.experiment_profiles import combination_search_config_for_profile
from app.services.factor_combination_background import refresh_symbol_combination_rankings
from app.services.factor_combination_service import MIN_COMBO_SIZE, CombinationSearchConfig

BACKGROUND_REFRESH_LOOP = "factor_combo_daily"
logger = logging.getLogger("uvicorn.error")


def combination_config(
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


def config_response(config: CombinationSearchConfig) -> dict:
    return {
        "baseFactorLimit": config.base_factor_limit,
        "comboSizes": list(config.combo_sizes),
        "resultLimit": config.result_limit,
        "prefilterLimit": config.prefilter_limit,
        "beamWidth": config.beam_width,
        "parallelWorkers": config.parallel_workers,
        "method": config.method,
    }


def background_refresh_combo_rankings(
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
