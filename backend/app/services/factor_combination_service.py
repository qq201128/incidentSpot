from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import pandas as pd

from app.services.agent_mined_factor_library import AGENT_FACTOR_SOURCE_FILE
from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS, run_factor_backtest_on_frame
from app.services.factor_candidate_selection import select_base_candidates
from app.services.factor_combination_ranker import (
    COMBINATION_METHOD,
    combo_backtest_frame,
    combination_definition,
    combination_result,
    enriched_combo_result,
    rank_combinations,
)
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_metric_enrichment import (
    factor_avg_abs_correlations,
    factor_score,
)
from app.services.factor_combination_payloads import (
    CombinationRankingReportPayload,
    build_combination_ranking_report,
)
from app.services.factor_learning_controls import (
    learning_blocked_factor_names,
    learning_weight,
    load_factor_learning_memory_for,
)
from app.services.factor_mined_candidates import build_mined_candidates
from app.services.factor_registry import FactorDefinition, FactorDirection, list_factors
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

DEFAULT_BASE_FACTOR_LIMIT = 48
DEFAULT_NATIVE_FACTOR_LIMIT = 32
DEFAULT_MINED_FACTOR_LIMIT = 12
DEFAULT_AGENT_FACTOR_LIMIT = 4
DEFAULT_RESULT_LIMIT = 200
DEFAULT_PREFILTER_LIMIT = 800
DEFAULT_BEAM_WIDTH = 800
DEFAULT_PARALLEL_WORKERS = 4
BASE_DIRECTION_SPLIT = 0.50
MIN_COMBO_SIZE = 2
DEFAULT_MAX_COMBO_SIZE = 3
DEFAULT_COMBO_SIZES = (MIN_COMBO_SIZE, DEFAULT_MAX_COMBO_SIZE)

@dataclass(frozen=True)
class CombinationSearchConfig:
    base_factor_limit: int = DEFAULT_BASE_FACTOR_LIMIT
    native_factor_limit: int = DEFAULT_NATIVE_FACTOR_LIMIT
    mined_factor_limit: int = DEFAULT_MINED_FACTOR_LIMIT
    agent_factor_limit: int = DEFAULT_AGENT_FACTOR_LIMIT
    combo_sizes: tuple[int, ...] = DEFAULT_COMBO_SIZES
    result_limit: int = DEFAULT_RESULT_LIMIT
    prefilter_limit: int = DEFAULT_PREFILTER_LIMIT
    beam_width: int = DEFAULT_BEAM_WIDTH
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS
    method: str = COMBINATION_METHOD

@dataclass(frozen=True)
class _BaseCandidate:
    factor: FactorDefinition
    metrics: dict[str, Any]
    orientation: int

@dataclass(frozen=True)
class _CombinationContext:
    frame: pd.DataFrame
    symbol: str
    duration: str

def run_factor_combination_ranking(
    symbol: str,
    duration: str,
    config: CombinationSearchConfig | None = None,
) -> dict[str, Any]:
    cfg = _validated_config(config or CombinationSearchConfig())
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    frame = load_factor_frame(symbol, duration)
    return run_factor_combination_ranking_on_frame(
        frame,
        symbol=symbol,
        duration=duration,
        config=cfg,
    )

def run_factor_combination_ranking_on_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    duration: str,
    config: CombinationSearchConfig,
) -> dict[str, Any]:
    cfg = _validated_config(config)
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    learning_memory = load_factor_learning_memory_for(symbol, duration)
    blocked_names = learning_blocked_factor_names(learning_memory)
    search_frame = frame.drop(columns=sorted(blocked_names), errors="ignore") if blocked_names else frame
    mined = build_mined_candidates(
        search_frame,
        symbol=symbol.upper(),
        duration=duration,
        excluded_factor_names=blocked_names,
    )
    base, base_failures = _base_candidates(mined.frame, symbol.upper(), duration)
    base.extend(_mined_base_candidates(mined.candidates))
    base = _enriched_base_candidates(base, mined.frame, learning_memory)
    selected = select_base_candidates(base, cfg, rank_key=_base_rank_key)
    context = _CombinationContext(mined.frame, symbol.upper(), duration)
    rank_result = _rank_combinations_with_diagnostics(context, selected, cfg)
    failures = [*base_failures, *mined.failures, *rank_result.failures]
    return build_combination_ranking_report(
        CombinationRankingReportPayload(
            symbol=symbol,
            duration=duration,
            config=cfg,
            selected=selected,
            ranking=rank_result.ranking,
            tested_count=rank_result.tested_count,
            failures=failures,
            mined_source_count=mined.source_count,
            search_diagnostics=rank_result.diagnostics,
        )
    )


def _base_candidates(
    frame: pd.DataFrame,
    symbol: str,
    duration: str,
) -> tuple[list[_BaseCandidate], list[dict[str, Any]]]:
    candidates: list[_BaseCandidate] = []
    failures: list[dict[str, Any]] = []
    for factor in list_factors():
        if factor.name not in frame.columns:
            continue
        try:
            metrics = run_factor_backtest_on_frame(factor, frame, symbol=symbol, duration=duration)
            if _usable_base_metrics(metrics):
                candidates.append(_BaseCandidate(factor, metrics, _factor_orientation(factor, metrics)))
        except Exception as exc:
            failures.append({"factorName": factor.name, "stage": "single_factor", "error": str(exc)})
    return candidates, failures


def _rank_combinations(
    context: _CombinationContext,
    candidates: list[_BaseCandidate],
    config: CombinationSearchConfig,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    result = _rank_combinations_with_diagnostics(context, candidates, config)
    return result.ranking, result.tested_count, result.failures


def _rank_combinations_with_diagnostics(
    context: _CombinationContext,
    candidates: list[_BaseCandidate],
    config: CombinationSearchConfig,
):
    return rank_combinations(context, candidates, config, result_func=_combination_result)


_combination_result = combination_result
_combo_backtest_frame = combo_backtest_frame
_combination_definition = combination_definition
_enriched_combo_result = enriched_combo_result


def _mined_base_candidates(candidates: tuple[Any, ...]) -> list[_BaseCandidate]:
    return [
        _BaseCandidate(item.factor, item.metrics, item.orientation)
        for item in candidates
        if item.factor.source_file == AGENT_FACTOR_SOURCE_FILE
    ]


def _enriched_base_candidates(
    candidates: list[_BaseCandidate],
    frame: pd.DataFrame,
    learning_memory: dict[str, Any] | None,
) -> list[_BaseCandidate]:
    correlations = factor_avg_abs_correlations(frame, [item.factor.name for item in candidates])
    enriched = []
    for candidate in candidates:
        metrics = {**candidate.metrics, "avgAbsCorrelation": correlations.get(candidate.factor.name)}
        metrics["factorScore"] = factor_score(metrics)
        weight = learning_weight(learning_memory, candidate.factor.name)
        metrics["learningWeight"] = round(weight, 6)
        metrics["learningScore"] = round(metrics["factorScore"] + weight, 6)
        enriched.append(_BaseCandidate(candidate.factor, metrics, candidate.orientation))
    return enriched


def _factor_orientation(factor: FactorDefinition, metrics: dict[str, Any]) -> int:
    if factor.direction == FactorDirection.LOWER_BETTER:
        return -1
    if factor.direction == FactorDirection.HIGHER_BETTER:
        return 1
    win_rate = _finite_float(metrics.get("winRate"))
    if win_rate is not None and win_rate < BASE_DIRECTION_SPLIT:
        return -1
    if win_rate is not None and win_rate > BASE_DIRECTION_SPLIT:
        return 1
    return 1 if _metric_sign(metrics) >= 0 else -1


def _metric_sign(metrics: dict[str, Any]) -> float:
    for key in ("icMean", "longShortReturn", "ir"):
        value = _finite_float(metrics.get(key))
        if value is not None and value != 0:
            return value
    return 1.0


def _usable_base_metrics(metrics: dict[str, Any]) -> bool:
    has_periods = int(metrics.get("totalPeriods") or 0) >= BACKTEST_MIN_PERIODS
    return has_periods and metrics.get("winRate") is not None


def _base_rank_key(candidate: _BaseCandidate) -> tuple[float, float, float, float, int]:
    row = candidate.metrics
    return (
        _directional_win_rate(candidate),
        _num(row.get("learningScore") if row.get("learningScore") is not None else row.get("factorScore")),
        _num(row.get("factorScore")),
        _num(row.get("winRate")),
        int(row.get("totalPeriods") or 0),
    )


def _directional_win_rate(candidate: _BaseCandidate) -> float:
    win_rate = _finite_float(candidate.metrics.get("winRate"))
    if win_rate is None:
        return float("-inf")
    return win_rate if candidate.orientation == 1 else 1.0 - win_rate


def _validated_config(config: CombinationSearchConfig) -> CombinationSearchConfig:
    sizes = tuple(sorted(set(int(size) for size in config.combo_sizes)))
    if config.method != COMBINATION_METHOD:
        raise ValueError(f"unsupported combination method: {config.method}")
    if not sizes or config.base_factor_limit < max(sizes):
        raise ValueError("combo sizes must be non-empty and base_factor_limit must be >= largest combo size")
    if config.result_limit <= 0 or any(size < MIN_COMBO_SIZE for size in sizes):
        raise ValueError("result_limit must be > 0 and combo sizes must be >= 2")
    if min(config.native_factor_limit, config.mined_factor_limit, config.agent_factor_limit) < 0:
        raise ValueError("factor source limits must be >= 0")
    if min(config.prefilter_limit, config.beam_width, config.parallel_workers) <= 0:
        raise ValueError("prefilter_limit, beam_width, and parallel_workers must be > 0")
    return CombinationSearchConfig(
        base_factor_limit=int(config.base_factor_limit),
        native_factor_limit=int(config.native_factor_limit),
        mined_factor_limit=int(config.mined_factor_limit),
        agent_factor_limit=int(config.agent_factor_limit),
        combo_sizes=sizes,
        result_limit=int(config.result_limit),
        prefilter_limit=int(config.prefilter_limit),
        beam_width=int(config.beam_width),
        parallel_workers=int(config.parallel_workers),
        method=config.method,
    )


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None


def _num(value: Any) -> float:
    number = _finite_float(value)
    return number if number is not None else float("-inf")
