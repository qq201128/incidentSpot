from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Any

import pandas as pd

from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS, run_factor_backtest_on_frame
from app.services.factor_candidate_selection import select_base_candidates
from app.services.factor_combo_scoring import combination_score
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_metric_enrichment import (
    enrich_factor_results,
    factor_avg_abs_correlations,
    factor_score,
)
from app.services.factor_combination_payloads import (
    CombinationRankingReportPayload,
    build_combination_ranking_report,
    combo_display_name,
    member_avg_correlation,
    member_payloads,
)
from app.services.factor_mined_candidates import build_mined_candidates
from app.services.factor_registry import FactorCategory, FactorDefinition, FactorDirection, list_factors
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

COMBINATION_METHOD = "expanding_oriented_zscore_mean_v1"
COMBO_SOURCE_FILE = "factor_combination_service.py"
DEFAULT_BASE_FACTOR_LIMIT = 16
DEFAULT_NATIVE_FACTOR_LIMIT = 10
DEFAULT_MINED_FACTOR_LIMIT = 4
DEFAULT_AGENT_FACTOR_LIMIT = 2
DEFAULT_RESULT_LIMIT = 200
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
    mined = build_mined_candidates(frame, symbol=symbol.upper(), duration=duration)
    base, base_failures = _base_candidates(mined.frame, symbol.upper(), duration)
    base.extend(_mined_base_candidates(mined.candidates))
    base = _enriched_base_candidates(base, mined.frame)
    selected = select_base_candidates(base, cfg, rank_key=_base_rank_key)
    context = _CombinationContext(mined.frame, symbol.upper(), duration)
    ranking, tested_count, combo_failures = _rank_combinations(context, selected, cfg)
    failures = [*base_failures, *mined.failures, *combo_failures]
    return build_combination_ranking_report(
        CombinationRankingReportPayload(
            symbol=symbol,
            duration=duration,
            config=cfg,
            selected=selected,
            ranking=ranking,
            tested_count=tested_count,
            failures=failures,
            mined_source_count=mined.source_count,
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
    ranking: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for size in config.combo_sizes:
        for members in combinations(candidates, size):
            result, failure = _combination_result(context, members)
            if result is not None:
                ranking.append(result)
            if failure is not None:
                failures.append(failure)
    enrich_factor_results(ranking)
    ranking.sort(key=_combo_rank_key, reverse=True)
    tested_count = len(ranking)
    return ranking[: config.result_limit], tested_count, failures


def _combination_result(
    context: _CombinationContext,
    members: tuple[_BaseCandidate, ...],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    factor_def = _combination_definition(members, context.duration)
    try:
        combo_frame = _combo_backtest_frame(context.frame, members, factor_def.name)
        result = run_factor_backtest_on_frame(
            factor_def,
            combo_frame,
            symbol=context.symbol,
            duration=context.duration,
        )
        return _enriched_combo_result(result, members), None
    except Exception as exc:
        failure = {"factorName": factor_def.name, "stage": "combination", "error": str(exc)}
        return None, failure


def _combo_backtest_frame(
    frame: pd.DataFrame,
    members: tuple[_BaseCandidate, ...],
    combo_name: str,
) -> pd.DataFrame:
    out = frame[["close"]].copy()
    if "open_time" in frame.columns:
        out["open_time"] = frame["open_time"]
    out[combo_name] = combination_score(frame, member_payloads(members))
    return out


def _combination_definition(
    members: tuple[_BaseCandidate, ...],
    duration: str,
) -> FactorDefinition:
    names = [member.factor.name for member in members]
    return FactorDefinition(
        name="combo__" + "__".join(names),
        category=FactorCategory.PERFORMANCE,
        description="组合因子：" + " + ".join(member.factor.description for member in members),
        formula=f"{COMBINATION_METHOD}(" + ", ".join(names) + ")",
        source_file=COMBO_SOURCE_FILE,
        timeframes=(duration,),
        direction=FactorDirection.HIGHER_BETTER,
        parameters={"members": names, "method": COMBINATION_METHOD},
    )


def _enriched_combo_result(result: dict[str, Any], members: tuple[_BaseCandidate, ...]) -> dict[str, Any]:
    member_rows = member_payloads(members)
    payload = {
        **result,
        "comboSize": len(members),
        "method": COMBINATION_METHOD,
        "members": member_rows,
        "avgAbsCorrelation": member_avg_correlation(members),
    }
    payload["factorDisplayName"] = combo_display_name(member_rows)
    payload["description"] = payload["factorDisplayName"]
    payload["factorScore"] = factor_score(payload)
    return payload


def _mined_base_candidates(candidates: tuple[Any, ...]) -> list[_BaseCandidate]:
    return [_BaseCandidate(item.factor, item.metrics, item.orientation) for item in candidates]


def _enriched_base_candidates(
    candidates: list[_BaseCandidate],
    frame: pd.DataFrame,
) -> list[_BaseCandidate]:
    correlations = factor_avg_abs_correlations(frame, [item.factor.name for item in candidates])
    enriched = []
    for candidate in candidates:
        metrics = {**candidate.metrics, "avgAbsCorrelation": correlations.get(candidate.factor.name)}
        metrics["factorScore"] = factor_score(metrics)
        enriched.append(_BaseCandidate(candidate.factor, metrics, candidate.orientation))
    return enriched


def _factor_orientation(factor: FactorDefinition, metrics: dict[str, Any]) -> int:
    if factor.direction == FactorDirection.LOWER_BETTER:
        return -1
    if factor.direction == FactorDirection.HIGHER_BETTER:
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


def _base_rank_key(candidate: _BaseCandidate) -> tuple[float, float, float, int]:
    row = candidate.metrics
    return (
        _num(row.get("factorScore")),
        _num(row.get("winRate")),
        _num(row.get("sharpe")),
        int(row.get("totalPeriods") or 0),
    )


def _combo_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        _num(row.get("factorScore")),
        _num(row.get("winRate")),
        _num(row.get("profitFactor")),
        _num(row.get("sharpe")),
    )


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
    return CombinationSearchConfig(
        base_factor_limit=int(config.base_factor_limit),
        native_factor_limit=int(config.native_factor_limit),
        mined_factor_limit=int(config.mined_factor_limit),
        agent_factor_limit=int(config.agent_factor_limit),
        combo_sizes=sizes,
        result_limit=int(config.result_limit),
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
