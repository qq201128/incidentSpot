from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Any, Callable

import pandas as pd

from app.services.factor_backtest_service import run_factor_backtest_on_frame
from app.services.factor_combination_payloads import combo_display_name, member_avg_correlation, member_payloads
from app.services.factor_combination_search_diagnostics import combination_search_diagnostics
from app.services.factor_combination_walk_forward import (
    VALIDATION_MIN_PROFIT_FACTOR,
    VALIDATION_MIN_WIN_RATE,
    walk_forward_validation,
)
from app.services.factor_combo_scoring import combination_score
from app.services.factor_metric_enrichment import enrich_factor_results, factor_score
from app.services.factor_registry import FactorCategory, FactorDefinition, FactorDirection

COMBINATION_METHOD = "expanding_oriented_zscore_mean_v1"
COMBO_SOURCE_FILE = "factor_combination_service.py"
PREFILTER_FACTOR_SCORE_WEIGHT = 1.0
PREFILTER_WIN_RATE_WEIGHT = 35.0
PREFILTER_PROFIT_FACTOR_WEIGHT = 12.0
PREFILTER_SHARPE_WEIGHT = 6.0
PREFILTER_DIVERSITY_WEIGHT = 7.0
PROFIT_FACTOR_TARGET_SPAN = 0.25
SHARPE_TARGET_SPAN = 2.0


@dataclass(frozen=True)
class CombinationRankResult:
    ranking: list[dict[str, Any]]
    tested_count: int
    failures: list[dict[str, Any]]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _PlannedCombination:
    members: tuple[Any, ...]
    prefilter_score: float


def rank_combinations(
    context: Any,
    candidates: list[Any],
    config: Any,
    *,
    result_func: Callable[[Any, tuple[Any, ...]], tuple[dict[str, Any] | None, dict[str, Any] | None]] | None = None,
) -> CombinationRankResult:
    plan, generated_count = _combination_plan(candidates, config)
    rows, failures = _evaluate_plan(context, plan, config, result_func or combination_result)
    enrich_factor_results(rows)
    failures.extend(_walk_forward_failures(rows))
    ranking = [row for row in rows if row["walkForwardPassed"]]
    ranking.sort(key=_combo_rank_key, reverse=True)
    if rows and not ranking:
        failures.append({"stage": "walk_forward", "error": "no_walk_forward_combo_passed"})
    diagnostics = combination_search_diagnostics(config, candidates, plan, rows, failures, generated_count)
    return CombinationRankResult(ranking[: config.result_limit], len(plan), failures, diagnostics)


def combination_result(
    context: Any,
    members: tuple[Any, ...],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    factor_def = combination_definition(members, context.duration)
    try:
        combo_frame = combo_backtest_frame(context.frame, members, factor_def.name)
        result = run_factor_backtest_on_frame(
            factor_def,
            combo_frame,
            symbol=context.symbol,
            duration=context.duration,
        )
        return enriched_combo_result(result, members, combo_frame, factor_def), None
    except Exception as exc:
        failure = {"factorName": factor_def.name, "stage": "combination", "error": str(exc)}
        return None, failure


def combination_definition(
    members: tuple[Any, ...],
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


def combo_backtest_frame(
    frame: pd.DataFrame,
    members: tuple[Any, ...],
    combo_name: str,
) -> pd.DataFrame:
    out = frame[["close"]].copy()
    if "open_time" in frame.columns:
        out["open_time"] = frame["open_time"]
    out[combo_name] = combination_score(frame, member_payloads(members))
    return out


def enriched_combo_result(
    result: dict[str, Any],
    members: tuple[Any, ...],
    combo_frame: pd.DataFrame,
    factor_def: FactorDefinition,
) -> dict[str, Any]:
    member_rows = member_payloads(members)
    walk_forward = walk_forward_validation(combo_frame, factor_def, factor_def.timeframes[0])
    payload = _combo_payload(result, members, member_rows, walk_forward)
    payload["factorDisplayName"] = combo_display_name(member_rows)
    payload["description"] = payload["factorDisplayName"]
    payload["factorScore"] = factor_score(payload)
    return payload


def _combination_plan(candidates: list[Any], config: Any) -> tuple[list[_PlannedCombination], int]:
    requested_sizes = set(config.combo_sizes)
    max_size = max(requested_sizes)
    layers: dict[int, list[_PlannedCombination]] = {}
    selected: list[_PlannedCombination] = []
    generated_count = 0
    for size in range(2, max_size + 1):
        layer, generated = _build_layer(candidates, layers.get(size - 1), size, config.beam_width)
        generated_count += generated
        layers[size] = layer[: config.beam_width]
        if size in requested_sizes:
            selected.extend(layer)
    selected.sort(key=lambda item: item.prefilter_score, reverse=True)
    return selected[: config.prefilter_limit], generated_count


def _build_layer(
    candidates: list[Any],
    previous: list[_PlannedCombination] | None,
    size: int,
    beam_width: int,
) -> tuple[list[_PlannedCombination], int]:
    raw_members = combinations(candidates, size) if previous is None else _expanded_members(previous, candidates)
    plans = [_PlannedCombination(tuple(members), _prefilter_score(tuple(members))) for members in raw_members]
    plans.sort(key=lambda item: item.prefilter_score, reverse=True)
    return plans[:beam_width], len(plans)


def _expanded_members(previous: list[_PlannedCombination], candidates: list[Any]) -> list[tuple[Any, ...]]:
    seen: set[tuple[str, ...]] = set()
    expanded: list[tuple[Any, ...]] = []
    by_name = {candidate.factor.name: candidate for candidate in candidates}
    for item in previous:
        current_names = {member.factor.name for member in item.members}
        for candidate in candidates:
            if candidate.factor.name in current_names:
                continue
            names = tuple(sorted([*current_names, candidate.factor.name]))
            if names in seen:
                continue
            seen.add(names)
            expanded.append(tuple(by_name[name] for name in names))
    return expanded


def _evaluate_plan(
    context: Any,
    plan: list[_PlannedCombination],
    config: Any,
    result_func: Callable[[Any, tuple[Any, ...]], tuple[dict[str, Any] | None, dict[str, Any] | None]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = _parallel_results(context, plan, config.parallel_workers, result_func)
    rows = [result for result, _failure in results if result is not None]
    failures = [failure for _result, failure in results if failure is not None]
    return rows, failures


def _parallel_results(
    context: Any,
    plan: list[_PlannedCombination],
    workers: int,
    result_func: Callable[[Any, tuple[Any, ...]], tuple[dict[str, Any] | None, dict[str, Any] | None]],
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    if workers == 1 or len(plan) <= 1:
        return [result_func(context, item.members) for item in plan]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda item: result_func(context, item.members), plan))


def _combo_payload(result: dict[str, Any], members: tuple[Any, ...], member_rows: list[dict], walk_forward: Any) -> dict:
    return {
        **result,
        "comboSize": len(members),
        "method": COMBINATION_METHOD,
        "members": member_rows,
        "avgAbsCorrelation": member_avg_correlation(members),
        "prefilterScore": round(_prefilter_score(members), 6),
        "walkForward": walk_forward.payload,
        "walkForwardPassed": walk_forward.passed,
        "walkForwardFailureReason": walk_forward.failure_reason,
    }


def _prefilter_score(members: tuple[Any, ...]) -> float:
    scores = [_member_prefilter_score(member) for member in members]
    diversity = _diversity_score(members) * PREFILTER_DIVERSITY_WEIGHT
    return sum(scores) / len(scores) + diversity


def _member_prefilter_score(member: Any) -> float:
    row = member.metrics
    return (
        _num(row.get("factorScore")) * PREFILTER_FACTOR_SCORE_WEIGHT
        + _target_ratio(_directional_win_rate(member), VALIDATION_MIN_WIN_RATE, 1.0) * PREFILTER_WIN_RATE_WEIGHT
        + _target_ratio(row.get("profitFactor"), VALIDATION_MIN_PROFIT_FACTOR, PROFIT_FACTOR_TARGET_SPAN)
        * PREFILTER_PROFIT_FACTOR_WEIGHT
        + _target_ratio(row.get("sharpe"), 0.0, SHARPE_TARGET_SPAN) * PREFILTER_SHARPE_WEIGHT
    )


def _directional_win_rate(member: Any) -> float | None:
    win_rate = _finite_float(member.metrics.get("winRate"))
    if win_rate is None:
        return None
    return win_rate if int(member.orientation) == 1 else 1.0 - win_rate


def _diversity_score(members: tuple[Any, ...]) -> float:
    values = [_finite_float(member.metrics.get("avgAbsCorrelation")) for member in members]
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return 0.0
    return 1.0 - _clamp01(sum(finite_values) / len(finite_values))


def _walk_forward_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "factorName": row["factorName"],
            "stage": "walk_forward",
            "error": row["walkForwardFailureReason"],
        }
        for row in rows
        if not row["walkForwardPassed"]
    ]


def _combo_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        _num(row.get("factorScore")),
        _num(row.get("winRate")),
        _num(row.get("profitFactor")),
        _num(row.get("sharpe")),
    )


def _target_ratio(value: Any, target: float, span: float) -> float:
    number = _finite_float(value)
    if number is None:
        return 0.0
    return _clamp01((number - target) / span)


def _num(value: Any) -> float:
    number = _finite_float(value)
    return number if number is not None else float("-inf")


def _clamp01(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None
