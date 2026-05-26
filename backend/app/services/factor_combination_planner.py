from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Any, Callable

import pandas as pd

from app.services.factor_combination_walk_forward import VALIDATION_MIN_PROFIT_FACTOR, VALIDATION_MIN_WIN_RATE

PREFILTER_FACTOR_SCORE_WEIGHT = 1.0
PREFILTER_WIN_RATE_WEIGHT = 35.0
PREFILTER_PROFIT_FACTOR_WEIGHT = 12.0
PREFILTER_SHARPE_WEIGHT = 6.0
PREFILTER_DIVERSITY_WEIGHT = 7.0
PAIRWISE_DIVERSITY_WEIGHT = 9.0
PROFIT_FACTOR_TARGET_SPAN = 0.25
SHARPE_TARGET_SPAN = 2.0
DIVERSITY_RETENTION_LIMIT = 3


@dataclass(frozen=True)
class PlannedCombination:
    members: tuple[Any, ...]
    prefilter_score: float
    pairwise: dict[str, Any]


@dataclass(frozen=True)
class EvaluatedPlan:
    plan: PlannedCombination
    result: dict[str, Any] | None
    failure: dict[str, Any] | None


@dataclass(frozen=True)
class StagedEvaluation:
    rows: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    plan: list[PlannedCombination]
    generated_count: int
    stages: list[dict[str, Any]]


def staged_evaluation(
    context: Any,
    candidates: list[Any],
    config: Any,
    result_func: Callable[[Any, tuple[Any, ...]], tuple[dict[str, Any] | None, dict[str, Any] | None]],
) -> StagedEvaluation:
    requested_sizes = set(config.combo_sizes)
    max_size = max(requested_sizes)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    evaluated_plan: list[PlannedCombination] = []
    generated_count = 0
    stages = []
    previous: list[PlannedCombination] | None = None
    for size in range(2, max_size + 1):
        layer, generated = _build_layer(context.frame, candidates, previous, size, config.beam_width)
        selected = layer[: config.prefilter_limit]
        evaluated = _evaluate_plan_items(context, selected, config, result_func)
        survivors = _surviving_plans(evaluated)
        retained = _retained_diverse_plans(evaluated, survivors)
        rows.extend(_requested_rows(evaluated, size in requested_sizes))
        failures.extend(_evaluated_failures(evaluated))
        evaluated_plan.extend(selected)
        generated_count += generated
        stages.append(_evaluated_stage_payload(size, generated, layer, selected, survivors, retained, config.beam_width))
        previous = [*survivors, *retained]
        if size < max_size and not previous:
            break
    return StagedEvaluation(rows, failures, evaluated_plan, generated_count, stages)


def prefilter_score(members: tuple[Any, ...], pairwise: dict[str, Any]) -> float:
    scores = [_member_prefilter_score(member) for member in members]
    diversity = _diversity_score(members) * PREFILTER_DIVERSITY_WEIGHT
    pairwise_diversity = _pairwise_diversity_score(pairwise) * PAIRWISE_DIVERSITY_WEIGHT
    return sum(scores) / len(scores) + diversity + pairwise_diversity


def pairwise_diversity_payload(frame: pd.DataFrame, members: tuple[Any, ...]) -> dict[str, Any]:
    names = [member.factor.name for member in members if member.factor.name in frame.columns]
    correlations = _pairwise_correlations(frame, names)
    values = [abs(item["correlation"]) for item in correlations]
    avg_abs = sum(values) / len(values) if values else None
    max_abs = max(values) if values else None
    return {
        "pairwiseCorrelations": correlations,
        "pairwiseAvgAbsCorrelation": _round_or_none(avg_abs, 6),
        "pairwiseMaxAbsCorrelation": _round_or_none(max_abs, 6),
        "pairwiseDiversityScore": _round_or_none(_pairwise_diversity_score({"pairwiseMaxAbsCorrelation": max_abs}), 6),
    }


def _build_layer(frame, candidates: list[Any], previous: list[PlannedCombination] | None, size: int, beam_width: int):
    raw_members = combinations(candidates, size) if previous is None else _expanded_members(previous, candidates)
    plans = [_planned_combination(frame, tuple(members)) for members in raw_members]
    plans.sort(key=lambda item: item.prefilter_score, reverse=True)
    return plans[:beam_width], len(plans)


def _planned_combination(frame: pd.DataFrame, members: tuple[Any, ...]) -> PlannedCombination:
    pairwise = pairwise_diversity_payload(frame, members)
    return PlannedCombination(members, prefilter_score(members, pairwise), pairwise)


def _evaluated_stage_payload(size, generated, layer, selected, survivors, retained, beam_width) -> dict[str, Any]:
    return {
        **_stage_payload(size, generated, layer, beam_width),
        "evaluated": len(selected),
        "survivors": len(survivors),
        "retainedForExpansion": len(retained),
        "filter": "walk_forward_passed",
    }


def _stage_payload(size: int, generated: int, layer: list[PlannedCombination], beam_width: int) -> dict[str, Any]:
    return {
        "stage": f"size_{size}",
        "comboSize": size,
        "generated": generated,
        "passed": len(layer),
        "rejected": max(generated - len(layer), 0),
        "beamWidth": beam_width,
    }


def _expanded_members(previous: list[PlannedCombination], candidates: list[Any]) -> list[tuple[Any, ...]]:
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


def _evaluate_plan_items(
    context: Any,
    plan: list[PlannedCombination],
    config: Any,
    result_func: Callable[[Any, tuple[Any, ...]], tuple[dict[str, Any] | None, dict[str, Any] | None]],
) -> list[EvaluatedPlan]:
    results = _parallel_results(context, plan, config.parallel_workers, result_func)
    return [EvaluatedPlan(item, result, failure) for item, (result, failure) in zip(plan, results)]


def _surviving_plans(evaluated: list[EvaluatedPlan]) -> list[PlannedCombination]:
    return [item.plan for item in evaluated if _stage_passed(item.result)]


def _retained_diverse_plans(evaluated: list[EvaluatedPlan], survivors: list[PlannedCombination]) -> list[PlannedCombination]:
    survivor_keys = {_member_key(item.members) for item in survivors}
    failed = [item.plan for item in evaluated if not _stage_passed(item.result)]
    diverse = sorted(failed, key=lambda item: _retention_key(item, survivor_keys), reverse=True)
    return diverse[:DIVERSITY_RETENTION_LIMIT]


def _stage_passed(row: dict[str, Any] | None) -> bool:
    return bool(row is not None and row.get("walkForwardPassed") is True)


def _retention_key(plan: PlannedCombination, survivor_keys: set[tuple[str, ...]]) -> tuple[float, float]:
    novelty = 1.0 if _member_key(plan.members) not in survivor_keys else 0.0
    return (_pairwise_diversity_score(plan.pairwise), novelty)


def _member_key(members: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(sorted(member.factor.name for member in members))


def _requested_rows(evaluated: list[EvaluatedPlan], include: bool) -> list[dict[str, Any]]:
    if not include:
        return []
    return [item.result for item in evaluated if item.result is not None]


def _evaluated_failures(evaluated: list[EvaluatedPlan]) -> list[dict[str, Any]]:
    return [item.failure for item in evaluated if item.failure is not None]


def _parallel_results(context, plan, workers, result_func) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    if workers == 1 or len(plan) <= 1:
        return [result_func(context, item.members) for item in plan]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda item: result_func(context, item.members), plan))


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


def _pairwise_correlations(frame: pd.DataFrame, names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            corr = _spearman_pair(frame[left], frame[right])
            if corr is not None:
                rows.append({"left": left, "right": right, "correlation": round(corr, 6)})
    return rows


def _spearman_pair(left: pd.Series, right: pd.Series) -> float | None:
    valid = pd.concat([left, right], axis=1).replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return None
    corr = valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman")
    return float(corr) if corr is not None and isfinite(float(corr)) else None


def _pairwise_diversity_score(pairwise: dict[str, Any]) -> float:
    value = _finite_float(pairwise.get("pairwiseMaxAbsCorrelation"))
    return 0.0 if value is None else 1.0 - _clamp01(value)


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


def _round_or_none(value: float | None, decimals: int) -> float | None:
    if value is None or not isfinite(float(value)):
        return None
    return round(float(value), decimals)
