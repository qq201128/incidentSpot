from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable

import pandas as pd

from app.services.factor_backtest_service import run_factor_backtest_on_frame
from app.services.factor_combination_planner import pairwise_diversity_payload, prefilter_score, staged_evaluation
from app.services.factor_combination_payloads import combo_display_name, member_avg_correlation, member_payloads
from app.services.factor_combination_search_diagnostics import combination_search_diagnostics
from app.services.factor_combination_walk_forward import walk_forward_validation
from app.services.factor_combo_scoring import combination_score
from app.services.factor_metric_enrichment import enrich_factor_results, factor_score
from app.services.factor_registry import FactorCategory, FactorDefinition, FactorDirection

COMBINATION_METHOD = "expanding_oriented_zscore_mean_v1"
COMBO_SOURCE_FILE = "factor_combination_service.py"


@dataclass(frozen=True)
class CombinationRankResult:
    ranking: list[dict[str, Any]]
    tested_count: int
    failures: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def rank_combinations(
    context: Any,
    candidates: list[Any],
    config: Any,
    *,
    result_func: Callable[[Any, tuple[Any, ...]], tuple[dict[str, Any] | None, dict[str, Any] | None]] | None = None,
) -> CombinationRankResult:
    evaluation = staged_evaluation(context, candidates, config, result_func or combination_result)
    rows = evaluation.rows
    failures = evaluation.failures
    enrich_factor_results(rows)
    failures.extend(_walk_forward_failures(rows))
    ranking = [row for row in rows if row["walkForwardPassed"]]
    ranking.sort(key=_combo_rank_key, reverse=True)
    if rows and not ranking:
        failures.append({"stage": "walk_forward", "error": "no_walk_forward_combo_passed"})
    diagnostics = combination_search_diagnostics(
        config, candidates, evaluation.plan, rows, failures, evaluation.generated_count, evaluation.stages
    )
    return CombinationRankResult(ranking[: config.result_limit], len(evaluation.plan), failures, diagnostics)


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
        return enriched_combo_result(result, members, context.frame, factor_def), None
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
        description=combo_display_name(member_payloads(members)),
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
    source_frame: pd.DataFrame,
    factor_def: FactorDefinition,
) -> dict[str, Any]:
    member_rows = member_payloads(members)
    combo_frame = combo_backtest_frame(source_frame, members, factor_def.name)
    walk_forward = walk_forward_validation(combo_frame, factor_def, factor_def.timeframes[0])
    payload = _combo_payload(result, members, member_rows, walk_forward)
    pairwise = pairwise_diversity_payload(source_frame, members)
    payload.update(pairwise)
    payload["prefilterScore"] = round(prefilter_score(members, pairwise), 6)
    payload["factorDisplayName"] = combo_display_name(member_rows)
    payload["description"] = payload["factorDisplayName"]
    payload["factorScore"] = factor_score(payload)
    return payload


def _combo_payload(result: dict[str, Any], members: tuple[Any, ...], member_rows: list[dict], walk_forward: Any) -> dict:
    return {
        **result,
        "comboSize": len(members),
        "method": COMBINATION_METHOD,
        "members": member_rows,
        "avgAbsCorrelation": member_avg_correlation(members),
        "prefilterScore": round(prefilter_score(members, {}), 6),
        "walkForward": walk_forward.payload,
        "walkForwardPassed": walk_forward.passed,
        "walkForwardFailureReason": walk_forward.failure_reason,
    }
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


def _num(value: Any) -> float:
    number = _finite_float(value)
    return number if number is not None else float("-inf")
def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None
