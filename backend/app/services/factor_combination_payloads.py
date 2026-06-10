from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from app.services.agent_mined_factor_library import AGENT_FACTOR_SOURCE_FILE
from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS
from app.services.factor_combo_display import combo_display_name as build_combo_display_name
from app.services.factor_registry import factor_payload

MINED_FACTOR_SOURCE_FILE = "mined_factor_library.json"
REPORT_FAILURE_LIMIT = 50
CORRELATION_DECIMALS = 4


@dataclass(frozen=True)
class CombinationRankingReportPayload:
    symbol: str
    duration: str
    config: Any
    selected: list[Any]
    ranking: list[dict[str, Any]]
    evaluated_ranking: list[dict[str, Any]]
    tested_count: int
    failures: list[dict[str, Any]]
    mined_source_count: int
    search_diagnostics: dict[str, Any]


def build_combination_ranking_report(payload: CombinationRankingReportPayload) -> dict[str, Any]:
    return {
        "symbol": payload.symbol.upper(),
        "duration": payload.duration,
        "ranking": payload.ranking,
        "evaluatedRanking": payload.evaluated_ranking,
        "total": len(payload.ranking),
        "evaluatedTotal": len(payload.evaluated_ranking),
        "searchConfig": config_payload(payload.config),
        "baseFactors": [base_payload(item) for item in payload.selected],
        "baseFactorCount": len(payload.selected),
        "minedFactorSourceCount": payload.mined_source_count,
        "minedFactorUsedCount": mined_factor_count(payload.selected),
        "agentMinedFactorUsedCount": agent_mined_factor_count(payload.selected),
        "testedCombinationCount": payload.tested_count,
        "failureCount": len(payload.failures),
        "failures": payload.failures[:REPORT_FAILURE_LIMIT],
        "searchDiagnostics": payload.search_diagnostics,
    }


def member_payloads(members: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": member.factor.name,
            "displayName": member.factor.description,
            "category": member.factor.category.value,
            "orientation": member.orientation,
            "singleWinRate": member.metrics.get("winRate"),
            "singleIr": member.metrics.get("ir"),
            "singleSharpe": member.metrics.get("sharpe"),
            "singleScore": member.metrics.get("factorScore"),
            "avgAbsCorrelation": member.metrics.get("avgAbsCorrelation"),
        }
        for member in members
    ]


def member_avg_correlation(members: tuple[Any, ...]) -> float | None:
    values = [_finite_float(member.metrics.get("avgAbsCorrelation")) for member in members]
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return round(sum(finite_values) / len(finite_values), CORRELATION_DECIMALS)


def combo_display_name(members: list[dict[str, Any]]) -> str:
    return build_combo_display_name(members)


def base_payload(candidate: Any) -> dict[str, Any]:
    return {
        **factor_payload(candidate.factor),
        "orientation": candidate.orientation,
        "singleWinRate": candidate.metrics.get("winRate"),
        "directionalWinRate": directional_win_rate(candidate),
        "singleIr": candidate.metrics.get("ir"),
        "singleSharpe": candidate.metrics.get("sharpe"),
        "avgAbsCorrelation": candidate.metrics.get("avgAbsCorrelation"),
        "factorScore": candidate.metrics.get("factorScore"),
        "learningWeight": candidate.metrics.get("learningWeight"),
        "learningScore": candidate.metrics.get("learningScore"),
    }


def mined_factor_count(selected: list[Any]) -> int:
    return sum(1 for item in selected if item.factor.source_file == MINED_FACTOR_SOURCE_FILE)


def agent_mined_factor_count(selected: list[Any]) -> int:
    return sum(1 for item in selected if item.factor.source_file == AGENT_FACTOR_SOURCE_FILE)


def config_payload(config: Any) -> dict[str, Any]:
    return {
        "baseFactorLimit": config.base_factor_limit,
        "nativeFactorLimit": config.native_factor_limit,
        "minedFactorLimit": config.mined_factor_limit,
        "agentFactorLimit": config.agent_factor_limit,
        "comboSizes": list(config.combo_sizes),
        "resultLimit": config.result_limit,
        "prefilterLimit": config.prefilter_limit,
        "beamWidth": config.beam_width,
        "parallelWorkers": config.parallel_workers,
        "lookbackDays": config.lookback_days,
        "lookbackBars": config.lookback_bars,
        "method": config.method,
        "minPeriods": BACKTEST_MIN_PERIODS,
    }


def directional_win_rate(candidate: Any) -> float | None:
    value = _finite_float(candidate.metrics.get("winRate"))
    if value is None:
        return None
    return value if int(candidate.orientation) == 1 else round(1.0 - value, 6)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None
