from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.factor_backtest_service import run_factor_backtest_on_frame
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_metric_enrichment import enrich_factor_results
from app.services.factor_registry import FactorDefinition, factor_payload, list_factors
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

BACKTEST_DURATION_ORDER = ("10m", "30m", "60m", "1d")


@dataclass(frozen=True)
class _BacktestContext:
    frame: Any
    symbol: str


@dataclass(frozen=True)
class _FactorBacktestTask:
    context: _BacktestContext
    factor: FactorDefinition
    duration: str


def run_all_factor_backtests(
    symbol: str,
    durations: tuple[str, ...] = BACKTEST_DURATION_ORDER,
) -> dict[str, Any]:
    _validate_durations(durations)
    context = _BacktestContext(frame=load_factor_frame(symbol), symbol=symbol.upper())
    factors = list_factors()
    results, failures = _backtest_factor_matrix(context, factors, durations)
    enrich_factor_results(results, duration_scoped=True)
    return {
        "symbol": symbol.upper(),
        "durations": list(durations),
        "factorCount": len(factors),
        "testedCount": len(results),
        "failureCount": len(failures),
        "results": results,
        "failures": failures,
    }


def _validate_durations(durations: tuple[str, ...]) -> None:
    unsupported = [duration for duration in durations if duration not in SUPPORTED_RULE_DURATIONS]
    if unsupported:
        raise ValueError(f"unsupported durations: {', '.join(unsupported)}")


def _backtest_factor_matrix(
    context: _BacktestContext,
    factors: list[FactorDefinition],
    durations: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for duration in durations:
        for factor in factors:
            task = _FactorBacktestTask(context=context, factor=factor, duration=duration)
            _append_factor_backtest(results, failures, task)
    return results, failures


def _append_factor_backtest(
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    task: _FactorBacktestTask,
) -> None:
    try:
        result = run_factor_backtest_on_frame(
            task.factor,
            task.context.frame,
            symbol=task.context.symbol,
            duration=task.duration,
        )
        results.append(result)
    except Exception as exc:
        failures.append({
            "factor": factor_payload(task.factor),
            "duration": task.duration,
            "error": str(exc),
        })
