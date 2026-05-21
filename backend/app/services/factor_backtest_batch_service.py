from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.factor_backtest_service import run_factor_backtest_on_frame
from app.services.factor_backtest_materialization import (
    materialized_frame_for_factor,
    materialized_frame_for_mined_row,
)
from app.services.factor_catalog import list_single_factor_definitions
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_mined_library import mined_factor_definition, mined_factor_rows_for_duration
from app.services.factor_metric_enrichment import enrich_factor_results
from app.services.factor_registry import FactorDefinition, factor_payload
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

BACKTEST_DURATION_ORDER = ("10m", "30m", "60m", "1d")


@dataclass(frozen=True)
class _BacktestContext:
    frames: dict[str, Any]
    symbol: str


@dataclass(frozen=True)
class _FactorBacktestTask:
    context: _BacktestContext
    factor: FactorDefinition
    duration: str
    row: dict[str, Any] | None = None


def run_all_factor_backtests(
    symbol: str,
    durations: tuple[str, ...] = BACKTEST_DURATION_ORDER,
) -> dict[str, Any]:
    _validate_durations(durations)
    frames = {duration: load_factor_frame(symbol, duration) for duration in durations}
    context = _BacktestContext(frames=frames, symbol=symbol.upper())
    results, failures, factor_count = _backtest_factor_matrix(context, durations)
    enrich_factor_results(results, duration_scoped=True)
    return {
        "symbol": symbol.upper(),
        "durations": list(durations),
        "factorCount": factor_count,
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
    durations: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    factor_count = 0
    for duration in durations:
        tasks = _tasks_for_duration(context, duration)
        factor_count += len(tasks)
        for task in tasks:
            _append_factor_backtest(results, failures, task)
    return results, failures, factor_count


def _tasks_for_duration(context: _BacktestContext, duration: str) -> list[_FactorBacktestTask]:
    singles = [
        _FactorBacktestTask(context=context, factor=factor, duration=duration)
        for factor in list_single_factor_definitions(symbol=context.symbol, duration=duration)
    ]
    combos = [
        _FactorBacktestTask(
            context=context,
            factor=mined_factor_definition(row),
            duration=duration,
            row=row,
        )
        for row in mined_factor_rows_for_duration(context.symbol, duration)
    ]
    return [*singles, *combos]


def _append_factor_backtest(
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    task: _FactorBacktestTask,
) -> None:
    try:
        frame = _materialized_task_frame(task)
        result = run_factor_backtest_on_frame(
            task.factor,
            frame,
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


def _materialized_task_frame(task: _FactorBacktestTask) -> Any:
    if task.row is not None:
        return materialized_frame_for_mined_row(
            task.context.frames[task.duration],
            row=task.row,
            symbol=task.context.symbol,
            duration=task.duration,
            factor_name=task.factor.name,
        )
    return materialized_frame_for_factor(
        task.context.frames[task.duration],
        task.factor,
        task.context.symbol,
        task.duration,
    )
