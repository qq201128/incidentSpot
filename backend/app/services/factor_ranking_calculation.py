from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.services.factor_backtest_materialization import materialized_frame_for_factor
from app.services.factor_catalog import list_single_factor_definitions
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_metric_enrichment import enrich_factor_results
from app.services.factor_registry import FactorDefinition
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

logger = logging.getLogger("uvicorn.error")
FAILURE_SAMPLE_LIMIT = 20


def run_factor_ranking(
    symbol: str,
    duration: str = "10m",
    category: str | None = None,
) -> list[dict[str, Any]]:
    return run_factor_ranking_report(symbol, duration, category)["ranking"]


def run_factor_ranking_report(
    symbol: str,
    duration: str = "10m",
    category: str | None = None,
) -> dict[str, Any]:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    sym = symbol.strip().upper()
    frame = load_factor_frame(sym, duration)
    factors = list_single_factor_definitions(category, symbol=sym, duration=duration)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    working = frame
    for factor_def in factors:
        working = _append_factor_result(results, failures, working, factor_def, sym, duration)
    _log_failures(sym, duration, failures)
    enrich_factor_results(results, frame=working)
    results.sort(key=lambda row: row.get("factorScore") or 0.0, reverse=True)
    return {
        "symbol": sym,
        "duration": duration,
        "category": category,
        "ranking": results,
        "total": len(results),
        "rankingDiagnostics": _ranking_diagnostics(factors, results, failures),
        "rankingFailures": failures,
    }


def _append_factor_result(
    results: list[dict[str, Any]],
    failures: list[dict[str, str]],
    frame: pd.DataFrame,
    factor_def: FactorDefinition,
    symbol: str,
    duration: str,
) -> pd.DataFrame:
    try:
        working = materialized_frame_for_factor(frame, factor_def, symbol, duration)
        if factor_def.name not in working.columns:
            failures.append(_failure(factor_def, "missing_feature_column"))
            return working
        from app.services.factor_backtest_service import run_factor_backtest_on_frame

        results.append(run_factor_backtest_on_frame(factor_def, working, symbol=symbol, duration=duration))
        return working
    except Exception as exc:
        failures.append(_failure(factor_def, str(exc)))
        return frame


def _failure(factor_def: FactorDefinition, error: str) -> dict[str, str]:
    return {
        "factorName": factor_def.name,
        "sourceFile": factor_def.source_file,
        "error": error,
    }


def _log_failures(symbol: str, duration: str, failures: list[dict[str, str]]) -> None:
    if not failures:
        return
    logger.warning(
        "factor ranking skipped factors: symbol=%s duration=%s failureCount=%s failures=%s",
        symbol,
        duration,
        len(failures),
        failures[:FAILURE_SAMPLE_LIMIT],
    )


def _ranking_diagnostics(
    factors: list[FactorDefinition],
    results: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "factorDefinitionCount": len(factors),
        "rankedFactorCount": len(results),
        "failureCount": len(failures),
        "failureSample": failures[:FAILURE_SAMPLE_LIMIT],
    }
