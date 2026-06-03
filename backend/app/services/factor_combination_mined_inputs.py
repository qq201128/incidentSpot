from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.agent_mined_factor_library import (
    agent_factor_rows_for_duration,
    build_agent_mined_candidates_from_rows,
    materialize_agent_factor_frame_for_rows,
)
from app.services.agent_mined_factor_library_helpers import row_ingested
from app.services.factor_metric_enrichment import factor_score
from app.services.factor_mined_candidates import MinedCandidateResult


def build_mined_candidates(
    frame: pd.DataFrame,
    *,
    symbol: str,
    duration: str,
    agent_factor_limit: int | None = None,
    excluded_factor_names: set[str] | None = None,
) -> MinedCandidateResult:
    rows = _selected_agent_rows(symbol, duration, agent_factor_limit, excluded_factor_names)
    source_count = _agent_source_count(symbol, duration)
    agent = materialize_agent_factor_frame_for_rows(
        frame,
        rows=rows,
        source_count=source_count,
        excluded_factor_names=excluded_factor_names,
    )
    candidates = build_agent_mined_candidates_from_rows(
        agent.frame,
        symbol=symbol,
        duration=duration,
        rows=rows,
        excluded_factor_names=excluded_factor_names,
    )
    return MinedCandidateResult(agent.frame, candidates, agent.source_count, agent.failures)


def _selected_agent_rows(
    symbol: str,
    duration: str,
    limit: int | None,
    excluded_factor_names: set[str] | None,
) -> list[dict[str, Any]]:
    excluded = excluded_factor_names or set()
    rows = [
        row
        for row in agent_factor_rows_for_duration(symbol, duration)
        if row_ingested(row) and str(row.get("factorName")) not in excluded
    ]
    if limit is None:
        return rows
    return sorted(rows, key=_agent_row_rank_key, reverse=True)[:limit]


def _agent_source_count(symbol: str, duration: str) -> int:
    return sum(1 for row in agent_factor_rows_for_duration(symbol, duration) if row_ingested(row))


def _agent_row_rank_key(row: dict[str, Any]) -> tuple[float, str]:
    return (_agent_row_score(row), str(row.get("factorName") or ""))


def _agent_row_score(row: dict[str, Any]) -> float:
    value = row.get("score")
    if value is not None:
        return float(value)
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        return factor_score(metrics)
    return 0.0
