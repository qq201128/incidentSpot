from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.services.agent_mined_factor_library import (
    AGENT_FACTOR_SOURCE_FILE,
    build_agent_mined_candidates_from_frame,
    materialize_agent_factor_frame,
)
from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS, run_factor_backtest_on_frame
from app.services.factor_combo_scoring import combination_score
from app.services.factor_mined_library import mined_factor_rows_for_duration
from app.services.factor_registry import FactorCategory, FactorDefinition, FactorDirection

MINED_FACTOR_SOURCE_FILE = "mined_factor_library.json"


@dataclass(frozen=True)
class MinedCandidate:
    factor: FactorDefinition
    metrics: dict[str, Any]
    orientation: int


@dataclass(frozen=True)
class MinedFrameResult:
    frame: pd.DataFrame
    source_count: int
    failures: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class MinedCandidateResult:
    frame: pd.DataFrame
    candidates: tuple[MinedCandidate, ...]
    source_count: int
    failures: tuple[dict[str, Any], ...]


def materialize_mined_factor_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    duration: str,
) -> MinedFrameResult:
    agent = materialize_agent_factor_frame(frame, symbol=symbol, duration=duration)
    rows = mined_factor_rows_for_duration(symbol, duration)
    return _materialize_mined_rows(
        agent.frame,
        rows,
        agent.source_count,
        agent.failures,
    )


def materialize_mined_factor_frame_for_rows(
    frame: pd.DataFrame,
    *,
    symbol: str,
    duration: str,
    target_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]] | None = None,
) -> MinedFrameResult:
    agent = materialize_agent_factor_frame(frame, symbol=symbol, duration=duration)
    rows = source_rows if source_rows is not None else mined_factor_rows_for_duration(symbol, duration)
    selected = _dependency_rows(target_rows, rows)
    return _materialize_mined_rows(
        agent.frame,
        selected,
        agent.source_count,
        agent.failures,
        source_count=len(rows),
    )


def materialize_mined_factor_frame_for_targets(
    frame: pd.DataFrame,
    *,
    symbol: str,
    duration: str,
    target_rows: list[dict[str, Any]],
) -> MinedFrameResult:
    source_rows = mined_factor_rows_for_duration(symbol, duration)
    selected = _target_and_dependency_rows(target_rows, source_rows)
    agent = materialize_agent_factor_frame(frame, symbol=symbol, duration=duration)
    return _materialize_mined_rows(
        agent.frame,
        selected,
        agent.source_count,
        agent.failures,
        source_count=len(source_rows),
    )


def _materialize_mined_rows(
    frame: pd.DataFrame,
    rows: list[dict[str, Any]],
    agent_count: int,
    agent_failures: tuple[dict[str, Any], ...],
    *,
    source_count: int | None = None,
) -> MinedFrameResult:
    if not rows:
        return MinedFrameResult(frame, agent_count, agent_failures)
    failures: list[dict[str, Any]] = list(agent_failures)
    pending: dict[str, pd.Series] = {}
    materialized = {str(column) for column in frame.columns}
    by_name = {str(row.get("factorName")): row for row in rows}
    for row in rows:
        try:
            _ensure_materialized(frame, pending, row, by_name, materialized, set())
        except Exception as exc:
            failures.append(_failure(row, "materialize_mined_factor", exc))
    working = _frame_with_pending_columns(frame, pending)
    count = len(rows) if source_count is None else source_count
    return MinedFrameResult(working, count + agent_count, tuple(failures))


def _dependency_rows(target_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {str(row.get("factorName")): row for row in source_rows}
    selected: dict[str, dict[str, Any]] = {}
    for row in target_rows:
        _collect_dependencies(row, by_name, selected, set())
    return list(selected.values())


def _target_and_dependency_rows(
    target_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = {str(row.get("factorName")): row for row in _dependency_rows(target_rows, source_rows)}
    for row in target_rows:
        selected[str(row.get("factorName"))] = row
    return list(selected.values())


def _collect_dependencies(
    row: dict[str, Any],
    by_name: dict[str, dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    visiting: set[str],
) -> None:
    for member in _members(row):
        name = str(member["name"])
        dependency = by_name.get(name)
        if dependency is None or name in selected:
            continue
        if name in visiting:
            raise ValueError(f"cycle in mined factor library: {name}")
        visiting.add(name)
        _collect_dependencies(dependency, by_name, selected, visiting)
        visiting.remove(name)
        selected[name] = dependency


def build_mined_candidates(
    frame: pd.DataFrame,
    *,
    symbol: str,
    duration: str,
) -> MinedCandidateResult:
    materialized = materialize_mined_factor_frame(frame, symbol=symbol, duration=duration)
    candidates: list[MinedCandidate] = []
    failures = list(materialized.failures)
    for row in mined_factor_rows_for_duration(symbol, duration):
        if str(row.get("factorName")) not in materialized.frame.columns:
            continue
        try:
            candidate = _candidate_from_row(materialized.frame, row, symbol, duration)
            if _usable_metrics(candidate.metrics):
                candidates.append(candidate)
        except Exception as exc:
            failures.append(_failure(row, "backtest_mined_factor", exc))
    candidates.extend(build_agent_mined_candidates_from_frame(materialized.frame, symbol=symbol, duration=duration))
    return MinedCandidateResult(
        materialized.frame,
        tuple(candidates),
        materialized.source_count,
        tuple(failures),
    )


def _ensure_materialized(
    frame: pd.DataFrame,
    pending: dict[str, pd.Series],
    row: dict[str, Any],
    by_name: dict[str, dict[str, Any]],
    materialized: set[str],
    visiting: set[str],
) -> None:
    name = str(row.get("factorName"))
    if name in materialized:
        return
    if name in visiting:
        raise ValueError(f"cycle in mined factor library: {name}")
    visiting.add(name)
    try:
        members = _members(row)
        for member in members:
            _ensure_member_materialized(frame, pending, member, by_name, materialized, visiting)
        pending[name] = combination_score(_score_frame(frame, pending, members), members)
        materialized.add(name)
    finally:
        visiting.remove(name)


def _ensure_member_materialized(
    frame: pd.DataFrame,
    pending: dict[str, pd.Series],
    member: dict[str, Any],
    by_name: dict[str, dict[str, Any]],
    materialized: set[str],
    visiting: set[str],
) -> None:
    member_name = str(member["name"])
    if member_name in materialized:
        return
    dependency = by_name.get(member_name)
    if dependency is None:
        raise ValueError(f"mined factor missing member column: {member_name}")
    _ensure_materialized(frame, pending, dependency, by_name, materialized, visiting)


def _score_frame(
    frame: pd.DataFrame,
    pending: dict[str, pd.Series],
    members: list[dict[str, Any]],
) -> pd.DataFrame:
    pending_names = [str(member["name"]) for member in members if str(member["name"]) in pending]
    if not pending_names:
        return frame
    pending_frame = pd.DataFrame({name: pending[name] for name in pending_names}, index=frame.index)
    return pd.concat([frame, pending_frame], axis=1, copy=False)


def _frame_with_pending_columns(frame: pd.DataFrame, pending: dict[str, pd.Series]) -> pd.DataFrame:
    if not pending:
        return frame
    pending_frame = pd.DataFrame(pending, index=frame.index)
    return pd.concat([frame, pending_frame], axis=1, copy=False)


def _candidate_from_row(
    frame: pd.DataFrame,
    row: dict[str, Any],
    symbol: str,
    duration: str,
) -> MinedCandidate:
    factor = _factor_definition(row, duration)
    metrics = run_factor_backtest_on_frame(factor, frame, symbol=symbol, duration=duration)
    return MinedCandidate(factor=factor, metrics=metrics, orientation=1)


def _factor_definition(row: dict[str, Any], duration: str) -> FactorDefinition:
    return FactorDefinition(
        name=str(row["factorName"]),
        category=FactorCategory.PERFORMANCE,
        description=str(row.get("factorDisplayName") or row["factorName"]),
        formula=str(row.get("formula") or row["factorName"]),
        source_file=MINED_FACTOR_SOURCE_FILE,
        timeframes=(duration,),
        direction=FactorDirection.HIGHER_BETTER,
        parameters={"members": [member["name"] for member in _members(row)]},
    )


def is_mined_factor_source(source_file: str) -> bool:
    return source_file in {MINED_FACTOR_SOURCE_FILE, AGENT_FACTOR_SOURCE_FILE}


def _members(row: dict[str, Any]) -> list[dict[str, Any]]:
    members = row.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"mined factor missing members: {row.get('factorName')}")
    return [dict(member) for member in members]


def _usable_metrics(metrics: dict[str, Any]) -> bool:
    return int(metrics.get("totalPeriods") or 0) >= BACKTEST_MIN_PERIODS and metrics.get("winRate") is not None


def _failure(row: dict[str, Any], stage: str, exc: Exception) -> dict[str, Any]:
    return {"factorName": row.get("factorName"), "stage": stage, "error": str(exc)}
