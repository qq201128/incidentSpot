from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

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
    rows = mined_factor_rows_for_duration(symbol, duration)
    if not rows:
        return MinedFrameResult(frame, 0, ())
    working = frame.copy()
    failures: list[dict[str, Any]] = []
    materialized = {str(column) for column in working.columns}
    by_name = {str(row.get("factorName")): row for row in rows}
    for row in rows:
        try:
            _ensure_materialized(working, row, by_name, materialized, set())
        except Exception as exc:
            failures.append(_failure(row, "materialize_mined_factor", exc))
    return MinedFrameResult(working, len(rows), tuple(failures))


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
    return MinedCandidateResult(
        materialized.frame,
        tuple(candidates),
        materialized.source_count,
        tuple(failures),
    )


def _ensure_materialized(
    frame: pd.DataFrame,
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
    members = _members(row)
    for member in members:
        member_name = str(member["name"])
        if member_name in materialized:
            continue
        dependency = by_name.get(member_name)
        if dependency is None:
            raise ValueError(f"mined factor missing member column: {member_name}")
        _ensure_materialized(frame, dependency, by_name, materialized, visiting)
    frame[name] = combination_score(frame, members)
    materialized.add(name)
    visiting.remove(name)


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


def _members(row: dict[str, Any]) -> list[dict[str, Any]]:
    members = row.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"mined factor missing members: {row.get('factorName')}")
    return [dict(member) for member in members]


def _usable_metrics(metrics: dict[str, Any]) -> bool:
    return int(metrics.get("totalPeriods") or 0) >= BACKTEST_MIN_PERIODS and metrics.get("winRate") is not None


def _failure(row: dict[str, Any], stage: str, exc: Exception) -> dict[str, Any]:
    return {"factorName": row.get("factorName"), "stage": stage, "error": str(exc)}
