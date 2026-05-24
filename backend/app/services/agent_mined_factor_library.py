from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.agent_candidate_reporting import (
    AGENT_CANDIDATE_HISTORY_PATH,
    agent_candidate_evaluation_summary,
    agent_candidate_promotion,
    append_agent_candidate_history,
)
from app.services.agent_formula_dedup import (
    filter_duplicate_factor_ideas,
    known_agent_formulas,
    normalize_agent_formula,
)
from app.services.agent_factor_formula import materialize_agent_formula
from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS, run_factor_backtest_on_frame
from app.services.factor_learning_common import SUCCESS_PROFIT_FACTOR_MIN, SUCCESS_WIN_RATE_MIN, utc_now
from app.services.factor_learning_memory_store import FACTOR_LEARNING_DIR
from app.services.json_atomic_io import load_json_object, save_json_object
from app.services.factor_metric_enrichment import enrich_factor_results, factor_score
from app.services.factor_registry import FactorCategory, FactorDefinition, FactorDirection

AGENT_FACTOR_LIBRARY_VERSION = "agent_mined_factor_library_v1"
AGENT_FACTOR_SOURCE_FILE = "agent_mined_factor_library.json"
AGENT_FACTOR_LIBRARY_PATH = FACTOR_LEARNING_DIR / AGENT_FACTOR_SOURCE_FILE
SUMMARY_LIMIT = 12

@dataclass(frozen=True)
class AgentFactorFrameResult:
    frame: pd.DataFrame
    source_count: int
    failures: tuple[dict[str, Any], ...]

@dataclass(frozen=True)
class AgentMinedCandidate:
    factor: FactorDefinition
    metrics: dict[str, Any]
    orientation: int

def process_agent_factor_candidates(memory: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    symbol = str(memory["symbol"]).strip().upper()
    duration = str(memory["duration"])
    ideas, _skipped = filter_duplicate_factor_ideas(_candidate_ideas(memory), known_agent_formulas(symbol, duration))
    rows, records = _evaluate_ideas(memory, frame, ideas)
    evaluation = agent_candidate_evaluation_summary(records)
    updated = deepcopy(memory)
    updated["agentCandidatePromotion"] = agent_candidate_promotion(records)
    updated["agentCandidateEvaluation"] = evaluation
    updated["llmAgent"] = _merge_agent_review(updated.get("llmAgent") or {}, evaluation)
    if rows:
        _save_library(_library_with_rows(rows))
    if ideas:
        append_agent_candidate_history(updated, records, evaluation, AGENT_CANDIDATE_HISTORY_PATH)
    summary = agent_mined_factor_library_summary(str(updated["symbol"]), str(updated["duration"]))
    updated["agentMinedFactorLibrary"] = summary
    from app.services.qualified_factor_simulation_slots import sync_qualified_simulation_slots

    updated["simulationSlotSync"] = sync_qualified_simulation_slots(symbol, duration)
    return updated

def agent_mined_factor_library_summary(symbol: str, duration: str) -> dict[str, Any]:
    rows = agent_factor_rows_for_duration(symbol, duration)
    promoted = _ingested_agent_rows(rows)
    simulation_eligible = [row for row in promoted if _simulation_eligible(row.get("metrics") or {})]
    promoted.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    return {
        "version": AGENT_FACTOR_LIBRARY_VERSION,
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "total": len(promoted),
        "simulationEligibleTotal": len(simulation_eligible),
        "candidateTotal": len(rows),
        "rejectedTotal": len(promoted) - len(simulation_eligible),
        "thresholds": _threshold_payload(),
        "factors": promoted[:SUMMARY_LIMIT],
    }

def materialize_agent_factor_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    duration: str,
    excluded_factor_names: set[str] | None = None,
) -> AgentFactorFrameResult:
    excluded = excluded_factor_names or set()
    rows = _ingested_agent_rows(agent_factor_rows_for_duration(symbol, duration))
    failures: list[dict[str, Any]] = []
    working = frame
    for row in rows:
        try:
            if str(row.get("factorName")) in excluded:
                continue
            working = _with_agent_column(working, row)
        except Exception as exc:
            failures.append(_failure(row, "materialize_agent_factor", exc))
    return AgentFactorFrameResult(working, len(rows), tuple(failures))

def build_agent_mined_candidates_from_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    duration: str,
    excluded_factor_names: set[str] | None = None,
) -> tuple[AgentMinedCandidate, ...]:
    excluded = excluded_factor_names or set()
    candidates = []
    for row in _ingested_agent_rows(agent_factor_rows_for_duration(symbol, duration)):
        factor_name = str(row.get("factorName"))
        if factor_name in excluded or factor_name not in frame.columns:
            continue
        factor = _factor_definition(row, duration)
        metrics = run_factor_backtest_on_frame(factor, frame, symbol=symbol, duration=duration)
        if _usable_metrics(metrics):
            candidates.append(AgentMinedCandidate(factor, metrics, _orientation(metrics)))
    return tuple(candidates)

def agent_factor_rows_for_duration(symbol: str, duration: str) -> list[dict[str, Any]]:
    rows = load_agent_factor_library().get("factors") or []
    return [
        deepcopy(row)
        for row in rows
        if _row_symbol(row) == symbol.strip().upper() and str(row.get("duration")) == duration
    ]

def load_agent_factor_library(path: Path | None = None) -> dict[str, Any]:
    target = path or AGENT_FACTOR_LIBRARY_PATH
    if not target.exists():
        return _empty_library()
    payload = load_json_object(target)
    if not isinstance(payload, dict):
        raise ValueError(f"agent mined factor library is not an object: {target}")
    return payload

def _evaluate_ideas(
    memory: dict[str, Any],
    frame: pd.DataFrame,
    ideas: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    library = load_agent_factor_library()
    existing = library.get("factors") or []
    symbol = str(memory["symbol"]).strip().upper()
    duration = str(memory["duration"])
    known = set(known_agent_formulas(symbol, duration))
    records = []
    for idea in ideas:
        record = _record_for_idea(memory, frame, idea, existing, known)
        records.append(record)
        formula = normalize_agent_formula(str(record.get("formula") or ""))
        if formula:
            known.add(formula)
    library_rows = [_library_row(record) for record in records]
    rows = _merged_rows(existing, [row for row in library_rows if row is not None])
    return rows, records

def _record_for_idea(
    memory: dict[str, Any],
    frame: pd.DataFrame,
    idea: dict[str, Any],
    existing: list[dict[str, Any]],
    known: set[str],
) -> dict[str, Any]:
    base = _record_base(memory, idea)
    formula = normalize_agent_formula(str(base["formula"]))
    if formula and (formula in known or _duplicate_formula(formula, existing)):
        return {**base, "status": "duplicate_existing"}
    try:
        working = _with_agent_column(frame, base)
        metrics = _backtest_record(base, working)
        return {**base, "metrics": metrics, "status": "promoted"}
    except Exception as exc:
        return {**base, "status": "failed", "error": str(exc)}

def _with_agent_column(frame: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    name = str(row["factorName"])
    if name in frame.columns:
        return frame
    series = materialize_agent_formula(frame, str(row["formula"]))
    return pd.concat([frame, pd.DataFrame({name: series}, index=frame.index)], axis=1, copy=False)

def _backtest_record(row: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    factor = _factor_definition(row, str(row["duration"]))
    metrics = run_factor_backtest_on_frame(factor, frame, symbol=str(row["symbol"]), duration=str(row["duration"]))
    enrich_factor_results([metrics], frame=frame)
    return metrics

def _library_row(record: dict[str, Any]) -> dict[str, Any] | None:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None
    now = utc_now()
    return {
        **_library_identity(record),
        "metrics": metrics,
        "score": factor_score(metrics),
        "candidateStatus": str(record.get("status") or "unknown"),
        "qualityPassed": _simulation_eligible(metrics),
        "firstSeenAt": now,
        "lastSeenAt": now,
        "promotionCount": 1 if record.get("status") == "promoted" else 0,
    }

def _library_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": str(row["symbol"]).strip().upper(),
        "duration": str(row["duration"]),
        "factorName": str(row["factorName"]),
        "factorDisplayName": str(row["displayName"]),
        "formula": str(row["formula"]),
        "source": AGENT_FACTOR_SOURCE_FILE,
        "idea": dict(row.get("idea") or {}),
    }

def _record_base(memory: dict[str, Any], idea: dict[str, Any]) -> dict[str, Any]:
    formula = str(idea.get("formulaHint") or "").strip()
    if not formula:
        raise ValueError("agent candidate formulaHint is required")
    return {
        "symbol": str(memory["symbol"]).strip().upper(),
        "duration": str(memory["duration"]),
        "factorName": _factor_name(idea, formula),
        "displayName": str(idea.get("displayNameZh") or idea.get("nameHint") or "Agent单因子"),
        "formula": formula,
        "idea": deepcopy(idea),
        "seenAt": utc_now(),
    }

def _factor_name(idea: dict[str, Any], formula: str) -> str:
    label = str(idea.get("nameHint") or idea.get("displayNameZh") or "agent_factor")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_").lower() or "agent_factor"
    digest = hashlib.sha1(formula.encode("utf-8")).hexdigest()[:10]
    return f"agent__{slug[:36]}__{digest}"

def _factor_definition(row: dict[str, Any], duration: str) -> FactorDefinition:
    return FactorDefinition(
        name=str(row["factorName"]),
        category=FactorCategory.STATISTIC,
        description=str(row.get("factorDisplayName") or row.get("displayName") or row["factorName"]),
        formula=str(row["formula"]),
        source_file=AGENT_FACTOR_SOURCE_FILE,
        timeframes=(duration,),
        direction=FactorDirection.NEUTRAL,
    )

def _merged_rows(existing: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {_row_key(row): deepcopy(row) for row in existing}
    for row in candidates:
        previous = by_key.get(_row_key(row))
        if previous:
            row["firstSeenAt"] = previous.get("firstSeenAt") or row["firstSeenAt"]
            row["promotionCount"] = int(previous.get("promotionCount") or 0) + (
                1 if row.get("qualityPassed") else 0
            )
        by_key[_row_key(row)] = row
    return sorted(by_key.values(), key=lambda row: float(row.get("score") or 0.0), reverse=True)

def _save_library(payload: dict[str, Any]) -> None:
    _save_json(AGENT_FACTOR_LIBRARY_PATH, payload)

def _save_json(path: Path, payload: dict[str, Any]) -> None:
    save_json_object(path, payload)

def _library_with_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"version": AGENT_FACTOR_LIBRARY_VERSION, "updatedAt": utc_now(), "thresholds": _threshold_payload(), "factors": rows}

def _candidate_ideas(memory: dict[str, Any]) -> list[dict[str, Any]]:
    plan = (((memory.get("llmAgent") or {}).get("review") or {}).get("factorMiningPlan") or {})
    return [dict(item) for item in plan.get("candidateFactorIdeas") or []]

def _simulation_eligible(metrics: dict[str, Any]) -> bool:
    return _num(metrics.get("winRate")) >= SUCCESS_WIN_RATE_MIN and _num(metrics.get("profitFactor")) >= SUCCESS_PROFIT_FACTOR_MIN


def _usable_metrics(metrics: dict[str, Any]) -> bool:
    return int(metrics.get("totalPeriods") or 0) >= BACKTEST_MIN_PERIODS and metrics.get("winRate") is not None


def _orientation(metrics: dict[str, Any]) -> int:
    return 1 if _num(metrics.get("ir")) >= 0 else -1


def _ingested_agent_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _row_ingested(row)]


def _row_ingested(row: dict[str, Any]) -> bool:
    if str(row.get("candidateStatus") or "") == "promoted":
        return True
    return isinstance(row.get("metrics"), dict) and str(row.get("candidateStatus") or "") not in {
        "failed",
        "duplicate_existing",
    }


def _merge_agent_review(llm_agent: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(llm_agent)
    review = updated.get("review") if isinstance(updated.get("review"), dict) else {}
    review = deepcopy(review)
    review["evaluation"] = evaluation
    updated["review"] = review
    return updated


def _duplicate_formula(formula: str, existing: list[dict[str, Any]]) -> bool:
    normalized = normalize_agent_formula(formula)
    return any(normalize_agent_formula(str(row.get("formula") or "")) == normalized for row in existing)


def _empty_library() -> dict[str, Any]:
    return {"version": AGENT_FACTOR_LIBRARY_VERSION, "thresholds": _threshold_payload(), "factors": []}


def _threshold_payload() -> dict[str, float]:
    return {"minWinRate": SUCCESS_WIN_RATE_MIN, "minProfitFactor": SUCCESS_PROFIT_FACTOR_MIN}


def _failure(row: dict[str, Any], stage: str, exc: Exception) -> dict[str, Any]:
    return {"factorName": row.get("factorName"), "stage": stage, "error": str(exc)}


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (_row_symbol(row), str(row.get("duration")), str(row.get("factorName")))


def _row_symbol(row: dict[str, Any]) -> str:
    return str(row.get("symbol") or "").strip().upper()


def _num(value: Any) -> float:
    return float(value) if value is not None else 0.0
