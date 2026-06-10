from __future__ import annotations

import hashlib
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
from app.services.agent_factor_categories import (
    adjusted_score as _adjusted_category_score,
    category_share as _category_share,
    category_saturation as _category_saturation,
    factor_category as _factor_category,
)
from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS, run_factor_backtest_on_frame
from app.services.factor_learning_common import SUCCESS_PROFIT_FACTOR_MIN, SUCCESS_WIN_RATE_MIN, utc_now
from app.services.factor_learning_memory_store import FACTOR_LEARNING_DIR
from app.services.json_atomic_io import load_json_object, save_json_object
from app.services.factor_metric_enrichment import enrich_factor_results, factor_score
from app.services.factor_registry import FactorCategory, FactorDefinition, FactorDirection
from app.services.agent_mined_factor_library_helpers import (
    duplicate_formula as _duplicate_formula,
    empty_library as _empty_library,
    failure as _failure,
    ingested_agent_rows as _ingested_agent_rows,
    library_with_rows as _library_with_rows,
    merge_agent_review as _merge_agent_review,
    merged_rows as _merged_rows,
    num as _num,
    orientation as _orientation,
    row_key as _row_key,
    row_symbol as _row_symbol,
    simulation_eligible as _simulation_eligible,
    threshold_payload as _threshold_payload,
    usable_metrics as _usable_metrics,
)

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
        "categoryShare": _category_share(promoted),
        "factors": promoted[:SUMMARY_LIMIT],
    }

def materialize_agent_factor_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    duration: str,
    excluded_factor_names: set[str] | None = None,
) -> AgentFactorFrameResult:
    rows = _ingested_agent_rows(agent_factor_rows_for_duration(symbol, duration))
    return materialize_agent_factor_frame_for_rows(
        frame,
        rows=rows,
        source_count=len(rows),
        excluded_factor_names=excluded_factor_names,
    )

def materialize_agent_factor_frame_for_rows(
    frame: pd.DataFrame,
    *,
    rows: list[dict[str, Any]],
    source_count: int | None = None,
    excluded_factor_names: set[str] | None = None,
) -> AgentFactorFrameResult:
    excluded = excluded_factor_names or set()
    selected = _ingested_agent_rows(rows)
    failures: list[dict[str, Any]] = []
    working = frame
    for row in selected:
        try:
            if str(row.get("factorName")) in excluded:
                continue
            working = _with_agent_column(working, row)
        except Exception as exc:
            failures.append(_failure(row, "materialize_agent_factor", exc))
    count = len(selected) if source_count is None else source_count
    return AgentFactorFrameResult(working, count, tuple(failures))

def build_agent_mined_candidates_from_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    duration: str,
    excluded_factor_names: set[str] | None = None,
) -> tuple[AgentMinedCandidate, ...]:
    rows = _ingested_agent_rows(agent_factor_rows_for_duration(symbol, duration))
    return build_agent_mined_candidates_from_rows(
        frame,
        symbol=symbol,
        duration=duration,
        rows=rows,
        excluded_factor_names=excluded_factor_names,
    )

def build_agent_mined_candidates_from_rows(
    frame: pd.DataFrame,
    *,
    symbol: str,
    duration: str,
    rows: list[dict[str, Any]],
    excluded_factor_names: set[str] | None = None,
) -> tuple[AgentMinedCandidate, ...]:
    excluded = excluded_factor_names or set()
    candidates = []
    for row in _ingested_agent_rows(rows):
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
    category_context = [_row for _row in existing if _row_symbol(_row) == symbol and str(_row.get("duration")) == duration]
    library_rows = []
    for idea in ideas:
        record = _record_for_idea(memory, frame, idea, existing, known, category_context)
        records.append(record)
        formula = normalize_agent_formula(str(record.get("formula") or ""))
        if formula:
            known.add(formula)
        row = _library_row(record)
        if row is not None:
            library_rows.append(row)
            category_context.append(row)
    rows = _merged_rows(existing, [row for row in library_rows if row is not None])
    return rows, records

def _record_for_idea(
    memory: dict[str, Any],
    frame: pd.DataFrame,
    idea: dict[str, Any],
    existing: list[dict[str, Any]],
    known: set[str],
    category_context: list[dict[str, Any]],
) -> dict[str, Any]:
    base = _record_base(memory, idea)
    formula = normalize_agent_formula(str(base["formula"]))
    if formula and (formula in known or _duplicate_formula(formula, existing)):
        return {**base, "status": "duplicate_existing"}
    try:
        working = _with_agent_column(frame, base)
        metrics = _backtest_record(base, working)
        saturation = _category_saturation(str(base["factorCategory"]), category_context)
        status = "category_saturated" if saturation["saturated"] and _simulation_eligible(metrics) else "promoted"
        return {**base, "metrics": metrics, "status": status, "categorySaturation": saturation}
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
    category_saturated = str(record.get("status") or "") == "category_saturated"
    quality_passed = _simulation_eligible(metrics) and not category_saturated
    raw_score = factor_score(metrics)
    return {
        **_library_identity(record),
        "metrics": metrics,
        "score": _adjusted_category_score(raw_score, record.get("categorySaturation") or {}),
        "rawScore": raw_score,
        "factorCategory": record.get("factorCategory") or _factor_category(record),
        "categorySaturation": record.get("categorySaturation") or {},
        "candidateStatus": str(record.get("status") or "unknown"),
        "qualityPassed": quality_passed,
        "firstSeenAt": now,
        "lastSeenAt": now,
        "promotionCount": 1 if quality_passed and record.get("status") == "promoted" else 0,
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
        "factorCategory": str(idea.get("factorCategory") or _factor_category({"formula": formula, "idea": idea})),
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

def _save_library(payload: dict[str, Any]) -> None:
    _save_json(AGENT_FACTOR_LIBRARY_PATH, payload)

def _save_json(path: Path, payload: dict[str, Any]) -> None:
    save_json_object(path, payload)

def _candidate_ideas(memory: dict[str, Any]) -> list[dict[str, Any]]:
    plan = (((memory.get("llmAgent") or {}).get("review") or {}).get("factorMiningPlan") or {})
    return [dict(item) for item in plan.get("candidateFactorIdeas") or []]
