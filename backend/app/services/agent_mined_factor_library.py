from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.agent_factor_formula import materialize_agent_formula
from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS, run_factor_backtest_on_frame
from app.services.factor_learning_common import SUCCESS_PROFIT_FACTOR_MIN, SUCCESS_WIN_RATE_MIN, utc_now
from app.services.factor_learning_memory_store import FACTOR_LEARNING_DIR
from app.services.factor_metric_enrichment import enrich_factor_results, factor_score
from app.services.factor_registry import FactorCategory, FactorDefinition, FactorDirection

AGENT_FACTOR_LIBRARY_VERSION = "agent_mined_factor_library_v1"
AGENT_CANDIDATE_HISTORY_VERSION = "agent_factor_candidate_history_v1"
AGENT_FACTOR_SOURCE_FILE = "agent_mined_factor_library.json"
AGENT_FACTOR_LIBRARY_PATH = FACTOR_LEARNING_DIR / AGENT_FACTOR_SOURCE_FILE
AGENT_CANDIDATE_HISTORY_PATH = FACTOR_LEARNING_DIR / "agent_factor_candidate_history.json"
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
    ideas = _candidate_ideas(memory)
    rows, records = _evaluate_ideas(memory, frame, ideas)
    if rows:
        _save_library(_library_with_rows(rows))
    if ideas:
        _append_history(memory, records)
    summary = agent_mined_factor_library_summary(str(memory["symbol"]), str(memory["duration"]))
    return {**memory, "agentMinedFactorLibrary": summary, "agentCandidatePromotion": _promotion(records)}

def agent_mined_factor_library_summary(symbol: str, duration: str) -> dict[str, Any]:
    rows = agent_factor_rows_for_duration(symbol, duration)
    rows.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    return {
        "version": AGENT_FACTOR_LIBRARY_VERSION,
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "total": len(rows),
        "thresholds": _threshold_payload(),
        "factors": rows[:SUMMARY_LIMIT],
    }

def materialize_agent_factor_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    duration: str,
) -> AgentFactorFrameResult:
    rows = agent_factor_rows_for_duration(symbol, duration)
    failures: list[dict[str, Any]] = []
    working = frame
    for row in rows:
        try:
            working = _with_agent_column(working, row)
        except Exception as exc:
            failures.append(_failure(row, "materialize_agent_factor", exc))
    return AgentFactorFrameResult(working, len(rows), tuple(failures))

def build_agent_mined_candidates_from_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    duration: str,
) -> tuple[AgentMinedCandidate, ...]:
    candidates = []
    for row in agent_factor_rows_for_duration(symbol, duration):
        if str(row.get("factorName")) not in frame.columns:
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
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
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
    records = [_record_for_idea(memory, frame, idea, existing) for idea in ideas]
    promoted = [_library_row(record) for record in records if record["status"] == "promoted"]
    rows = _merged_rows(existing, [row for row in promoted if row is not None])
    return rows, records

def _record_for_idea(
    memory: dict[str, Any],
    frame: pd.DataFrame,
    idea: dict[str, Any],
    existing: list[dict[str, Any]],
) -> dict[str, Any]:
    base = _record_base(memory, idea)
    if _duplicate_formula(base["formula"], existing):
        return {**base, "status": "duplicate_existing"}
    try:
        working = _with_agent_column(frame, base)
        metrics = _backtest_record(base, working)
        return {**base, "metrics": metrics, "status": _status(metrics)}
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
        "firstSeenAt": now,
        "lastSeenAt": now,
        "promotionCount": 1,
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
            row["promotionCount"] = int(previous.get("promotionCount") or 0) + 1
        by_key[_row_key(row)] = row
    return sorted(by_key.values(), key=lambda row: float(row.get("score") or 0.0), reverse=True)

def _append_history(memory: dict[str, Any], records: list[dict[str, Any]]) -> None:
    history = _load_history()
    runs = history.get("runs") or []
    runs.append({"symbol": memory["symbol"], "duration": memory["duration"], "seenAt": utc_now(), "candidates": records})
    _save_json(AGENT_CANDIDATE_HISTORY_PATH, {**history, "updatedAt": utc_now(), "runs": runs})

def _load_history() -> dict[str, Any]:
    if not AGENT_CANDIDATE_HISTORY_PATH.exists():
        return {"version": AGENT_CANDIDATE_HISTORY_VERSION, "runs": []}
    with AGENT_CANDIDATE_HISTORY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def _save_library(payload: dict[str, Any]) -> None:
    _save_json(AGENT_FACTOR_LIBRARY_PATH, payload)

def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _library_with_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"version": AGENT_FACTOR_LIBRARY_VERSION, "updatedAt": utc_now(), "thresholds": _threshold_payload(), "factors": rows}


def _candidate_ideas(memory: dict[str, Any]) -> list[dict[str, Any]]:
    plan = (((memory.get("llmAgent") or {}).get("review") or {}).get("factorMiningPlan") or {})
    return [dict(item) for item in plan.get("candidateFactorIdeas") or []]


def _status(metrics: dict[str, Any]) -> str:
    return "promoted" if _promotable(metrics) else "rejected_metrics"


def _promotable(metrics: dict[str, Any]) -> bool:
    return _num(metrics.get("winRate")) >= SUCCESS_WIN_RATE_MIN and _num(metrics.get("profitFactor")) >= SUCCESS_PROFIT_FACTOR_MIN


def _usable_metrics(metrics: dict[str, Any]) -> bool:
    return int(metrics.get("totalPeriods") or 0) >= BACKTEST_MIN_PERIODS and metrics.get("winRate") is not None


def _orientation(metrics: dict[str, Any]) -> int:
    return 1 if _num(metrics.get("ir")) >= 0 else -1


def _promotion(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"candidateCount": len(records), "promoted": sum(1 for item in records if item["status"] == "promoted"), "records": records}


def _duplicate_formula(formula: str, existing: list[dict[str, Any]]) -> bool:
    return any(str(row.get("formula") or "") == formula for row in existing)


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
