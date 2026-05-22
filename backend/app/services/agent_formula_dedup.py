from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from app.services.agent_candidate_reporting import load_agent_candidate_history
from app.services.factor_learning_memory_store import FACTOR_LEARNING_DIR
from app.services.json_atomic_io import load_json_object

AGENT_FACTOR_LIBRARY_PATH = FACTOR_LEARNING_DIR / "agent_mined_factor_library.json"

DO_NOT_SUGGEST_FORMULA_LIMIT = 120
_DUPLICATE_HISTORY_STATUSES = frozenset({"duplicate_existing"})


def normalize_agent_formula(formula: str) -> str:
    return " ".join(str(formula or "").strip().split())


def library_agent_formulas(symbol: str, duration: str) -> frozenset[str]:
    sym = symbol.strip().upper()
    dur = str(duration)
    formulas: set[str] = set()
    for row in _load_agent_factor_rows():
        if not isinstance(row, dict):
            continue
        if str(row.get("symbol") or "").strip().upper() != sym or str(row.get("duration")) != dur:
            continue
        _add_formula(formulas, str(row.get("formula") or ""))
    return frozenset(formulas)


def duplicate_history_formulas(symbol: str, duration: str) -> frozenset[str]:
    sym = symbol.strip().upper()
    dur = str(duration)
    formulas: set[str] = set()
    for run in load_agent_candidate_history().get("runs") or []:
        if not isinstance(run, dict):
            continue
        if str(run.get("symbol") or "").strip().upper() != sym or str(run.get("duration")) != dur:
            continue
        for candidate in run.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("status") or "") not in _DUPLICATE_HISTORY_STATUSES:
                continue
            _add_formula(formulas, str(candidate.get("formula") or ""))
    return frozenset(formulas)


def known_agent_formulas(symbol: str, duration: str) -> frozenset[str]:
    library = library_agent_formulas(symbol, duration)
    duplicates = duplicate_history_formulas(symbol, duration)
    return library | duplicates


def is_known_agent_formula(formula: str, known: frozenset[str]) -> bool:
    normalized = normalize_agent_formula(formula)
    return bool(normalized) and normalized in known


def filter_duplicate_factor_ideas(
    ideas: list[dict[str, Any]],
    known: frozenset[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for idea in ideas:
        if not isinstance(idea, dict):
            continue
        hint = normalize_agent_formula(str(idea.get("formulaHint") or ""))
        if hint and hint in known:
            dropped.append(idea)
            continue
        kept.append(idea)
        if hint:
            known = known | frozenset({hint})
    return kept, dropped


def limited_do_not_suggest_formulas(symbol: str, duration: str) -> list[str]:
    sym = symbol.strip().upper()
    dur = str(duration)
    promoted: list[str] = []
    other: list[str] = []
    for row in _load_agent_factor_rows():
        if not isinstance(row, dict):
            continue
        if str(row.get("symbol") or "").strip().upper() != sym or str(row.get("duration")) != dur:
            continue
        formula = normalize_agent_formula(str(row.get("formula") or ""))
        if not formula or not _looks_executable_formula(formula):
            continue
        if _row_promoted(row):
            promoted.append(formula)
        else:
            other.append(formula)
    duplicates = sorted(duplicate_history_formulas(sym, dur))
    ordered: list[str] = []
    for formula in promoted + duplicates + sorted(set(other)):
        if formula not in ordered:
            ordered.append(formula)
    return ordered[:DO_NOT_SUGGEST_FORMULA_LIMIT]


def filter_agent_review_duplicates(review: dict[str, Any], symbol: str, duration: str) -> dict[str, Any]:
    updated = deepcopy(review)
    plan = updated.get("factorMiningPlan")
    if not isinstance(plan, dict):
        return updated
    ideas = [dict(item) for item in plan.get("candidateFactorIdeas") or [] if isinstance(item, dict)]
    known = known_agent_formulas(symbol, duration)
    kept, dropped = filter_duplicate_factor_ideas(ideas, known)
    plan = {**plan, "candidateFactorIdeas": kept}
    if dropped:
        notes = list(plan.get("skippedDuplicateFormulas") or [])
        for idea in dropped:
            hint = normalize_agent_formula(str(idea.get("formulaHint") or ""))
            if hint and hint not in notes:
                notes.append(hint)
        plan["skippedDuplicateFormulas"] = notes[:DO_NOT_SUGGEST_FORMULA_LIMIT]
    updated["factorMiningPlan"] = plan
    return updated


def _load_agent_factor_rows() -> list[dict[str, Any]]:
    if not AGENT_FACTOR_LIBRARY_PATH.exists():
        return []
    payload = load_json_object(AGENT_FACTOR_LIBRARY_PATH)
    if not isinstance(payload, dict):
        return []
    rows = payload.get("factors") or []
    return [row for row in rows if isinstance(row, dict)]


def _add_formula(formulas: set[str], raw: str) -> None:
    normalized = normalize_agent_formula(raw)
    if normalized and _looks_executable_formula(normalized):
        formulas.add(normalized)


def _looks_executable_formula(formula: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", formula):
        return False
    if re.search(r"\bor\b", formula, flags=re.IGNORECASE):
        return False
    return True


def _row_promoted(row: dict[str, Any]) -> bool:
    if isinstance(row.get("qualityPassed"), bool):
        return bool(row.get("qualityPassed"))
    return str(row.get("candidateStatus") or "") == "promoted"
