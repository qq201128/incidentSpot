from __future__ import annotations

from typing import Any

from app.services.agent_factor_categories import category_budget_payload, category_share
from app.services.factor_learning_retrieval import build_factor_learning_retrieval

AGENT_PROMPT_FORMULA_BLOCK_LIMIT = 48
AGENT_PROMPT_NAME_BLOCK_LIMIT = 16
AGENT_PROMPT_LIBRARY_ROW_LIMIT = 4
AGENT_PROMPT_COMBO_ROW_LIMIT = 3
AGENT_PROMPT_RETRIEVAL_NAME_LIMIT = 16
AGENT_PROMPT_RETRIEVAL_ROW_LIMIT = 8
AGENT_PROMPT_PATTERN_ROW_LIMIT = 3
AGENT_PROMPT_PATTERN_MEMBER_LIMIT = 2
AGENT_PROMPT_TEXT_LIMIT = 96
AGENT_PROMPT_SHORT_TEXT_LIMIT = 72


def compact_memory(memory: dict[str, Any], blocklist: list[Any]) -> dict[str, Any]:
    agent_factor_names = factor_names(memory.get("agentMinedFactorLibrary") or {})
    return {
        "symbol": memory.get("symbol"),
        "duration": memory.get("duration"),
        "source": slim_source(memory.get("source") or {}),
        "retrieval": slim_retrieval(build_factor_learning_retrieval(memory)),
        "factorMining": slim_factor_mining(memory.get("factorMining") or {}),
        "lossMemory": slim_loss_memory(memory.get("lossMemory") or {}),
        "filters": slim_filters(memory.get("filters") or {}),
        "weights": top_weights(memory.get("weights") or {}),
        "minedFactorLibrary": slim_mined_library(memory.get("minedFactorLibrary") or {}),
        "agentMinedFactorLibrary": slim_agent_library(memory.get("agentMinedFactorLibrary") or {}),
        "agentFactorCategoryBudget": category_budget_payload(),
        "doNotSuggestFactorNames": limited_strings(agent_factor_names, AGENT_PROMPT_NAME_BLOCK_LIMIT),
        "doNotSuggestFactorNameTotal": len(agent_factor_names),
        "doNotSuggestFormulas": limited_strings(blocklist, AGENT_PROMPT_FORMULA_BLOCK_LIMIT),
        "doNotSuggestFormulaTotal": len(blocklist),
        "monitoring": slim_monitoring(memory.get("monitoring") or {}),
    }


def slim_source(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": source.get("status"),
        "rankingRefreshSource": source.get("rankingRefreshSource"),
        "minedFrameFailureCount": source.get("minedFrameFailureCount"),
        "learningRefreshSource": source.get("learningRefreshSource"),
    }


def slim_retrieval(retrieval: dict[str, Any]) -> dict[str, Any]:
    blocked = limited_strings(retrieval.get("blockedFactorNames") or [], AGENT_PROMPT_RETRIEVAL_NAME_LIMIT)
    excluded = limited_strings(retrieval.get("miningExcludedFactorNames") or [], AGENT_PROMPT_RETRIEVAL_NAME_LIMIT)
    return {
        "blockedFactorNames": blocked,
        "blockedFactorNameTotal": len(retrieval.get("blockedFactorNames") or []),
        "miningExcludedFactorNames": excluded,
        "miningExcludedFactorNameTotal": len(retrieval.get("miningExcludedFactorNames") or []),
        "successPatterns": slim_pattern_rows(retrieval.get("successPatterns") or [], "pattern"),
        "forbiddenRegions": slim_pattern_rows(retrieval.get("forbiddenRegions") or [], "region"),
        "lossPatterns": slim_pattern_rows(retrieval.get("lossPatterns") or [], "pattern"),
        "topWeights": slim_weight_rows(retrieval.get("topWeights") or []),
        "summary": retrieval.get("summary") or {},
    }


def slim_factor_mining(factor_mining: dict[str, Any]) -> dict[str, Any]:
    success = factor_mining.get("successPatterns") or []
    forbidden = factor_mining.get("forbiddenRegions") or []
    return {"successPatternTotal": len(success), "forbiddenRegionTotal": len(forbidden)}


def slim_pattern_rows(rows: list[Any], label_key: str) -> list[dict[str, Any]]:
    slim = []
    for row in rows[:AGENT_PROMPT_PATTERN_ROW_LIMIT]:
        if not isinstance(row, dict):
            continue
        item = pattern_row(row, label_key)
        feature = truncate_text(row.get("feature"), AGENT_PROMPT_SHORT_TEXT_LIMIT)
        if feature:
            item["feature"] = feature
        slim.append(item)
    return slim


def pattern_row(row: dict[str, Any], label_key: str) -> dict[str, Any]:
    return {
        label_key: truncate_text(row.get(label_key) or row.get("pattern") or row.get("region"), AGENT_PROMPT_SHORT_TEXT_LIMIT),
        "support": row.get("support"),
        "members": limited_strings(row.get("members") or [], AGENT_PROMPT_PATTERN_MEMBER_LIMIT),
    }


def slim_filters(filters: dict[str, Any]) -> dict[str, Any]:
    slim: dict[str, Any] = {}
    for key, value in filters.items():
        name = truncate_text(key, AGENT_PROMPT_SHORT_TEXT_LIMIT)
        if isinstance(value, list):
            slim[name] = limited_strings(value, AGENT_PROMPT_RETRIEVAL_NAME_LIMIT)
            slim[f"{name}Total"] = len(value)
            continue
        slim[name] = value if not isinstance(value, str) else truncate_text(value)
    return slim


def slim_loss_memory(loss_memory: dict[str, Any]) -> dict[str, Any]:
    patterns = loss_memory.get("patterns") or []
    return {
        "status": loss_memory.get("status"),
        "sampleCount": loss_memory.get("sampleCount"),
        "lossCount": loss_memory.get("lossCount"),
        "patterns": slim_pattern_rows(patterns, "pattern"),
        "patternTotal": len(patterns),
    }


def slim_monitoring(monitoring: dict[str, Any]) -> dict[str, Any]:
    issues = monitoring.get("issues") or []
    return {
        "status": monitoring.get("status"),
        "issues": [slim_monitoring_issue(item) for item in issues[:AGENT_PROMPT_PATTERN_ROW_LIMIT] if isinstance(item, dict)],
        "issueTotal": len(issues),
    }


def slim_monitoring_issue(issue: dict[str, Any]) -> dict[str, Any]:
    return {str(key): truncate_text(value) if isinstance(value, str) else value for key, value in issue.items()}


def slim_mined_library(library: dict[str, Any]) -> dict[str, Any]:
    rows = [mined_library_row(row) for row in library.get("factors") or [] if isinstance(row, dict)]
    return {
        "total": library.get("total"),
        "duration": library.get("duration"),
        "symbol": library.get("symbol"),
        "factors": rows[:AGENT_PROMPT_COMBO_ROW_LIMIT],
    }


def mined_library_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    members = row.get("members") or []
    return {
        "factorName": truncate_text(row.get("factorName"), 96),
        "factorDisplayName": truncate_text(row.get("factorDisplayName")),
        "formula": truncate_text(row.get("formula")),
        "method": row.get("method"),
        "memberNames": limited_strings([item.get("name") or item for item in members if item], AGENT_PROMPT_PATTERN_MEMBER_LIMIT),
        "winRate": metrics.get("winRate"),
        "profitFactor": metrics.get("profitFactor"),
        "ir": metrics.get("ir"),
        "score": row.get("score") or row.get("factorScore"),
    }


def slim_agent_library(library: dict[str, Any]) -> dict[str, Any]:
    rows = [agent_library_row(row) for row in library.get("factors") or [] if isinstance(row, dict)]
    return {
        "total": library.get("total"),
        "candidateTotal": library.get("candidateTotal"),
        "rejectedTotal": library.get("rejectedTotal"),
        "duration": library.get("duration"),
        "symbol": library.get("symbol"),
        "categoryShare": library.get("categoryShare") or category_share(list(library.get("factors") or [])),
        "factors": rows[:AGENT_PROMPT_LIBRARY_ROW_LIMIT],
    }


def agent_library_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return {
        "factorName": truncate_text(row.get("factorName"), AGENT_PROMPT_SHORT_TEXT_LIMIT),
        "factorDisplayName": truncate_text(row.get("factorDisplayName")),
        "formula": truncate_text(row.get("formula")),
        "factorCategory": row.get("factorCategory"),
        "winRate": metrics.get("winRate"),
        "profitFactor": metrics.get("profitFactor"),
        "qualityPassed": row.get("qualityPassed"),
    }


def truncate_text(value: Any, limit: int = AGENT_PROMPT_TEXT_LIMIT) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def limited_strings(values: list[Any], limit: int) -> list[str]:
    items = []
    for value in values:
        text = truncate_text(value, AGENT_PROMPT_SHORT_TEXT_LIMIT)
        if text:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def slim_weight_rows(rows: list[Any]) -> list[dict[str, Any]]:
    return [
        {"name": truncate_text(row.get("name"), AGENT_PROMPT_SHORT_TEXT_LIMIT), "weight": row.get("weight")}
        for row in rows[:AGENT_PROMPT_RETRIEVAL_ROW_LIMIT]
        if isinstance(row, dict)
    ]


def top_weights(weights: dict[str, Any]) -> dict[str, Any]:
    pairs = sorted(weights.items(), key=lambda item: float(item[1]), reverse=True)
    return dict(pairs[:20])


def factor_names(library: dict[str, Any]) -> list[str]:
    return [str(row.get("factorName")) for row in library.get("factors") or [] if row.get("factorName")]
