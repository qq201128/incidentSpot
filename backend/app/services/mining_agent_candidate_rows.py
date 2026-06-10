from __future__ import annotations

from typing import Any


def agent_candidate_rows(memory: dict) -> list[dict[str, Any]]:
    ideas = candidate_ideas(memory)
    promotion = memory.get("agentCandidatePromotion") or {}
    record_index = promotion_record_index(promotion)
    reviewed_at = agent_reviewed_at(memory)
    rows = []
    matched_record_ids: set[str] = set()
    for index, idea in enumerate(ideas):
        record = lookup_promotion_record(record_index, idea)
        row_id = str(record.get("factorName") or idea.get("nameHint") or idea.get("displayNameZh") or f"idea-{index}")
        if record:
            matched_record_ids.add(row_id)
        rows.append(agent_candidate_row(idea, record, row_id=row_id, agent_reviewed_at=reviewed_at))
    rows.extend(unmatched_record_rows(promotion, matched_record_ids, reviewed_at))
    return rows


def unmatched_record_rows(promotion: dict, matched_ids: set[str], reviewed_at: str | None) -> list[dict[str, Any]]:
    rows = []
    for record in promotion.get("records") or []:
        row_id = str(record.get("factorName") or record.get("formula") or "")
        if row_id in matched_ids:
            continue
        rows.append(agent_candidate_row({}, record, row_id=row_id or "record", agent_reviewed_at=reviewed_at))
    return rows


def promotion_record_index(promotion: dict) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for item in promotion.get("records") or []:
        if not isinstance(item, dict):
            continue
        idea = item.get("idea") if isinstance(item.get("idea"), dict) else {}
        for alias in promotion_aliases(item, idea):
            text = str(alias or "").strip()
            if text:
                index[text] = item
    return index


def promotion_aliases(item: dict[str, Any], idea: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item.get("factorName"),
        item.get("nameHint"),
        item.get("displayName"),
        idea.get("nameHint"),
        idea.get("displayNameZh"),
        idea.get("formulaHint"),
        item.get("formula"),
    )


def lookup_promotion_record(index: dict[str, dict[str, Any]], idea: dict[str, Any]) -> dict[str, Any]:
    for alias in (idea.get("nameHint"), idea.get("displayNameZh"), idea.get("formulaHint")):
        text = str(alias or "").strip()
        if text and text in index:
            return index[text]
    return {}


def agent_candidate_row(
    idea: dict[str, Any],
    record: dict[str, Any],
    *,
    row_id: str,
    agent_reviewed_at: str | None,
) -> dict[str, Any]:
    idea_payload = idea if isinstance(idea, dict) else {}
    record_payload = record if isinstance(record, dict) else {}
    has_idea = bool(idea_payload)
    return {
        "id": row_id,
        "factorName": candidate_display_name(record_payload, idea_payload),
        "operatorTrace": idea_payload.get("operatorTrace") or [],
        "formulaHint": record_payload.get("formula") or idea_payload.get("formulaHint"),
        "factorCategory": record_payload.get("factorCategory") or idea_payload.get("factorCategory"),
        "categorySaturation": record_payload.get("categorySaturation") or {},
        "rationale": candidate_rationale(record_payload, idea_payload),
        "validationStatus": validation_status_label(record_payload, idea_payload if has_idea else {}),
        "validationStatusKey": validation_status_key(record_payload, idea_payload if has_idea else {}),
        "source": "Agent",
        "createdAt": record_evaluated_at(record_payload, agent_reviewed_at),
        "agentReviewedAt": agent_reviewed_at,
    }


def candidate_display_name(record: dict[str, Any], idea: dict[str, Any]) -> Any:
    return (
        record.get("displayName")
        or idea.get("displayNameZh")
        or idea.get("nameHint")
        or record.get("factorName")
        or "—"
    )


def candidate_rationale(record: dict[str, Any], idea: dict[str, Any]) -> Any:
    return idea.get("rationaleZh") or idea.get("rationale") or record.get("error") or record.get("reason")


def record_evaluated_at(record: dict[str, Any], reviewed_at: str | None) -> str | None:
    return str(record.get("seenAt") or record.get("evaluatedAt") or reviewed_at or "") or None


def agent_reviewed_at(memory: dict) -> str | None:
    agent = memory.get("llmAgent") or {}
    reviewed_at = agent.get("reviewedAt")
    if reviewed_at:
        return str(reviewed_at)
    refresh = memory.get("refreshTask") or {}
    if refresh.get("runAgent") and refresh.get("status") == "completed":
        updated = refresh.get("updatedAt")
        return str(updated) if updated else None
    return None


def candidate_ideas(memory: dict) -> list[dict]:
    return list(memory.get("llmAgent", {}).get("review", {}).get("factorMiningPlan", {}).get("candidateFactorIdeas") or [])


def validation_status_key(record: dict, idea: dict) -> str:
    status = str(record.get("status") or "")
    if status == "promoted":
        return "promoted"
    if status == "rejected_metrics":
        return "rejected_metrics"
    if status == "category_saturated":
        return "category_saturated"
    if status == "failed":
        return "failed"
    if status == "duplicate_existing":
        return "duplicate"
    if idea and not record:
        return "pending_backtest"
    if record:
        return "materialized"
    return "pending_backtest"


def validation_status_label(record: dict, idea: dict) -> str:
    labels = {
        "promoted": "已入库",
        "rejected_metrics": "入库：未达标",
        "category_saturated": "类别过量",
        "failed": "物化失败",
        "duplicate": "已存在",
        "pending_backtest": "待回测",
        "materialized": "公式已物化",
    }
    return labels.get(validation_status_key(record, idea), "待回测")
