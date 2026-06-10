from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.factor_learning_common import utc_now
from app.services.factor_learning_memory_store import FACTOR_LEARNING_DIR
from app.services.json_atomic_io import load_json_object, save_json_object

AGENT_CANDIDATE_HISTORY_VERSION = "agent_factor_candidate_history_v1"
AGENT_CANDIDATE_HISTORY_PATH = FACTOR_LEARNING_DIR / "agent_factor_candidate_history.json"
KNOWN_STATUSES = ("promoted", "category_saturated", "duplicate_existing", "rejected_metrics", "failed")


def agent_candidate_promotion(records: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = sum(1 for item in records if item["status"] == "promoted")
    return {"candidateCount": len(records), "promoted": promoted, "records": records}


def agent_candidate_evaluation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {status: 0 for status in KNOWN_STATUSES}
    for record in records:
        status = str(record.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    promoted = counts.get("promoted", 0)
    return {
        "generatedCount": len(records),
        "promotedCount": promoted,
        "rejectedCount": len(records) - promoted,
        "statusCounts": counts,
        "topPromotedFactors": _promoted_factor_names(records),
        "engineSupportBacklog": _engine_support_backlog(records),
    }


def append_agent_candidate_history(
    memory: dict[str, Any],
    records: list[dict[str, Any]],
    evaluation: dict[str, Any],
    path: Path | None = None,
) -> None:
    target = path or AGENT_CANDIDATE_HISTORY_PATH
    history = load_agent_candidate_history(target)
    runs = history.get("runs") or []
    runs.append(
        {
            "symbol": memory["symbol"],
            "duration": memory["duration"],
            "seenAt": utc_now(),
            "evaluation": evaluation,
            "candidates": records,
        }
    )
    _save_json(target, {**history, "updatedAt": utc_now(), "runs": runs})


def load_agent_candidate_history(path: Path | None = None) -> dict[str, Any]:
    target = path or AGENT_CANDIDATE_HISTORY_PATH
    if not target.exists():
        return {"version": AGENT_CANDIDATE_HISTORY_VERSION, "runs": []}
    return load_json_object(target)


def _promoted_factor_names(records: list[dict[str, Any]]) -> list[str]:
    names = []
    for record in records:
        if record.get("status") == "promoted" and record.get("factorName"):
            names.append(str(record["factorName"]))
    return names[:8]


def _engine_support_backlog(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    items = []
    for record in records:
        if record.get("status") == "failed":
            items.append(_engine_support_item(record))
    return items[:12]


def _engine_support_item(record: dict[str, Any]) -> dict[str, str]:
    return {
        "factorName": str(record.get("factorName") or ""),
        "formula": str(record.get("formula") or ""),
        "error": str(record.get("error") or ""),
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    save_json_object(path, payload)
