from __future__ import annotations

from typing import Any

ANSWER_LIMIT = 10


def candidate_pool_answers(
    focused: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    avoid_next_search: list[dict[str, Any]],
) -> dict[str, Any]:
    stable = [row for row in focused if row["status"] == "paper_stable"]
    collecting = [row for row in focused if row["status"] == "paper_collecting"]
    failed = [row for row in ranked if row["status"] in {"paper_failed", "invalid_data_leakage"}]
    return {
        "currentMostStable": [_summary(row) for row in stable[:ANSWER_LIMIT]],
        "collectingSamples": [_summary(row) for row in collecting[:ANSWER_LIMIT]],
        "failedCandidates": [_failure_summary(row) for row in failed[:ANSWER_LIMIT]],
        "failureReasons": _failure_reasons(failed, failures),
        "avoidNextSearch": avoid_next_search[:ANSWER_LIMIT],
    }


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") or {}
    return {
        "candidateKey": row.get("candidateKey"),
        "candidateType": row.get("candidateType"),
        "status": row.get("status"),
        "reason": row.get("reason"),
        "paperLiveWinRate": row.get("paperLiveWinRate"),
        "paperLiveSampleCount": row.get("paperLiveSampleCount"),
        "profitFactor": metrics.get("profitFactor"),
        "avgReturn": metrics.get("avgReturn"),
    }


def _failure_summary(row: dict[str, Any]) -> dict[str, Any]:
    summary = _summary(row)
    summary["failureReason"] = row.get("reason")
    return summary


def _failure_reasons(
    failed: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        {"candidateKey": row.get("candidateKey"), "reason": row.get("reason")}
        for row in failed[:ANSWER_LIMIT]
    ]
    rows.extend(
        {"candidateKey": row.get("candidateKey"), "reason": row.get("reason")}
        for row in failures[:ANSWER_LIMIT]
    )
    return rows[:ANSWER_LIMIT]
