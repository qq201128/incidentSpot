from __future__ import annotations


def daily_loop_failure_details(report: dict) -> dict:
    return {"failedResults": [_failed_result_summary(result) for result in report.get("results") or [] if result.get("status") != "passed"]}


def _failed_result_summary(result: dict) -> dict:
    return {
        "symbol": result.get("symbol"),
        "duration": result.get("duration"),
        "status": result.get("status"),
        "failedStages": _failed_stage_summaries(result),
    }


def _failed_stage_summaries(result: dict) -> list[dict]:
    return [
        {"stage": stage.get("stage"), "reason": stage.get("reason"), "exceptionType": stage.get("exceptionType")}
        for stage in result.get("stages") or []
        if stage.get("status") == "failed"
    ]
