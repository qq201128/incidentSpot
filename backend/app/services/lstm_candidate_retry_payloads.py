from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.kline_timing import MS_PER_MINUTE
from app.services.lstm_candidate_keys import search_key_for_config
from app.services.lstm_candidate_search import search_space_size
from app.services.lstm_config import LstmTrainingConfig
from app.services.rule_config import DURATION_TO_MINUTES, SUPPORTED_RULE_DURATIONS

TRAINED_STATUSES = {"trained", "shadow_active", "trade_active"}


def skipped_result(
    symbol: str,
    duration: str,
    status: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "duration": duration,
        "status": "skipped",
        "reason": decision["reason"],
        "activeModelStatus": status.get("activeModelStatus") or status.get("status"),
        "lastAttemptStatus": status.get("lastAttemptStatus"),
    }


def search_exhausted_result(symbol: str, duration: str, status: dict[str, Any], decision: dict[str, Any], config: Any) -> dict[str, Any]:
    return {
        **skipped_result(symbol, duration, status, decision),
        "reason": "candidate_search_exhausted",
        "searchSpaceTotal": search_space_size(config.search),
    }


def trained_result(
    symbol: str,
    duration: str,
    status: dict[str, Any],
    decision: dict[str, Any],
    ranking: dict[str, Any],
    promotion: dict[str, Any],
    trainings: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "duration": duration,
        "status": training_batch_status(trainings),
        "reason": decision["reason"],
        "previousActiveModelStatus": status.get("activeModelStatus") or status.get("status"),
        "rankingTotal": len(ranking.get("ranking") or []),
        "promotion": promotion,
        "training": training_summary(trainings[-1]),
        "candidates": [candidate_result_summary(training) for training in trainings],
    }


def training_batch_status(trainings: list[dict[str, Any]]) -> str:
    if any(str(item.get("status") or "") in {"trade_active", "trained"} for item in trainings):
        return "trade_active"
    if any(str(item.get("status") or "") == "shadow_active" for item in trainings):
        return "shadow_active"
    if trainings:
        return str(trainings[-1].get("status") or "failed")
    return "skipped"


def training_summary(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status", "modelVersion", "trainedAt", "sampleCounts",
        "candidateStatus", "promotionReason", "validationFailureReason",
    )
    return {key: report.get(key) for key in keys if key in report}


def candidate_result_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = training_summary(report)
    summary["searchKey"] = report.get("searchKey")
    return {key: value for key, value in summary.items() if value is not None}


def failed_training_report(config: LstmTrainingConfig, profile: str, exc: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "candidateStatus": "failed",
        "symbol": config.symbol,
        "duration": config.duration,
        "modelVersion": None,
        "searchKey": search_key_for_config(config, profile),
        "trainedAt": utc_now(),
        "validationFailureReason": str(exc),
    }


def summary_status(results: list[dict[str, Any]]) -> str:
    if any(result.get("status") not in {"skipped", *TRAINED_STATUSES} for result in results):
        return "completed_with_rejections"
    if any(result.get("status") in TRAINED_STATUSES for result in results):
        return "trained"
    return "skipped"


def normalized_symbols(symbols: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(symbol.strip().upper() for symbol in symbols if symbol.strip())
    if not normalized:
        raise ValueError("at least one LSTM retry symbol is required")
    return normalized


def validated_durations(durations: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(duration.strip() for duration in durations if duration.strip())
    unsupported = [duration for duration in selected if duration not in SUPPORTED_RULE_DURATIONS]
    if not selected:
        raise ValueError("at least one LSTM retry duration is required")
    if unsupported:
        raise ValueError(f"unsupported LSTM retry durations: {', '.join(unsupported)}")
    return selected


def duration_ms(duration: str) -> int:
    return int(DURATION_TO_MINUTES[duration]) * MS_PER_MINUTE


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
