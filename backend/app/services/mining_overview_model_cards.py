from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from app.services.lstm_combo_snapshot import current_combo_snapshot
from app.services.model_family_config import MODEL_FAMILIES, model_family_label
from app.services.model_family_status_service import model_family_status


@dataclass(frozen=True)
class _ModelCardContext:
    status: dict[str, Any]
    progress: dict[str, Any]
    library: dict[str, Any]
    rules: dict[str, Any]


def model_cards(symbol: str, duration: str) -> list[dict[str, Any]]:
    shared_combo = current_combo_snapshot(symbol, duration)

    def load_card(family: str) -> dict[str, Any]:
        status = model_family_status(
            family,
            symbol,
            duration,
            current_combo_snapshot=shared_combo,
        )
        return model_card(status)

    with ThreadPoolExecutor(max_workers=min(6, len(MODEL_FAMILIES))) as pool:
        return list(pool.map(load_card, MODEL_FAMILIES))


def model_card(status: dict[str, Any]) -> dict[str, Any]:
    family = status.get("modelFamily") or "lstm"
    progress = status.get("candidateSearchProgress") or {}
    library = status.get("candidateLibrary") or {}
    rules = status.get("trainingRules") or {}
    context = _ModelCardContext(status, progress, library, rules)
    return {
        "modelFamily": family,
        "label": model_family_label(family),
        "strategyKey": status.get("strategyKey"),
        "cardState": _card_state(status),
        "cardStateLabel": _card_state_label(status),
        "predictionReadyLabel": _prediction_ready_label(status),
        "validationWinRate": _validation_win_rate(status),
        "testWinRate": status.get("testWinRate"),
        "searchStatus": progress.get("status") or "idle",
        "searchProgress": _search_progress_payload(context),
        "latestCandidateLabel": _latest_candidate_label(progress),
        "candidateLibraryTotal": int(library.get("total") or 0),
        "latestFailureReason": _latest_failure_reason(status, progress, library),
        "latestLogPath": _latest_log_path(progress),
        "updatedAt": progress.get("updatedAt") or status.get("trainedAt"),
        "blockedReason": status.get("shadowPredictionBlockedReason"),
        "status": status.get("status"),
    }


def _validation_win_rate(status: dict[str, Any]) -> Any:
    validation = (status.get("validationGate") or {}).get("metrics") or {}
    if validation.get("winRate") is not None:
        return validation.get("winRate")
    return status.get("validationWinRate")


def _search_progress_payload(context: _ModelCardContext) -> dict[str, Any]:
    return {
        "completed": int(context.progress.get("completed") or 0),
        "total": int(context.progress.get("total") or context.rules.get("searchSpaceTotal") or 0),
        "percent": float(context.progress.get("percent") or 0),
    }


def _latest_failure_reason(status: dict[str, Any], progress: dict[str, Any], library: dict[str, Any]) -> str | None:
    job = progress.get("modelSearchJob") or {}
    explicit = (
        progress.get("failureReason")
        or _last_failure_reason(progress)
        or job.get("failure_reason")
        or job.get("rejection_reason")
    )
    if explicit:
        return str(explicit)
    if _prediction_ready(status) or _has_successful_candidate(progress, library):
        return None
    return status.get("validationFailureReason")


def _last_failure_reason(progress: dict[str, Any]) -> str | None:
    failure = progress.get("lastFailure")
    if not isinstance(failure, dict):
        return None
    return failure.get("error") or failure.get("reason")


def _has_successful_candidate(progress: dict[str, Any], library: dict[str, Any]) -> bool:
    if str(progress.get("status") or "") in {"trade_active", "trained", "shadow_active"}:
        return True
    return bool(library.get("bestTradeCandidate") or library.get("bestShadowCandidate"))


def _latest_log_path(progress: dict[str, Any]) -> str | None:
    job = progress.get("modelSearchJob") or {}
    return progress.get("logPath") or job.get("log_path")


def _card_state(status: dict[str, Any]) -> str:
    progress = status.get("candidateSearchProgress") or {}
    if progress.get("status") in {"queued", "running"}:
        return "searching"
    if _prediction_ready(status):
        return "ready"
    active = status.get("activeModelStatus") or status.get("status")
    if active in {None, "untrained", "insufficient_samples", "queued", "training"}:
        return "pending_train"
    return "blocked"


def _card_state_label(status: dict[str, Any]) -> str:
    mapping = {
        "ready": "可模拟下单",
        "searching": "搜索中",
        "pending_train": "待训练",
        "blocked": "已阻断",
    }
    return mapping[_card_state(status)]


def _prediction_ready_label(status: dict[str, Any]) -> str:
    if _prediction_ready(status):
        return "就绪"
    return "未就绪"


def _prediction_ready(status: dict[str, Any]) -> bool:
    return bool(
        status.get("shadowPredictionReady")
        or status.get("shadowPredictionBlockedReason") == "combo_snapshot_mismatch"
    )


def _latest_candidate_label(progress: dict[str, Any]) -> str | None:
    latest = progress.get("latestCompleted")
    if not latest:
        return None
    cfg = latest.get("config") or {}
    status = latest.get("status") or "—"
    return f"{status} · w{cfg.get('featureWindow', '—')}"
