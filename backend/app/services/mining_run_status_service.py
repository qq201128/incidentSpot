from __future__ import annotations

from typing import Any

from app.services.mining_agent_candidate_rows import agent_reviewed_at
from app.services.mining_overview_sidebar import refresh_status_key


def mining_run_status(memory: dict, models: list[dict], search_queue: dict[str, Any]) -> dict[str, Any]:
    sections = {
        "localReplay": _local_replay_status(memory),
        "agentMining": _agent_mining_status(memory),
        "modelSearch": _model_search_status(search_queue),
        "worker": search_queue["workerStatus"],
        "cache": _cache_status(memory, models),
    }
    model_statuses = [_model_runtime_status(row, search_queue) for row in models]
    return {
        "version": "mining_run_status_v1",
        "overall": _overall_status(sections, model_statuses),
        "sections": sections,
        "models": model_statuses,
    }


def _local_replay_status(memory: dict) -> dict[str, Any]:
    refresh = memory.get("refreshTask") or {}
    state = str(refresh.get("status") or refresh_status_key(refresh, memory.get("source") or {}) or "idle")
    updated = refresh.get("updatedAt") or memory.get("updatedAt")
    return _section_status("localReplay", state, updated_at=updated, detail=refresh.get("error"))


def _agent_mining_status(memory: dict) -> dict[str, Any]:
    agent = memory.get("llmAgent") or {}
    state = str(agent.get("status") or ("done" if agent.get("review") else "idle"))
    return _section_status("agentMining", state, updated_at=agent_reviewed_at(memory), detail=agent.get("error"))


def _model_search_status(search_queue: dict[str, Any]) -> dict[str, Any]:
    worker = search_queue["workerStatus"]
    return _section_status(
        "modelSearch",
        worker.get("state") or "idle",
        detail=worker.get("latestFailureReason"),
        latest_log_path=worker.get("latestLogPath"),
        counts=search_queue.get("counts") or {},
    )


def _cache_status(memory: dict, models: list[dict]) -> dict[str, Any]:
    updated = [row.get("updatedAt") for row in models if row.get("updatedAt")]
    updated.append(memory.get("updatedAt"))
    latest = sorted(str(item) for item in updated if item)[-1] if any(updated) else None
    return _section_status("cache", "ready" if latest else "idle", updated_at=latest)


def _section_status(name: str, state: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "state": _normalized_state(state), "rawState": state, **extra}


def _model_runtime_status(model: dict[str, Any], search_queue: dict[str, Any]) -> dict[str, Any]:
    worker = search_queue["workerStatus"]
    state = _model_state(model, worker)
    return {
        "modelFamily": model["modelFamily"],
        "state": state,
        "ready": model.get("cardState") == "ready",
        "searchStatus": model.get("searchStatus"),
        "pendingWorker": state == "worker_required",
        "latestFailureReason": model.get("latestFailureReason") or worker.get("latestFailureReason"),
        "latestLogPath": model.get("latestLogPath") or worker.get("latestLogPath"),
        "candidateLibraryTotal": model.get("candidateLibraryTotal"),
        "latestCandidateLabel": model.get("latestCandidateLabel"),
        "updatedAt": model.get("updatedAt"),
    }


def _model_state(model: dict[str, Any], worker: dict[str, Any]) -> str:
    if model.get("searchStatus") in {"queued", "running"}:
        return worker.get("state") or model["searchStatus"]
    if model.get("cardState") == "ready":
        return "ready"
    if worker.get("state") == "failed" and model.get("latestFailureReason"):
        return "failed"
    return model.get("searchStatus") or model.get("cardState") or "idle"


def _overall_status(sections: dict[str, dict], models: list[dict]) -> dict[str, Any]:
    states = [section["state"] for section in sections.values()] + [model["state"] for model in models]
    if "failed" in states:
        state = "failed"
    elif any(item in states for item in ("running", "queued")):
        state = "running"
    elif "worker_required" in states:
        state = "worker_required"
    elif "ready" in states:
        state = "ready"
    else:
        state = "idle"
    return {"state": state}


def _normalized_state(state: str) -> str:
    if state in {"pending", "queued"}:
        return "queued"
    if state in {"running", "training"}:
        return "running"
    if state in {"failed", "error"}:
        return "failed"
    if state in {"completed", "done", "ready"}:
        return "ready"
    return state or "idle"
