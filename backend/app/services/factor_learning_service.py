from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from app.services.agent_mined_factor_library import (
    agent_mined_factor_library_summary,
    process_agent_factor_candidates,
)
from app.services.factor_learning_refresh_stale import (
    is_refresh_task_stale,
    stale_refresh_task_error,
)
from app.services.factor_learning_refresh_tasks import mark_factor_learning_refresh_failed
from app.services.factor_combo_monitor_service import factor_combo_monitor_report
from app.services.factor_learning_common import utc_now
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_learning_core import build_factor_learning_memory
from app.services.factor_learning_llm_agent import (
    AGENT_NAME,
    AGENT_PROVIDER,
    attach_llm_agent_review,
    is_llm_agent_run_stale,
    stale_llm_agent_error,
)
from app.services.siliconflow_chat_client import resolved_siliconflow_model
from app.services.factor_learning_memory_store import (
    FACTOR_LEARNING_VERSION,
    load_factor_learning_memory,
    save_factor_learning_memory,
)
from app.services.llm_provider_registry import llm_model_metadata, llm_provider_availability
from app.services.factor_mined_candidates import materialize_mined_factor_frame
from app.services.factor_mined_library import enrich_mined_factor_library_summary, mined_factor_library_summary
from app.services.factor_learning_predictions import settled_factor_combo_predictions
from app.services.factor_learning_ranking import current_ranking_report
from app.services.forward_validation_service import settle_due_predictions
from app.services.lstm_shadow_learning import lstm_shadow_learning_summary
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

def get_factor_learning_memory(symbol: str, duration: str) -> dict[str, Any] | None:
    _validate_duration(duration)
    memory = load_factor_learning_memory(symbol, duration)
    if memory is None:
        return None
    memory = recover_stale_learning_task_memory(memory)
    return _enrich_learning_memory(memory)


def recover_stale_learning_task_memory(memory: dict[str, Any]) -> dict[str, Any]:
    memory = recover_stale_refresh_task_memory(memory)
    return recover_stale_llm_agent_memory(memory)


def recover_stale_refresh_task_memory(memory: dict[str, Any]) -> dict[str, Any]:
    task = memory.get("refreshTask")
    if not isinstance(task, dict) or not is_refresh_task_stale(task):
        return memory
    sym = str(memory.get("symbol") or "").strip().upper()
    dur = str(memory.get("duration") or "")
    error = stale_refresh_task_error(task)
    run_agent = bool(task.get("runAgent"))
    if sym and dur:
        updated = mark_factor_learning_refresh_failed(sym, dur, error, run_agent=run_agent)
        agent = updated.get("llmAgent") or {}
        if run_agent and str(agent.get("status") or "") in {"pending", "running"}:
            return mark_factor_learning_agent_failed(sym, dur, error)
        return updated
    updated = deepcopy(memory)
    updated["refreshTask"] = {
        **task,
        "status": "failed",
        "error": error,
        "updatedAt": utc_now(),
    }
    return _save_memory_payload(updated)


def recover_stale_llm_agent_memory(memory: dict[str, Any]) -> dict[str, Any]:
    agent = memory.get("llmAgent")
    if not isinstance(agent, dict):
        return memory
    if not is_llm_agent_run_stale(agent) and not _orphaned_llm_agent_after_refresh(memory, agent):
        return memory
    sym = str(memory.get("symbol") or "").strip().upper()
    dur = str(memory.get("duration") or "")
    error = _orphaned_llm_agent_error(memory, agent) or stale_llm_agent_error(agent)
    if sym and dur:
        return mark_factor_learning_agent_failed(sym, dur, error)
    return _save_factor_learning_agent_status(memory, "failed", error)


def _orphaned_llm_agent_after_refresh(memory: dict[str, Any], agent: dict[str, Any]) -> bool:
    task = memory.get("refreshTask")
    if not isinstance(task, dict):
        return False
    if str(task.get("status") or "") not in {"failed", "completed"}:
        return False
    status = str(agent.get("status") or "")
    if status not in {"pending", "running"}:
        return False
    if agent.get("review"):
        return False
    # Refresh can finish before the LLM returns; running is normal until stale timeout.
    if status == "running":
        return is_llm_agent_run_stale(agent)
    return True


def _orphaned_llm_agent_error(memory: dict[str, Any], agent: dict[str, Any]) -> str | None:
    if not _orphaned_llm_agent_after_refresh(memory, agent):
        return None
    task = memory.get("refreshTask") or {}
    if str(task.get("status") or "") == "failed":
        detail = str(task.get("error") or "").strip()
        if detail:
            return f"联网挖掘未执行：复盘任务已失败（{detail}）"
        return "联网挖掘未执行：复盘任务已失败，请重新点击联网挖掘。"
    return "联网挖掘未执行：复盘已完成但未写回 Agent review，请重新点击联网挖掘。"

def refresh_factor_learning_memory(
    symbol: str,
    duration: str,
    ranking_report: dict[str, Any] | None = None,
    *,
    run_llm_agent: bool = False,
    factor_lookback_days: int | None = None,
) -> dict[str, Any]:
    _validate_duration(duration)
    sym = symbol.strip().upper()
    previous_memory = load_factor_learning_memory(sym, duration)
    base_frame = load_factor_frame(sym, duration, lookback_days=factor_lookback_days)
    report = ranking_report or current_ranking_report(
        sym,
        duration,
        base_frame,
        use_cache=factor_lookback_days is None,
    )
    settlement = settle_due_predictions(sym, duration)
    mined_frame = materialize_mined_factor_frame(base_frame, symbol=sym, duration=duration)
    predictions = settled_factor_combo_predictions(sym, duration)
    memory = build_factor_learning_memory(
        mined_frame.frame,
        report,
        predictions,
        symbol=sym,
        duration=duration,
        settlement_sweep=settlement,
        mined_frame_failures=list(mined_frame.failures),
        mined_library=mined_factor_library_summary(sym, duration),
        agent_mined_library=agent_mined_factor_library_summary(sym, duration),
        monitoring_report=factor_combo_monitor_report(sym, duration),
        lstm_shadow=lstm_shadow_learning_summary(sym, duration),
        previous_memory=previous_memory,
    )
    if run_llm_agent:
        return _attach_agent_review_and_save(memory, factor_lookback_days=factor_lookback_days)
    return _save_memory_payload(memory)

def mark_factor_learning_agent_pending(memory: dict[str, Any]) -> dict[str, Any]:
    return _save_factor_learning_agent_status(memory, "pending")

def mark_factor_learning_agent_running(memory: dict[str, Any]) -> dict[str, Any]:
    return _save_factor_learning_agent_status(memory, "running")

def mark_factor_learning_agent_failed(symbol: str, duration: str, error: str) -> dict[str, Any]:
    _validate_duration(duration)
    sym = symbol.strip().upper()
    memory = load_factor_learning_memory(sym, duration) or _queued_memory(sym, duration)
    return _save_factor_learning_agent_status(memory, "failed", error)

def run_factor_learning_llm_agent(
    symbol: str,
    duration: str,
    *,
    factor_lookback_days: int | None = None,
) -> dict[str, Any]:
    _validate_duration(duration)
    memory = load_factor_learning_memory(symbol, duration)
    if memory is None:
        raise ValueError(f"factor learning memory not found for {symbol.upper()} {duration}")
    return _attach_agent_review_and_save(memory, factor_lookback_days=factor_lookback_days)

def _enrich_learning_memory(memory: dict[str, Any]) -> dict[str, Any]:
    enriched = deepcopy(memory)
    library = enriched.get("minedFactorLibrary")
    if isinstance(library, dict):
        enriched["minedFactorLibrary"] = enrich_mined_factor_library_summary(library)
    return enriched

def _attach_agent_review_and_save(
    memory: dict[str, Any],
    *,
    factor_lookback_days: int | None = None,
) -> dict[str, Any]:
    try:
        reviewed = attach_llm_agent_review(memory)
        frame = load_factor_frame(
            str(reviewed["symbol"]),
            str(reviewed["duration"]),
            lookback_days=factor_lookback_days,
        )
        promoted = process_agent_factor_candidates(reviewed, frame)
        return _save_memory_payload(promoted)
    except Exception as exc:
        _save_factor_learning_agent_status(memory, "failed", str(exc))
        raise

def _save_factor_learning_agent_status(
    memory: dict[str, Any],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    updated = deepcopy(memory)
    updated["llmAgent"] = _agent_status_payload(status, error, previous=updated.get("llmAgent"))
    return _save_memory_payload(updated)

def _agent_status_payload(
    status: str,
    error: str | None,
    *,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = utc_now()
    payload = {
        "agent": AGENT_NAME,
        "provider": AGENT_PROVIDER,
        "status": status,
        "updatedAt": now,
        "model": _resolved_agent_model(previous),
    }
    payload.update(_agent_registry_payload(str(payload["model"])))
    if status in {"pending", "running"}:
        payload["agentStartedAt"] = now
    elif isinstance(previous, dict) and previous.get("agentStartedAt"):
        payload["agentStartedAt"] = previous.get("agentStartedAt")
    if error:
        payload["error"] = error
    return payload

def _resolved_agent_model(previous: dict[str, Any] | None) -> str:
    if isinstance(previous, dict):
        model = str(previous.get("model") or "").strip()
        if model:
            return model
    try:
        return resolved_siliconflow_model()
    except RuntimeError:
        return ""

def _agent_registry_payload(model: str) -> dict[str, Any]:
    if not model:
        return {"capabilities": [], "availability": llm_provider_availability(AGENT_PROVIDER)}
    return {
        "capabilities": llm_model_metadata(AGENT_PROVIDER, model)["capabilities"],
        "availability": llm_provider_availability(AGENT_PROVIDER, model),
    }

def _queued_memory(symbol: str, duration: str) -> dict[str, Any]:
    return {
        "version": FACTOR_LEARNING_VERSION,
        "symbol": symbol,
        "duration": duration,
        "updatedAt": utc_now(),
        "source": {"status": "queued"},
    }

def _save_memory_payload(memory: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(memory)
    payload.pop("memoryPath", None)
    path = save_factor_learning_memory(payload)
    return {**payload, "memoryPath": _path_payload(path)}

def _validate_duration(duration: str) -> None:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")

def _path_payload(path: Path) -> str:
    return str(path)
