from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from app.db.session import get_conn
from app.services.agent_mined_factor_library import (
    agent_mined_factor_library_summary,
    process_agent_factor_candidates,
)
from app.services.factor_cache_metadata import cache_is_usable
from app.services.factor_learning_refresh_stale import (
    is_refresh_task_stale,
    stale_refresh_task_error,
)
from app.services.factor_learning_refresh_tasks import mark_factor_learning_refresh_failed
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_combination_cache_service import save_cached_combination_ranking
from app.services.factor_combination_service import CombinationSearchConfig
from app.services.factor_combination_service import run_factor_combination_ranking_on_frame
from app.services.factor_combo_monitor_service import factor_combo_monitor_report
from app.services.factor_combo_simulation_keys import (
    BATCH_COMBO_KEY_PREFIX,
    BATCH_HIGH_WINRATE_KEY_PREFIX,
    factor_combo_simulation_strategy_keys,
    high_winrate_factor_combo_simulation_strategy_keys,
)
from app.services.factor_learning_common import finite, utc_now
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_learning_core import build_factor_learning_memory
from app.services.factor_learning_patterns import factor_rows
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
from app.services.factor_mined_candidates import materialize_mined_factor_frame
from app.services.factor_mined_library import enrich_mined_factor_library_summary, mined_factor_library_summary
from app.services.forward_validation_service import settle_due_predictions
from app.services.lstm_config import lstm_shadow_strategy_key
from app.services.lstm_shadow_learning import lstm_shadow_learning_summary
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

COMBO_FACTOR_PREFIXES = ("combo__", "goal_combo__")
LEARNING_METRIC_KEYS = ("winRate", "profitFactor", "sharpe", "ir")

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
) -> dict[str, Any]:
    _validate_duration(duration)
    sym = symbol.strip().upper()
    previous_memory = load_factor_learning_memory(sym, duration)
    base_frame = load_factor_frame(sym, duration)
    report = ranking_report or _current_ranking_report(sym, duration, base_frame)
    settlement = settle_due_predictions(sym, duration)
    mined_frame = materialize_mined_factor_frame(base_frame, symbol=sym, duration=duration)
    predictions = _settled_factor_combo_predictions(sym, duration)
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
        return _attach_agent_review_and_save(memory)
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

def run_factor_learning_llm_agent(symbol: str, duration: str) -> dict[str, Any]:
    _validate_duration(duration)
    memory = load_factor_learning_memory(symbol, duration)
    if memory is None:
        raise ValueError(f"factor learning memory not found for {symbol.upper()} {duration}")
    return _attach_agent_review_and_save(memory)

def _enrich_learning_memory(memory: dict[str, Any]) -> dict[str, Any]:
    enriched = deepcopy(memory)
    library = enriched.get("minedFactorLibrary")
    if isinstance(library, dict):
        enriched["minedFactorLibrary"] = enrich_mined_factor_library_summary(library)
    return enriched

def _attach_agent_review_and_save(memory: dict[str, Any]) -> dict[str, Any]:
    try:
        reviewed = attach_llm_agent_review(memory)
        frame = load_factor_frame(str(reviewed["symbol"]), str(reviewed["duration"]))
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

def _current_ranking_report(symbol: str, duration: str, frame: pd.DataFrame) -> dict[str, Any]:
    cached = get_cached_combination_ranking(symbol, duration)
    fresh = _fresh_cached_ranking(cached)
    if fresh is not None:
        return {**fresh, "learningRefreshSource": "cache"}
    stale = _stale_cached_ranking_for_learning(cached)
    if stale is not None:
        return {**stale, "learningRefreshSource": "stale_cache"}
    report = run_factor_combination_ranking_on_frame(
        frame,
        symbol=symbol,
        duration=duration,
        config=CombinationSearchConfig(),
    )
    save_cached_combination_ranking(report)
    return {**report, "learningRefreshSource": "rebuilt_cache"}


def _fresh_cached_ranking(cached: dict[str, Any] | None) -> dict[str, Any] | None:
    if cached is None:
        return None
    if not cache_is_usable(cached):
        return None
    if not _has_learning_metric_rows(cached):
        return None
    return cached


def _stale_cached_ranking_for_learning(cached: dict[str, Any] | None) -> dict[str, Any] | None:
    """Reuse combination cache for learning/agent refresh without a full combo search rebuild."""
    if cached is None:
        return None
    if cache_is_usable(cached):
        return None
    if not _has_learning_metric_rows(cached):
        return None
    return cached

def _has_learning_metric_rows(report: dict[str, Any]) -> bool:
    for row in factor_rows(report):
        name = str(row.get("name") or "")
        if name.startswith(COMBO_FACTOR_PREFIXES):
            continue
        if any(finite(row.get(key)) is not None for key in LEARNING_METRIC_KEYS):
            return True
    return False

def _settled_factor_combo_predictions(symbol: str, duration: str) -> list[dict[str, Any]]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT open_time, direction, confidence, trade_quality_score,
                   actual_return, prediction_correct, high_winrate_rule,
                   signal_key, strategy_key
            FROM predictions
            WHERE (
                signal_key IN ({placeholders})
                OR signal_key LIKE ?
                OR signal_key LIKE ?
            )
              AND symbol = ? AND duration = ?
              AND settled_at IS NOT NULL
            ORDER BY open_time
            """.format(placeholders=_strategy_placeholders()),
            (
                *_fixed_simulation_strategy_keys(duration),
                f"{BATCH_COMBO_KEY_PREFIX}%",
                f"{BATCH_HIGH_WINRATE_KEY_PREFIX}%",
                symbol.upper(),
                duration,
            ),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]

def _fixed_simulation_strategy_keys(duration: str) -> tuple[str, ...]:
    return (
        *factor_combo_simulation_strategy_keys(),
        *high_winrate_factor_combo_simulation_strategy_keys(),
        lstm_shadow_strategy_key(duration),
    )

def _strategy_placeholders() -> str:
    return ",".join("?" for _key in _fixed_simulation_strategy_keys_for_placeholders())

def _fixed_simulation_strategy_keys_for_placeholders() -> tuple[str, ...]:
    return (
        *factor_combo_simulation_strategy_keys(),
        *high_winrate_factor_combo_simulation_strategy_keys(),
        "factor_lstm_shadow_placeholder",
    )

def _validate_duration(duration: str) -> None:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")

def _path_payload(path: Path) -> str:
    return str(path)
