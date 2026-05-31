from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.services.factor_learning_service import get_factor_learning_memory
from app.services.lstm_combo_snapshot import current_combo_snapshot
from app.services.factor_operator_library import factor_operator_payload
from app.services.model_family_config import MODEL_FAMILIES, model_family_label
from app.services.model_family_status_service import model_family_status
from app.services.model_family_search_rules import DEFAULT_PARALLEL_WORKERS, TARGET_WIN_RATE_EXCLUSIVE
from app.services.model_search_resource import DEFAULT_INTERNAL_THREADS, DEFAULT_XGBOOST_PROCESS_WORKERS
from app.services.model_search_status_service import model_search_queue_status
from app.services.llm_provider_registry import DEFAULT_LLM_PROVIDER, llm_model_metadata, llm_provider_availability
from app.services.siliconflow_chat_client import DEFAULT_SILICONFLOW_MODEL, resolved_siliconflow_model
from app.services.mining_agent_candidate_rows import (
    agent_candidate_rows as _agent_candidate_rows,
    agent_reviewed_at as _agent_reviewed_at,
    candidate_ideas as _candidate_ideas,
)
from app.services.mining_overview_sidebar import (
    OPERATOR_CATEGORY_LABELS,
    RANKING_SOURCE_LABELS,
    ingestion_path as _ingestion_path,
    operators_sidebar as _operators_sidebar,
    refresh_status_key as _refresh_status_key,
    refresh_status_label as _refresh_status_label,
    sidebar_payload as _sidebar_payload,
)

def mining_overview(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    memory = get_factor_learning_memory(sym, duration)
    if memory is None:
        raise ValueError(f"factor learning memory not found for {sym} {duration}")

    operators = factor_operator_payload()
    models = _model_cards(sym, duration)
    search_queue = model_search_queue_status({"symbols": (sym,), "durations": (duration,)})
    agent_rows = _agent_candidate_rows(memory)
    promotion = memory.get("agentCandidatePromotion") or {}
    ideas = _candidate_ideas(memory)

    return {
        "symbol": sym,
        "duration": duration,
        "updatedAt": memory.get("updatedAt"),
        "header": _header_payload(memory, ideas, promotion, agent_rows),
        "summary": _summary_payload(memory, models, search_queue),
        "trainingRules": _training_rules_payload(search_queue),
        "modelSearchQueue": search_queue,
        "models": models,
        "agentCandidates": agent_rows,
        "sidebar": _sidebar_payload(memory, operators, ideas, promotion),
        "ingestionPath": _ingestion_path(memory, ideas, promotion),
        "operators": _operators_sidebar(operators),
        "memory": memory,
    }

def _model_cards(symbol: str, duration: str) -> list[dict[str, Any]]:
    shared_combo = current_combo_snapshot(symbol, duration)

    def load_card(family: str) -> dict[str, Any]:
        status = model_family_status(
            family,
            symbol,
            duration,
            current_combo_snapshot=shared_combo,
        )
        return _model_card(status)

    with ThreadPoolExecutor(max_workers=min(6, len(MODEL_FAMILIES))) as pool:
        return list(pool.map(load_card, MODEL_FAMILIES))


def _header_payload(memory: dict, ideas: list, promotion: dict, agent_rows: list) -> dict[str, Any]:
    refresh = memory.get("refreshTask") or {}
    agent = memory.get("llmAgent") or {}
    library = memory.get("agentMinedFactorLibrary") or {}
    pending = sum(1 for row in agent_rows if row.get("validationStatusKey") == "pending_backtest")
    idea_count = len(ideas)
    library_pair_count = int(library.get("candidateTotal") or promotion.get("candidateCount") or 0)
    agent_model = agent.get("model") or _configured_agent_model()
    agent_provider = agent.get("provider") or DEFAULT_LLM_PROVIDER
    return {
        "localReplayStatus": _refresh_status_key(refresh, memory.get("source") or {}),
        "localReplayLabel": _refresh_status_label(refresh, memory.get("source") or {}),
        "agentIdeaCount": idea_count,
        "agentLibraryPairCount": library_pair_count,
        "agentCandidateCount": idea_count,
        "pendingVerificationCount": pending,
        "agentStatus": agent.get("status") or ("done" if agent.get("review") else "idle"),
        "agentProvider": agent_provider,
        "agentModel": agent_model,
        "agentCapabilities": agent.get("capabilities") or _agent_capabilities(agent_provider, agent_model),
        "agentAvailability": agent.get("availability") or llm_provider_availability(agent_provider, agent_model),
        "agentReviewedAt": _agent_reviewed_at(memory),
        "memoryUpdatedAt": memory.get("updatedAt"),
    }

def _configured_agent_model() -> str:
    try:
        return resolved_siliconflow_model()
    except RuntimeError:
        return DEFAULT_SILICONFLOW_MODEL

def _agent_capabilities(provider: str, model: str) -> list[str]:
    return llm_model_metadata(provider, model)["capabilities"]

def _summary_payload(memory: dict, models: list[dict], search_queue: dict[str, Any]) -> dict[str, Any]:
    adaptive = memory.get("adaptiveLearning") or {}
    loss = memory.get("lossMemory") or {}
    promotion = memory.get("agentCandidatePromotion") or {}
    sample_count = int(adaptive.get("sampleCount") or loss.get("sampleCount") or 0)
    loss_count = int(loss.get("lossCount") or 0)
    win_count = max(sample_count - loss_count, 0) if sample_count else 0
    queue_counts = search_queue.get("counts") or {}
    searching = int(queue_counts.get("pending") or 0) + int(queue_counts.get("running") or 0)
    candidate_records = sum(int(row.get("candidateLibraryTotal") or 0) for row in models)
    candidate_records += int(promotion.get("candidateCount") or 0)
    return {
        "overallAccuracy": adaptive.get("overallAccuracy"),
        "accuracyCaption": "基于已结算预测样本",
        "sampleCount": sample_count,
        "lossSampleCount": loss_count,
        "winSampleCount": win_count,
        "searchingCount": searching,
        "searchParallel": _search_parallel_label(search_queue, searching),
        "searchPendingCount": int(queue_counts.get("pending") or 0),
        "searchRunningCount": int(queue_counts.get("running") or 0),
        "candidateRecordCount": candidate_records,
        "candidatePending": int(promotion.get("candidateCount") or 0) - int(promotion.get("promoted") or 0),
        "candidateCompleted": int(promotion.get("promoted") or 0),
        "readyModelCount": sum(1 for row in models if row.get("cardState") == "ready"),
        "totalModelCount": len(models),
    }

def _training_rules_payload(search_queue: dict[str, Any]) -> dict[str, Any]:
    target = int(TARGET_WIN_RATE_EXCLUSIVE * 100)
    return {
        "text": f"候选置信阈值下胜率必须严格 > {target}%，successive-halving 分阶段筛选",
        "targetWinRateExclusive": TARGET_WIN_RATE_EXCLUSIVE,
        "internalThreads": DEFAULT_INTERNAL_THREADS,
        "parallelWorkers": DEFAULT_PARALLEL_WORKERS,
        "xgboostProcessWorkers": DEFAULT_XGBOOST_PROCESS_WORKERS,
        "workerStatus": _worker_status(search_queue),
    }


def _search_parallel_label(search_queue: dict[str, Any], searching: int) -> str:
    running = search_queue.get("runningJobs") or []
    workers = max(len(running), 1 if searching else 0)
    return f"{searching} queued/running · worker {workers}"


def _worker_status(search_queue: dict[str, Any]) -> dict[str, Any]:
    counts = search_queue.get("counts") or {}
    pending = int(counts.get("pending") or 0)
    running = int(counts.get("running") or 0)
    if pending > 0 and running == 0:
        state = "worker_required"
    elif running > 0:
        state = "running"
    else:
        state = "idle"
    return {
        "state": state,
        "pendingJobs": pending,
        "runningJobs": running,
        "latestLogPath": search_queue.get("latestLogPath"),
    }

def _model_card(status: dict[str, Any]) -> dict[str, Any]:
    family = status.get("modelFamily") or "lstm"
    progress = status.get("candidateSearchProgress") or {}
    library = status.get("candidateLibrary") or {}
    rules = status.get("trainingRules") or {}
    validation = (status.get("validationGate") or {}).get("metrics") or {}
    validation_win = validation.get("winRate")
    if validation_win is None:
        validation_win = status.get("validationWinRate")
    return {
        "modelFamily": family,
        "label": model_family_label(family),
        "strategyKey": status.get("strategyKey"),
        "cardState": _card_state(status),
        "cardStateLabel": _card_state_label(status),
        "predictionReadyLabel": _prediction_ready_label(status),
        "validationWinRate": validation_win,
        "testWinRate": status.get("testWinRate"),
        "searchStatus": progress.get("status") or "idle",
        "searchProgress": {
            "completed": int(progress.get("completed") or 0),
            "total": int(progress.get("total") or rules.get("searchSpaceTotal") or 0),
            "percent": float(progress.get("percent") or 0),
        },
        "latestCandidateLabel": _latest_candidate_label(progress),
        "candidateLibraryTotal": int(library.get("total") or 0),
        "blockedReason": status.get("shadowPredictionBlockedReason"),
        "status": status.get("status"),
    }

def _card_state(status: dict) -> str:
    progress = status.get("candidateSearchProgress") or {}
    if progress.get("status") in {"queued", "running"}:
        return "searching"
    if _prediction_ready(status):
        return "ready"
    active = status.get("activeModelStatus") or status.get("status")
    if active in {None, "untrained", "insufficient_samples", "queued", "training"}:
        return "pending_train"
    return "blocked"

def _card_state_label(status: dict) -> str:
    mapping = {
        "ready": "可模拟下单",
        "searching": "搜索中",
        "pending_train": "待训练",
        "blocked": "已阻断",
    }
    return mapping[_card_state(status)]

def _prediction_ready_label(status: dict) -> str:
    if _prediction_ready(status):
        return "就绪"
    return "未就绪"


def _prediction_ready(status: dict) -> bool:
    return bool(
        status.get("shadowPredictionReady")
        or status.get("shadowPredictionBlockedReason") == "combo_snapshot_mismatch"
    )

def _latest_candidate_label(progress: dict) -> str | None:
    latest = progress.get("latestCompleted")
    if not latest:
        return None
    cfg = latest.get("config") or {}
    status = latest.get("status") or "—"
    return f"{status} · w{cfg.get('featureWindow', '—')}"

