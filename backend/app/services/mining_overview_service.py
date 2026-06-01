from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.factor_learning_service import get_factor_learning_memory
from app.services.factor_operator_library import factor_operator_payload
from app.services.model_family_search_rules import DEFAULT_PARALLEL_WORKERS, TARGET_WIN_RATE_EXCLUSIVE
from app.services.model_search_resource import DEFAULT_INTERNAL_THREADS, DEFAULT_XGBOOST_PROCESS_WORKERS
from app.services.model_search_status_service import model_search_queue_status
from app.services.mining_run_status_service import mining_run_status
from app.services.llm_provider_registry import DEFAULT_LLM_PROVIDER, llm_model_metadata, llm_provider_availability
from app.services.siliconflow_chat_client import DEFAULT_SILICONFLOW_MODEL, resolved_siliconflow_model
from app.services.mining_overview_model_cards import model_card as _model_card
from app.services.mining_overview_model_cards import model_cards as _model_cards
from app.services.mining_agent_candidate_rows import (
    agent_candidate_rows as _agent_candidate_rows,
    agent_reviewed_at as _agent_reviewed_at,
    candidate_ideas as _candidate_ideas,
)
from app.services.mining_overview_sidebar import (
    ingestion_path as _ingestion_path,
    operators_sidebar as _operators_sidebar,
    refresh_status_key as _refresh_status_key,
    refresh_status_label as _refresh_status_label,
    sidebar_payload as _sidebar_payload,
)


@dataclass(frozen=True)
class _HeaderContext:
    memory: dict[str, Any]
    ideas: list
    promotion: dict
    agent_rows: list


def mining_overview(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    memory = get_factor_learning_memory(sym, duration)
    if memory is None:
        raise ValueError(f"factor learning memory not found for {sym} {duration}")

    operators = factor_operator_payload()
    models = _model_cards(sym, duration)
    search_queue = model_search_queue_status(
        {"symbols": (sym,), "durations": (duration,)},
        include_symbol_details=False,
    )
    agent_rows = _agent_candidate_rows(memory)
    promotion = memory.get("agentCandidatePromotion") or {}
    ideas = _candidate_ideas(memory)

    return {
        "symbol": sym,
        "duration": duration,
        "updatedAt": memory.get("updatedAt"),
        "runStatus": mining_run_status(memory, models, search_queue),
        "header": _header_payload(_HeaderContext(memory, ideas, promotion, agent_rows)),
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
def _header_payload(context: _HeaderContext) -> dict[str, Any]:
    refresh = context.memory.get("refreshTask") or {}
    agent = context.memory.get("llmAgent") or {}
    source = context.memory.get("source") or {}
    provider = agent.get("provider") or DEFAULT_LLM_PROVIDER
    model = agent.get("model") or _configured_agent_model()
    return {
        "localReplayStatus": _refresh_status_key(refresh, source),
        "localReplayLabel": _refresh_status_label(refresh, source),
        **_agent_counts(context),
        **_agent_status_payload(agent, provider, model),
        "agentReviewedAt": _agent_reviewed_at(context.memory),
        "memoryUpdatedAt": context.memory.get("updatedAt"),
    }


def _agent_counts(context: _HeaderContext) -> dict[str, int]:
    library = context.memory.get("agentMinedFactorLibrary") or {}
    idea_count = len(context.ideas)
    return {
        "agentIdeaCount": idea_count,
        "agentLibraryPairCount": int(library.get("candidateTotal") or context.promotion.get("candidateCount") or 0),
        "agentCandidateCount": idea_count,
        "pendingVerificationCount": sum(
            1 for row in context.agent_rows if row.get("validationStatusKey") == "pending_backtest"
        ),
    }


def _agent_status_payload(agent: dict[str, Any], provider: str, model: str) -> dict[str, Any]:
    return {
        "agentStatus": agent.get("status") or ("done" if agent.get("review") else "idle"),
        "agentProvider": provider,
        "agentModel": model,
        "agentCapabilities": agent.get("capabilities") or _agent_capabilities(provider, model),
        "agentAvailability": agent.get("availability") or llm_provider_availability(provider, model),
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
    queue_counts = search_queue.get("counts") or {}
    samples = _sample_counts(adaptive, loss)
    search = _search_counts(queue_counts, search_queue)
    candidates = _candidate_counts(models, promotion)
    return {
        "overallAccuracy": adaptive.get("overallAccuracy"),
        "accuracyCaption": "基于已结算预测样本",
        **samples,
        **search,
        **candidates,
        "readyModelCount": sum(1 for row in models if row.get("cardState") == "ready"),
        "totalModelCount": len(models),
    }


def _sample_counts(adaptive: dict[str, Any], loss: dict[str, Any]) -> dict[str, int]:
    sample_count = int(adaptive.get("sampleCount") or loss.get("sampleCount") or 0)
    loss_count = int(loss.get("lossCount") or 0)
    return {
        "sampleCount": sample_count,
        "lossSampleCount": loss_count,
        "winSampleCount": max(sample_count - loss_count, 0) if sample_count else 0,
    }


def _search_counts(queue_counts: dict[str, Any], search_queue: dict[str, Any]) -> dict[str, Any]:
    pending = int(queue_counts.get("pending") or 0)
    running = int(queue_counts.get("running") or 0)
    searching = pending + running
    return {
        "searchingCount": searching,
        "searchParallel": _search_parallel_label(search_queue, searching),
        "searchPendingCount": pending,
        "searchRunningCount": running,
    }


def _candidate_counts(models: list[dict], promotion: dict[str, Any]) -> dict[str, int]:
    promoted = int(promotion.get("promoted") or 0)
    agent_candidates = int(promotion.get("candidateCount") or 0)
    model_candidates = sum(int(row.get("candidateLibraryTotal") or 0) for row in models)
    return {
        "candidateRecordCount": model_candidates + agent_candidates,
        "candidatePending": agent_candidates - promoted,
        "candidateCompleted": promoted,
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
    return search_queue["workerStatus"]
