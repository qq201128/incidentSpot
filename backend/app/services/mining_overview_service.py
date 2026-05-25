from __future__ import annotations

from typing import Any

from app.services.factor_learning_service import get_factor_learning_memory
from app.services.factor_operator_library import factor_operator_payload
from app.services.model_family_status_service import model_family_status
from app.services.siliconflow_chat_client import DEFAULT_SILICONFLOW_MODEL, resolved_siliconflow_model, siliconflow_config_from_env

MODEL_FAMILIES = (
    "lstm",
    "gru",
    "cnn",
    "transformer",
    "random_forest",
    "xgboost",
    "svm",
    "rl_strategy",
    "bayesian",
    "knn",
)

FAMILY_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn": "CNN",
    "transformer": "Transformer",
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
    "svm": "SVM",
    "rl_strategy": "QTableDirection",
    "bayesian": "Bayesian",
    "knn": "KNN",
}

OPERATOR_CATEGORY_LABELS = {
    "arithmetic": "算术类",
    "time_series": "时序类",
    "difference": "差分类",
    "cross_section": "截面类",
    "microstructure": "微观结构",
    "derived": "衍生指标",
}

RANKING_SOURCE_LABELS = {
    "cache": "ranking_cache",
    "rebuilt_cache": "ranking_cache",
    "provided": "ranking_rebuild",
}


def mining_overview(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    memory = get_factor_learning_memory(sym, duration)
    if memory is None:
        raise ValueError(f"factor learning memory not found for {sym} {duration}")

    operators = factor_operator_payload()
    models = [_model_card(model_family_status(family, sym, duration)) for family in MODEL_FAMILIES]
    agent_rows = _agent_candidate_rows(memory)
    promotion = memory.get("agentCandidatePromotion") or {}
    ideas = _candidate_ideas(memory)

    return {
        "symbol": sym,
        "duration": duration,
        "updatedAt": memory.get("updatedAt"),
        "header": _header_payload(memory, ideas, promotion, agent_rows),
        "summary": _summary_payload(memory, models),
        "trainingRules": _training_rules_payload(),
        "models": models,
        "agentCandidates": agent_rows,
        "sidebar": _sidebar_payload(memory, operators, ideas, promotion),
        "ingestionPath": _ingestion_path(memory, ideas, promotion),
        "operators": _operators_sidebar(operators),
        "memory": memory,
    }


def _header_payload(memory: dict, ideas: list, promotion: dict, agent_rows: list) -> dict[str, Any]:
    refresh = memory.get("refreshTask") or {}
    agent = memory.get("llmAgent") or {}
    library = memory.get("agentMinedFactorLibrary") or {}
    pending = sum(1 for row in agent_rows if row.get("validationStatusKey") == "pending_backtest")
    idea_count = len(ideas)
    library_pair_count = int(library.get("candidateTotal") or promotion.get("candidateCount") or 0)
    return {
        "localReplayStatus": _refresh_status_key(refresh, memory.get("source") or {}),
        "localReplayLabel": _refresh_status_label(refresh, memory.get("source") or {}),
        "agentIdeaCount": idea_count,
        "agentLibraryPairCount": library_pair_count,
        "agentCandidateCount": idea_count,
        "pendingVerificationCount": pending,
        "agentStatus": agent.get("status") or ("done" if agent.get("review") else "idle"),
        "agentModel": agent.get("model") or _configured_agent_model(),
        "agentReviewedAt": _agent_reviewed_at(memory),
        "memoryUpdatedAt": memory.get("updatedAt"),
    }


def _configured_agent_model() -> str:
    try:
        return resolved_siliconflow_model()
    except RuntimeError:
        return DEFAULT_SILICONFLOW_MODEL


def _summary_payload(memory: dict, models: list[dict]) -> dict[str, Any]:
    adaptive = memory.get("adaptiveLearning") or {}
    loss = memory.get("lossMemory") or {}
    promotion = memory.get("agentCandidatePromotion") or {}
    sample_count = int(adaptive.get("sampleCount") or loss.get("sampleCount") or 0)
    loss_count = int(loss.get("lossCount") or 0)
    win_count = max(sample_count - loss_count, 0) if sample_count else 0
    searching = sum(1 for row in models if row.get("searchStatus") in {"queued", "running"})
    candidate_records = sum(int(row.get("candidateLibraryTotal") or 0) for row in models)
    candidate_records += int(promotion.get("candidateCount") or 0)
    return {
        "overallAccuracy": adaptive.get("overallAccuracy"),
        "accuracyCaption": "基于已结算预测样本",
        "sampleCount": sample_count,
        "lossSampleCount": loss_count,
        "winSampleCount": win_count,
        "searchingCount": searching,
        "searchParallel": f"{searching} / {len(MODEL_FAMILIES)}",
        "candidateRecordCount": candidate_records,
        "candidatePending": int(promotion.get("candidateCount") or 0) - int(promotion.get("promoted") or 0),
        "candidateCompleted": int(promotion.get("promoted") or 0),
        "readyModelCount": sum(1 for row in models if row.get("cardState") == "ready"),
        "totalModelCount": len(models),
    }


def _training_rules_payload() -> dict[str, Any]:
    return {
        "text": "validation 与 test 胜率都必须 > 70%，全量 search grid 并行执行",
        "parallelWorkers": 10,
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
        "label": FAMILY_LABELS.get(family, family),
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
    if status.get("shadowPredictionReady"):
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
    if status.get("shadowPredictionReady"):
        return "就绪"
    if status.get("shadowPredictionBlockedReason") == "combo_snapshot_mismatch":
        return "组合变化"
    return "未就绪"


def _latest_candidate_label(progress: dict) -> str | None:
    latest = progress.get("latestCompleted")
    if not latest:
        return None
    cfg = latest.get("config") or {}
    status = latest.get("status") or "—"
    return f"{status} · w{cfg.get('featureWindow', '—')}"


def _agent_candidate_rows(memory: dict) -> list[dict[str, Any]]:
    ideas = _candidate_ideas(memory)
    promotion = memory.get("agentCandidatePromotion") or {}
    record_index = _promotion_record_index(promotion)
    agent_reviewed_at = _agent_reviewed_at(memory)
    rows = []
    matched_record_ids: set[str] = set()
    for index, idea in enumerate(ideas):
        record = _lookup_promotion_record(record_index, idea)
        row_id = str(record.get("factorName") or idea.get("nameHint") or idea.get("displayNameZh") or f"idea-{index}")
        if record:
            matched_record_ids.add(row_id)
        rows.append(_agent_candidate_row(idea, record, row_id=row_id, agent_reviewed_at=agent_reviewed_at))
    for record in promotion.get("records") or []:
        row_id = str(record.get("factorName") or record.get("formula") or "")
        if row_id in matched_record_ids:
            continue
        rows.append(_agent_candidate_row({}, record, row_id=row_id or "record", agent_reviewed_at=agent_reviewed_at))
    return rows


def _promotion_record_index(promotion: dict) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for item in promotion.get("records") or []:
        if not isinstance(item, dict):
            continue
        idea = item.get("idea") if isinstance(item.get("idea"), dict) else {}
        aliases = (
            item.get("factorName"),
            item.get("nameHint"),
            item.get("displayName"),
            idea.get("nameHint"),
            idea.get("displayNameZh"),
            idea.get("formulaHint"),
            item.get("formula"),
        )
        for alias in aliases:
            text = str(alias or "").strip()
            if text:
                index[text] = item
    return index


def _lookup_promotion_record(index: dict[str, dict[str, Any]], idea: dict[str, Any]) -> dict[str, Any]:
    for alias in (
        idea.get("nameHint"),
        idea.get("displayNameZh"),
        idea.get("formulaHint"),
    ):
        text = str(alias or "").strip()
        if text and text in index:
            return index[text]
    return {}


def _agent_candidate_row(
    idea: dict[str, Any],
    record: dict[str, Any],
    *,
    row_id: str,
    agent_reviewed_at: str | None,
) -> dict[str, Any]:
    idea_payload = idea if isinstance(idea, dict) else {}
    record_payload = record if isinstance(record, dict) else {}
    has_idea = bool(idea_payload)
    return {
        "id": row_id,
        "factorName": (
            record_payload.get("displayName")
            or idea_payload.get("displayNameZh")
            or idea_payload.get("nameHint")
            or record_payload.get("factorName")
            or "—"
        ),
        "operatorTrace": idea_payload.get("operatorTrace") or [],
        "formulaHint": record_payload.get("formula") or idea_payload.get("formulaHint"),
        "rationale": (
            idea_payload.get("rationaleZh")
            or idea_payload.get("rationale")
            or record_payload.get("error")
            or record_payload.get("reason")
        ),
        "validationStatus": _validation_status_label(record_payload, idea_payload if has_idea else {}),
        "validationStatusKey": _validation_status_key(record_payload, idea_payload if has_idea else {}),
        "source": "Agent",
        "createdAt": _record_evaluated_at(record_payload, agent_reviewed_at),
        "agentReviewedAt": agent_reviewed_at,
    }


def _record_evaluated_at(record: dict[str, Any], agent_reviewed_at: str | None) -> str | None:
    return str(record.get("seenAt") or record.get("evaluatedAt") or agent_reviewed_at or "") or None


def _agent_reviewed_at(memory: dict) -> str | None:
    agent = memory.get("llmAgent") or {}
    reviewed_at = agent.get("reviewedAt")
    if reviewed_at:
        return str(reviewed_at)
    refresh = memory.get("refreshTask") or {}
    if refresh.get("runAgent") and refresh.get("status") == "completed":
        updated = refresh.get("updatedAt")
        return str(updated) if updated else None
    return None


def _candidate_ideas(memory: dict) -> list[dict]:
    return list(memory.get("llmAgent", {}).get("review", {}).get("factorMiningPlan", {}).get("candidateFactorIdeas") or [])


def _validation_status_key(record: dict, idea: dict) -> str:
    status = str(record.get("status") or "")
    if status == "promoted":
        return "promoted"
    if status == "rejected_metrics":
        return "rejected_metrics"
    if status == "failed":
        return "failed"
    if status == "duplicate_existing":
        return "duplicate"
    if idea and not record:
        return "pending_backtest"
    if record:
        return "materialized"
    return "pending_backtest"


def _validation_status_label(record: dict, idea: dict) -> str:
    key = _validation_status_key(record, idea)
    labels = {
        "promoted": "已入库",
        "rejected_metrics": "入库：未达标",
        "failed": "物化失败",
        "duplicate": "已存在",
        "pending_backtest": "待回测",
        "materialized": "公式已物化",
    }
    return labels.get(key, "待回测")


def _sidebar_payload(memory: dict, operators: dict, ideas: list, promotion: dict) -> dict[str, Any]:
    source = memory.get("source") or {}
    mining = memory.get("factorMining") or {}
    monitoring = memory.get("monitoring") or {}
    issues = list(monitoring.get("issues") or [])
    failures = list(source.get("minedFrameFailures") or [])
    return {
        "closedLoop": {
            "agentIngested": int((memory.get("agentMinedFactorLibrary") or {}).get("total") or 0),
            "comboBacktest": int((memory.get("minedFactorLibrary") or {}).get("total") or 0),
            "candidatePromoted": int(promotion.get("promoted") or 0),
            "candidateTotal": int(promotion.get("candidateCount") or len(ideas)),
            "monitorAlerts": len(issues),
            "frameFailures": int(source.get("minedFrameFailureCount") or len(failures)),
            "replaySource": RANKING_SOURCE_LABELS.get(str(source.get("rankingRefreshSource") or ""), source.get("rankingRefreshSource") or "—"),
        },
        "alerts": _sidebar_alerts(memory, failures),
        "successPatterns": list(mining.get("successPatterns") or [])[:6],
        "forbiddenRegions": list(mining.get("forbiddenRegions") or [])[:6],
        "weights": memory.get("weights") or {},
        "lossMemoryStatus": (memory.get("lossMemory") or {}).get("status"),
        "operatorTotal": int(operators.get("total") or 0),
    }


def _sidebar_alerts(memory: dict, failures: list) -> list[dict[str, str]]:
    alerts = []
    loss_status = (memory.get("lossMemory") or {}).get("status")
    if loss_status in {"insufficient_loss_or_win_samples", "insufficient_settled_predictions", "no_separable_loss_pattern"}:
        alerts.append(
            {
                "level": "warn",
                "message": f"lossMemoryStatus: {_loss_status_label(loss_status)}",
            }
        )
    if failures:
        first = failures[0]
        alerts.append(
            {
                "level": "error",
                "message": f"上一轮物化失败: {first.get('reason') or first.get('error') or first}",
                "detail": first,
            }
        )
    agent = memory.get("llmAgent") or {}
    if agent.get("status") == "failed":
        alerts.append({"level": "error", "message": f"Agent 挖掘失败: {agent.get('error') or '查看日志'}"})
    return alerts


def _ingestion_path(memory: dict, ideas: list, promotion: dict) -> list[dict[str, Any]]:
    records = list(promotion.get("records") or [])
    materialized = sum(1 for item in records if item.get("formula") or item.get("status") not in {None, "", "failed"})
    backtested = sum(1 for item in records if item.get("metrics"))
    promoted = int(promotion.get("promoted") or 0)
    refresh = memory.get("refreshTask") or {}
    combo_state = "running" if refresh.get("status") in {"queued", "running"} else "done" if int((memory.get("minedFactorLibrary") or {}).get("total") or 0) > 0 else "pending"
    return [
        {"key": "ideas", "label": "候选想法", "state": "done" if ideas else "pending", "detail": f"{len(ideas)}项"},
        {"key": "materialize", "label": "公式物化", "state": _step_state(materialized, len(ideas)), "detail": f"{materialized}/{max(len(ideas), len(records)) or 0}"},
        {"key": "backtest", "label": "单因子回测", "state": _step_state(backtested, materialized or len(records)), "detail": f"{backtested}/{materialized or len(records) or 0}"},
        {"key": "ingest", "label": "达标入库", "state": "done" if promoted else ("error" if records and not promoted else "pending"), "detail": "样本不足" if records and not promoted else f"{promoted}项"},
        {"key": "combo", "label": "组合搜索", "state": combo_state, "detail": "进行中" if combo_state == "running" else ("完成" if combo_state == "done" else "等待")},
    ]


def _step_state(done: int, total: int) -> str:
    if total <= 0:
        return "pending"
    if done >= total:
        return "done"
    if done > 0:
        return "running"
    return "pending"


def _operators_sidebar(operators: dict) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {}
    for item in operators.get("operators") or []:
        category = str(item.get("category") or "other")
        label = OPERATOR_CATEGORY_LABELS.get(category, category)
        grouped.setdefault(label, []).append(str(item.get("name") or ""))
    preview = []
    for label, names in grouped.items():
        preview.append({"category": label, "names": names[:6]})
    return {"total": operators.get("total") or 0, "preview": preview}


def _refresh_status_key(task: dict, source: dict) -> str:
    status = task.get("status") or source.get("status")
    if status == "completed":
        return "done"
    if status in {"queued", "running"}:
        return "running"
    if status == "failed":
        return "failed"
    return "idle"


def _refresh_status_label(task: dict, source: dict) -> str:
    key = _refresh_status_key(task, source)
    labels = {
        "done": "本地复盘完成",
        "running": "本地复盘中",
        "failed": "本地复盘失败",
        "idle": "待本地复盘",
    }
    return labels.get(key, "—")


def _loss_status_label(status: str) -> str:
    labels = {
        "insufficient_loss_or_win_samples": "样本偏少, 权重调整仅供观察",
        "insufficient_settled_predictions": "结算样本不足",
        "no_separable_loss_pattern": "暂无可分离亏损模式",
    }
    return labels.get(status, status)
