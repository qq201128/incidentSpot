from __future__ import annotations

from typing import Any

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


def sidebar_payload(memory: dict, operators: dict, ideas: list, promotion: dict) -> dict[str, Any]:
    source = memory.get("source") or {}
    mining = memory.get("factorMining") or {}
    monitoring = memory.get("monitoring") or {}
    issues = list(monitoring.get("issues") or [])
    failures = list(source.get("minedFrameFailures") or [])
    return {
        "closedLoop": closed_loop_payload(memory, promotion, ideas, source, issues, failures),
        "alerts": sidebar_alerts(memory, failures),
        "successPatterns": list(mining.get("successPatterns") or [])[:6],
        "forbiddenRegions": list(mining.get("forbiddenRegions") or [])[:6],
        "weights": memory.get("weights") or {},
        "lossMemoryStatus": (memory.get("lossMemory") or {}).get("status"),
        "operatorTotal": int(operators.get("total") or 0),
    }


def closed_loop_payload(
    memory: dict,
    promotion: dict,
    ideas: list,
    source: dict,
    issues: list,
    failures: list,
) -> dict[str, Any]:
    replay_source = str(source.get("rankingRefreshSource") or "")
    return {
        "agentIngested": int((memory.get("agentMinedFactorLibrary") or {}).get("total") or 0),
        "comboBacktest": int((memory.get("minedFactorLibrary") or {}).get("total") or 0),
        "candidatePromoted": int(promotion.get("promoted") or 0),
        "candidateTotal": int(promotion.get("candidateCount") or len(ideas)),
        "monitorAlerts": len(issues),
        "frameFailures": int(source.get("minedFrameFailureCount") or len(failures)),
        "replaySource": RANKING_SOURCE_LABELS.get(replay_source, source.get("rankingRefreshSource") or "—"),
    }


def sidebar_alerts(memory: dict, failures: list) -> list[dict[str, str]]:
    alerts = []
    loss_status = (memory.get("lossMemory") or {}).get("status")
    if loss_status in {"insufficient_loss_or_win_samples", "insufficient_settled_predictions", "no_separable_loss_pattern"}:
        alerts.append({"level": "warn", "message": f"lossMemoryStatus: {loss_status_label(loss_status)}"})
    if failures:
        first = failures[0]
        alerts.append({
            "level": "error",
            "message": f"上一轮物化失败: {first.get('reason') or first.get('error') or first}",
            "detail": first,
        })
    agent = memory.get("llmAgent") or {}
    if agent.get("status") == "failed":
        alerts.append({"level": "error", "message": f"Agent 挖掘失败: {agent.get('error') or '查看日志'}"})
    return alerts


def ingestion_path(memory: dict, ideas: list, promotion: dict) -> list[dict[str, Any]]:
    records = list(promotion.get("records") or [])
    materialized = sum(1 for item in records if item.get("formula") or item.get("status") not in {None, "", "failed"})
    backtested = sum(1 for item in records if item.get("metrics"))
    promoted = int(promotion.get("promoted") or 0)
    combo_state = combo_step_state(memory)
    return [
        {"key": "ideas", "label": "候选想法", "state": "done" if ideas else "pending", "detail": f"{len(ideas)}项"},
        {"key": "materialize", "label": "公式物化", "state": step_state(materialized, len(ideas)), "detail": f"{materialized}/{max(len(ideas), len(records)) or 0}"},
        {"key": "backtest", "label": "单因子回测", "state": step_state(backtested, materialized or len(records)), "detail": f"{backtested}/{materialized or len(records) or 0}"},
        {"key": "ingest", "label": "达标入库", "state": "done" if promoted else ("error" if records and not promoted else "pending"), "detail": "样本不足" if records and not promoted else f"{promoted}项"},
        {"key": "combo", "label": "组合搜索", "state": combo_state, "detail": "进行中" if combo_state == "running" else ("完成" if combo_state == "done" else "等待")},
    ]


def combo_step_state(memory: dict) -> str:
    refresh = memory.get("refreshTask") or {}
    if refresh.get("status") in {"queued", "running"}:
        return "running"
    if int((memory.get("minedFactorLibrary") or {}).get("total") or 0) > 0:
        return "done"
    return "pending"


def step_state(done: int, total: int) -> str:
    if total <= 0:
        return "pending"
    if done >= total:
        return "done"
    if done > 0:
        return "running"
    return "pending"


def operators_sidebar(operators: dict) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {}
    for item in operators.get("operators") or []:
        category = str(item.get("category") or "other")
        label = OPERATOR_CATEGORY_LABELS.get(category, category)
        grouped.setdefault(label, []).append(str(item.get("name") or ""))
    preview = [{"category": label, "names": names[:6]} for label, names in grouped.items()]
    return {"total": operators.get("total") or 0, "preview": preview}


def refresh_status_key(task: dict, source: dict) -> str:
    status = task.get("status") or source.get("status")
    if status == "completed":
        return "done"
    if status in {"queued", "running"}:
        return "running"
    if status == "failed":
        return "failed"
    return "idle"


def refresh_status_label(task: dict, source: dict) -> str:
    labels = {
        "done": "本地复盘完成",
        "running": "本地复盘中",
        "failed": "本地复盘失败",
        "idle": "待本地复盘",
    }
    return labels.get(refresh_status_key(task, source), "—")


def loss_status_label(status: str) -> str:
    labels = {
        "insufficient_loss_or_win_samples": "样本偏少, 权重调整仅供观察",
        "insufficient_settled_predictions": "结算样本不足",
        "no_separable_loss_pattern": "暂无可分离亏损模式",
    }
    return labels.get(status, status)
