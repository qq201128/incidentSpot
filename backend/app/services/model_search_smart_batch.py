"""
智能搜索批量入队服务
"""
from __future__ import annotations

from typing import Any

from app.services.model_search_resource import ModelSearchResourceConfig


def enqueue_smart_batch_search(
    symbols: tuple[str, ...],
    durations: tuple[str, ...],
    families: tuple[str, ...],
    mode: str = "balanced",
    resource: ModelSearchResourceConfig | None = None,
) -> dict[str, Any]:
    """
    批量入队智能模型搜索任务

    Args:
        symbols: 交易对列表
        durations: 周期列表
        families: 模型族列表
        mode: 搜索模式 (fast / balanced / exhaustive)
        resource: 资源配置

    Returns:
        队列状态
    """
    jobs = []

    for family in families:
        for symbol in symbols:
            for duration in durations:
                job = {
                    "family": family,
                    "symbol": symbol,
                    "duration": duration,
                    "mode": mode,
                    "type": "smart_batch_search",
                    "status": "queued",
                    "resource": resource.dict() if resource else None,
                }
                jobs.append(job)

    # TODO: 实际实现应该将任务写入队列（Redis / 数据库）
    # 并根据优先级和速度进行调度
    return {
        "jobs": jobs,
        "total": len(jobs),
        "message": f"Enqueued {len(jobs)} smart search tasks (mode={mode})",
        "breakdown": {
            "families": len(families),
            "symbols": len(symbols),
            "durations": len(durations),
        },
    }
