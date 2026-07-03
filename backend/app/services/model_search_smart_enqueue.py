"""
智能搜索入队服务 - 单个模型
"""
from __future__ import annotations

import json
from typing import Any

from app.services.model_search_resource import ModelSearchResourceConfig


def enqueue_smart_model_search(
    family: str,
    symbol: str,
    duration: str,
    mode: str = "balanced",
    priority: int = 50,
    resource: ModelSearchResourceConfig | None = None,
) -> dict[str, Any]:
    """
    入队智能模型搜索任务

    Args:
        family: 模型族
        symbol: 交易对
        duration: 周期
        mode: 搜索模式 (fast / balanced / exhaustive)
        priority: 优先级
        resource: 资源配置

    Returns:
        队列状态
    """
    # 构建任务描述
    job = {
        "family": family,
        "symbol": symbol,
        "duration": duration,
        "mode": mode,
        "priority": priority,
        "type": "smart_search",
        "status": "queued",
        "resource": resource.dict() if resource else None,
    }

    # TODO: 实际实现应该将任务写入队列（Redis / 数据库）
    # 这里简化为直接返回任务信息
    return {
        "jobs": [job],
        "total": 1,
        "message": f"Enqueued smart search for {family} {symbol} {duration} (mode={mode})",
    }
