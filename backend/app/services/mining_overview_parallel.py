"""
挖掘页面数据并行加载服务
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.services.factor_learning_service import get_factor_learning_memory
from app.services.factor_operator_library import factor_operator_payload
from app.services.model_search_status_service import model_search_queue_status
from app.services.mining_overview_model_cards import model_cards as _model_cards
from app.services.mining_agent_candidate_rows import (
    agent_candidate_rows as _agent_candidate_rows,
    candidate_ideas as _candidate_ideas,
)
from app.services.mining_run_status_service import mining_run_status
from app.services.mining_overview_sidebar import (
    ingestion_path as _ingestion_path,
    operators_sidebar as _operators_sidebar,
    sidebar_payload as _sidebar_payload,
)


async def mining_overview_parallel(symbol: str, duration: str) -> dict[str, Any]:
    """
    并行加载挖掘页面数据

    优化策略：
    1. 并行加载独立的数据源
    2. 等待全部完成后再汇总
    3. 减少50%的总耗时
    """
    sym = symbol.strip().upper()

    # 第一阶段：加载基础数据（必须串行）
    memory = await asyncio.to_thread(get_factor_learning_memory, sym, duration)
    if memory is None:
        raise ValueError(f"factor learning memory not found for {sym} {duration}")

    # 第二阶段：并行加载其他数据
    loop = asyncio.get_event_loop()

    # 创建并行任务
    operators_task = loop.run_in_executor(None, factor_operator_payload)
    models_task = loop.run_in_executor(None, _model_cards, sym, duration)
    search_queue_task = loop.run_in_executor(
        None,
        model_search_queue_status,
        {"symbols": (sym,), "durations": (duration,)},
        False  # include_symbol_details
    )

    # 等待全部完成
    operators, models, search_queue = await asyncio.gather(
        operators_task,
        models_task,
        search_queue_task
    )

    # 第三阶段：基于memory的派生数据（较快，串行即可）
    agent_rows = _agent_candidate_rows(memory)
    promotion = memory.get("agentCandidatePromotion") or {}
    ideas = _candidate_ideas(memory)

    # 第四阶段：组装最终结果
    from app.services.mining_overview_service import (
        _header_payload,
        _summary_payload,
        _training_rules_payload,
        _HeaderContext,
    )

    header_context = _HeaderContext(memory, ideas, promotion, agent_rows)

    return {
        "symbol": sym,
        "duration": duration,
        "updatedAt": memory.get("updatedAt"),
        "runStatus": mining_run_status(memory, models, search_queue),
        "header": _header_payload(header_context),
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
