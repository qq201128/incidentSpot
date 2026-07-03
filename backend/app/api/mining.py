from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query

from app.services.mining_overview_cache import get_cached_mining_overview
from app.services.mining_overview_service import mining_overview

router = APIRouter(prefix="/api/mining", tags=["mining"])


@router.get("/overview")
async def get_mining_overview(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    fresh: bool = Query(False, description="Skip overview cache (use while tasks are running)"),
    parallel: bool = Query(True, description="Use parallel loading"),
) -> dict:
    try:
        sym = symbol.upper()

        # 强制刷新：使用并行加载（不走缓存）
        if fresh:
            if parallel:
                from app.services.mining_overview_parallel import mining_overview_parallel
                return await mining_overview_parallel(sym, duration)
            else:
                return mining_overview(sym, duration)

        # 正常请求：先检查缓存
        cached = get_cached_mining_overview(sym, duration, build=mining_overview)

        # 如果是缓存命中，直接返回
        if cached.get("cache", {}).get("hit"):
            return cached

        # 缓存未命中：使用并行加载
        if parallel:
            from app.services.mining_overview_parallel import mining_overview_parallel
            data = await mining_overview_parallel(sym, duration)
            return {**data, "cache": {"hit": False, "ageSeconds": 0.0}}

        return cached

    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"因子学习记忆文件正在写入，请稍后重试：{exc}",
        ) from exc
    except ValueError as exc:
        detail = str(exc)
        status = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status, detail=detail) from exc
