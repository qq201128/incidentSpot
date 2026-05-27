from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query

from app.services.mining_overview_cache import get_cached_mining_overview
from app.services.mining_overview_service import mining_overview

router = APIRouter(prefix="/api/mining", tags=["mining"])


@router.get("/overview")
def get_mining_overview(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    fresh: bool = Query(False, description="Skip overview cache (use while tasks are running)"),
) -> dict:
    try:
        sym = symbol.upper()
        if fresh:
            return mining_overview(sym, duration)
        return get_cached_mining_overview(sym, duration, build=mining_overview)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"因子学习记忆文件正在写入，请稍后重试：{exc}",
        ) from exc
    except ValueError as exc:
        detail = str(exc)
        status = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status, detail=detail) from exc
