from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.ensemble_judge_service import (
    confirm_ensemble_stage,
    ensemble_ranking,
    ensemble_status,
    refresh_ensemble_judge,
)

router = APIRouter(prefix="/api/ensemble", tags=["ensemble"])


class ConfirmStagePayload(BaseModel):
    symbol: str = Field(min_length=6)
    duration: str
    stage: str


@router.get("/status")
def read_ensemble_status(symbol: str = Query(..., min_length=6), duration: str = Query("10m")) -> dict:
    return ensemble_status(symbol, duration)


@router.get("/ranking")
def read_ensemble_ranking(symbol: str = Query(..., min_length=6), duration: str = Query("10m")) -> dict:
    return ensemble_ranking(symbol, duration)


@router.post("/refresh")
def refresh_ensemble(symbol: str = Query(..., min_length=6), duration: str = Query("10m")) -> dict:
    return refresh_ensemble_judge(symbol, duration)


@router.post("/confirm-stage")
def confirm_stage(payload: ConfirmStagePayload) -> dict:
    try:
        return confirm_ensemble_stage(payload.symbol, payload.duration, payload.stage)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
