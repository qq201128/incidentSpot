from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.model_training_service import (
    activate_model_version,
    model_dashboard,
    start_training_task,
)


router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
def read_models() -> dict:
    return model_dashboard()


@router.post("/train")
async def trigger_training() -> dict:
    try:
        state = start_training_task()
        return {"accepted": True, "schedule": state}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{model_key}/versions/{version_id}/activate")
def activate_model(model_key: str, version_id: str) -> dict:
    try:
        return activate_model_version(model_key, version_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
