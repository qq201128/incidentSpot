from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.services.experiment_profiles import normalize_experiment_profile
from app.services.model_family_candidate_search_service import (
    ModelCandidateSearchConfig,
    model_training_config_for_profile,
    queue_total_for_family,
    run_model_candidate_search,
)
from app.services.model_family_candidates import (
    finish_model_candidate_progress,
    queue_model_candidate_progress,
)
from app.services.model_family_config import normalize_model_family
from app.services.model_family_prediction_service import predict_model_family_signal
from app.services.model_family_status_service import model_family_status
from app.services.model_family_training_service import train_model_family

router = APIRouter(prefix="/api/models", tags=["models"])
logger = logging.getLogger("uvicorn.error")


@router.get("/{family}/status")
def model_status(family: str, symbol: str = Query(..., min_length=6), duration: str = Query("10m")) -> dict:
    try:
        return model_family_status(family, symbol.upper(), duration)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{family}/train")
def model_train(
    family: str,
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    profile: str = Query("full"),
    feature_window: int | None = Query(None, alias="featureWindow"),
    epochs: int | None = Query(None),
    batch_size: int | None = Query(None, alias="batchSize"),
    min_samples: int | None = Query(None, alias="minSamples"),
    learning_rate: float | None = Query(None, alias="learningRate"),
    hidden_size: int | None = Query(None, alias="hiddenSize"),
    num_layers: int | None = Query(None, alias="numLayers"),
    min_move_bps: float | None = Query(None, alias="minMoveBps"),
) -> dict:
    try:
        config = model_training_config_for_profile(
            family,
            symbol.upper(),
            duration,
            normalize_experiment_profile(profile),
            feature_window=feature_window,
            epochs=epochs,
            batch_size=batch_size,
            min_samples=min_samples,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            num_layers=num_layers,
            min_move_bps=min_move_bps,
        )
        return train_model_family(config, publish_initial_baseline=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{family}/candidate-search")
def model_candidate_search(
    background_tasks: BackgroundTasks,
    family: str,
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    profile: str = Query("full"),
    parallel_workers: int = Query(10, alias="parallelWorkers", ge=1),
    reset_history: bool = Query(False, alias="resetHistory"),
) -> dict:
    try:
        selected = normalize_model_family(family)
        sym = symbol.upper()
        selected_profile = normalize_experiment_profile(profile)
        total = queue_total_for_family(selected)
        queued = queue_model_candidate_progress(
            selected,
            symbol=sym,
            duration=duration,
            profile=selected_profile,
            total=total,
            parallel_workers=parallel_workers,
        )
        background_tasks.add_task(
            _background_model_candidate_search,
            selected,
            sym,
            duration,
            selected_profile,
            parallel_workers,
            reset_history,
        )
        status = model_family_status(selected, sym, duration)
        return {
            **status,
            "candidateSearchProgress": status["candidateSearchProgress"] or queued,
            "message": f"{selected}候选搜索已排队。",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{family}/predict")
def model_predict(
    family: str,
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    entry_open_time: int | None = Query(None, alias="entryOpenTime"),
) -> dict:
    try:
        return predict_model_family_signal(family, symbol.upper(), duration, entry_open_time=entry_open_time)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _background_model_candidate_search(
    family: str,
    symbol: str,
    duration: str,
    profile: str,
    parallel_workers: int,
    reset_history: bool,
) -> None:
    try:
        report = run_model_candidate_search(
            ModelCandidateSearchConfig(family, symbol, duration, profile, parallel_workers, reset_history)
        )
        if str(report.get("status") or "") == "skipped":
            finish_model_candidate_progress(family, symbol=symbol, duration=duration, status="skipped")
    except Exception:
        finish_model_candidate_progress(family, symbol=symbol, duration=duration, status="failed")
        logger.exception("model candidate search failed: family=%s symbol=%s duration=%s", family, symbol, duration)
        raise
