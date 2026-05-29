from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.experiment_profiles import normalize_experiment_profile
from app.services.model_family_candidate_search_service import (
    model_training_config_for_profile,
)
from app.services.model_family_config import normalize_model_family
from app.services.model_family_prediction_service import predict_model_family_signal
from app.services.model_family_status_service import model_family_status
from app.services.model_family_training_service import train_model_family
from app.services.model_search_job_store import enqueue_model_search_jobs
from app.services.model_search_status_service import model_search_status_with_lifecycle
from app.services.runtime_symbols import parse_symbol_csv

router = APIRouter(prefix="/api/models", tags=["models"])


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
    family: str,
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    profile: str = Query("full"),
    reset_history: bool = Query(False, alias="resetHistory"),
) -> dict:
    try:
        selected = normalize_model_family(family)
        sym = symbol.upper()
        selected_profile = normalize_experiment_profile(profile)
        queued_job = enqueue_model_search_jobs(
            symbols=(sym,),
            durations=(duration,),
            families=(selected,),
            profile=selected_profile,
            reset_existing=reset_history,
        )
        status = model_family_status(selected, sym, duration)
        return {
            **status,
            "modelSearchJob": queued_job["jobs"][0],
            "modelSearchQueue": queued_job,
            "message": f"{selected}候选搜索已入队，等待 model search worker 执行。",
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


@router.get("/search/jobs/status")
def model_search_jobs_status(
    symbols: str | None = Query(None),
    duration: str | None = Query(None),
    family: str | None = Query(None),
    status: str | None = Query(None),
) -> dict:
    filters = {
        "symbols": parse_symbol_csv(symbols) if symbols else (),
        "durations": (duration,) if duration else (),
        "families": (normalize_model_family(family),) if family else (),
        "statuses": (status,) if status else (),
    }
    return model_search_status_with_lifecycle(filters)

