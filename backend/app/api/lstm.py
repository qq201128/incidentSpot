from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api.models import model_candidate_search
from app.services.experiment_profiles import normalize_experiment_profile
from app.services.lstm_candidate_search import DEFAULT_PARALLEL_WORKERS
from app.services.lstm_prediction_service import lstm_model_status, predict_lstm_signal
from app.services.lstm_shadow_learning import lstm_shadow_learning_summary

router = APIRouter(prefix="/api/lstm", tags=["lstm"])


@router.get("/status")
def lstm_status(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
) -> dict:
    try:
        status = lstm_model_status(symbol.upper(), duration)
        return {**status, "shadow": lstm_shadow_learning_summary(symbol.upper(), duration)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/train")
def lstm_train(
    *,
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
        _reject_direct_train_overrides(
            feature_window=feature_window,
            epochs=epochs,
            batch_size=batch_size,
            min_samples=min_samples,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            num_layers=num_layers,
            min_move_bps=min_move_bps,
        )
        return model_candidate_search(
            "lstm",
            symbol=symbol,
            duration=duration,
            profile=profile,
            reset_history=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _reject_direct_train_overrides(**values: object | None) -> None:
    normalize_experiment_profile(str(values.pop("profile", "full")))
    if any(value is not None for value in values.values()):
        raise ValueError("direct in-process LSTM training is disabled; enqueue candidate-search jobs instead")


@router.post("/candidate-search")
def lstm_candidate_search(
    *,
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    profile: str = Query("full"),
    reset_history: bool = Query(False, alias="resetHistory"),
    parallel_workers: int = Query(DEFAULT_PARALLEL_WORKERS, alias="parallelWorkers", ge=1),
) -> dict:
    try:
        normalize_experiment_profile(profile)
        return model_candidate_search(
            "lstm",
            symbol=symbol,
            duration=duration,
            profile=profile,
            reset_history=reset_history,
            parallel_workers=parallel_workers,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/predict")
def lstm_predict(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    entry_open_time: int | None = Query(None, alias="entryOpenTime"),
) -> dict:
    try:
        return predict_lstm_signal(symbol.upper(), duration, entry_open_time=entry_open_time)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
