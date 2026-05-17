from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.experiment_profiles import lstm_training_config_for_profile, normalize_experiment_profile
from app.services.lstm_prediction_service import lstm_model_status, predict_lstm_signal
from app.services.lstm_shadow_learning import lstm_shadow_learning_summary
from app.services.lstm_training_service import train_lstm_model

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
        selected_profile = normalize_experiment_profile(profile)
        config = lstm_training_config_for_profile(
            symbol.upper(),
            duration,
            selected_profile,
            feature_window=feature_window,
            epochs=epochs,
            batch_size=batch_size,
            min_samples=min_samples,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            num_layers=num_layers,
            min_move_bps=min_move_bps,
        )
        return train_lstm_model(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
