from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.lstm_config import LstmTrainingConfig
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
    feature_window: int = Query(64, alias="featureWindow"),
    epochs: int = Query(20),
    min_samples: int = Query(120, alias="minSamples"),
) -> dict:
    try:
        config = LstmTrainingConfig(
            symbol=symbol.upper(),
            duration=duration,
            feature_window=feature_window,
            epochs=epochs,
            min_samples=min_samples,
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
