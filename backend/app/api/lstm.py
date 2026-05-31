from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.services.experiment_profiles import lstm_training_config_for_profile, normalize_experiment_profile
from app.services.lstm_candidate_progress import finish_lstm_candidate_progress, queue_lstm_candidate_progress
from app.services.lstm_candidate_retry import LstmCandidateRetryConfig, run_lstm_candidate_retry
from app.services.lstm_candidate_search import (
    DEFAULT_PARALLEL_WORKERS,
    LstmCandidateSearchConfig,
    search_space_size,
)
from app.services.lstm_prediction_service import lstm_model_status, predict_lstm_signal
from app.services.lstm_shadow_learning import lstm_shadow_learning_summary
from app.services.lstm_training_service import train_lstm_model

router = APIRouter(prefix="/api/lstm", tags=["lstm"])
logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class CandidateSearchJob:
    symbol: str
    duration: str
    profile: str
    reset_history: bool
    parallel_workers: int


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


@router.post("/candidate-search")
def lstm_candidate_search(
    background_tasks: BackgroundTasks,
    *,
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    profile: str = Query("full"),
    reset_history: bool = Query(False, alias="resetHistory"),
    parallel_workers: int = Query(DEFAULT_PARALLEL_WORKERS, alias="parallelWorkers", ge=1),
) -> dict:
    try:
        sym_u = symbol.upper()
        selected_profile = normalize_experiment_profile(profile)
        reset_requested = reset_history is True
        search_config = LstmCandidateSearchConfig(parallel_workers=parallel_workers)
        search_total = search_space_size(search_config)
        job = CandidateSearchJob(
            symbol=sym_u,
            duration=duration,
            profile=selected_profile,
            reset_history=reset_requested,
            parallel_workers=search_config.parallel_workers,
        )
        queued = queue_lstm_candidate_progress(
            symbol=sym_u,
            duration=duration,
            profile=selected_profile,
            total=search_total,
            search_space_total=search_total,
            parallel_workers=search_config.parallel_workers,
        )
        background_tasks.add_task(_background_lstm_candidate_search, job)
        status = lstm_model_status(sym_u, duration)
        return {
            **status,
            "candidateSearchProgress": status["candidateSearchProgress"] or queued,
            "message": "LSTM候选搜索已排队。",
        }
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


def _background_lstm_candidate_search(job: CandidateSearchJob) -> None:
    search_config = LstmCandidateSearchConfig(parallel_workers=job.parallel_workers)
    config = LstmCandidateRetryConfig(
        symbols=(job.symbol,),
        durations=(job.duration,),
        profile=job.profile,
        search=search_config,
        manual_trigger=True,
        reset_history=job.reset_history,
    )
    try:
        report = run_lstm_candidate_retry(config)
        finish_lstm_candidate_progress(
            symbol=job.symbol,
            duration=job.duration,
            status=str(report.get("status") or "failed"),
        )
    except Exception:
        finish_lstm_candidate_progress(symbol=job.symbol, duration=job.duration, status="failed")
        logger.exception("lstm candidate search failed: %s %s", job.symbol, job.duration)
        raise
