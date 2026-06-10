from __future__ import annotations

import gc
import os
import json
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from app.services.lstm_feature_builder import LstmDataset, build_lstm_training_dataset
from app.services.model_family_candidate_process import train_candidate_in_process
from app.services.model_family_candidates import model_search_key, record_model_candidate
from app.services.model_family_config import ModelFamilyTrainingConfig, TORCH_MODEL_FAMILIES
from app.services.model_family_training_service import train_model_family

REAL_TRAIN_MODEL_FAMILY = train_model_family
PROCESS_EXECUTOR_FAMILIES = frozenset({
    "lstm",
    "gru",
    "cnn",
    "transformer",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "catboost",
    "logistic_elasticnet",
    "svm",
    "bayesian",
    "knn",
    "rl_strategy",
})
PROCESS_WORKERS_ENV = "MODEL_FAMILY_PROCESS_WORKERS"
XGBOOST_PROCESS_WORKERS_ENV = "MODEL_FAMILY_XGBOOST_PROCESS_WORKERS"
TORCH_JOBS_ENV = "MODEL_FAMILY_TORCH_JOBS"


@dataclass(frozen=True)
class CandidateTrainingResult:
    config: ModelFamilyTrainingConfig
    report: dict[str, Any]


@dataclass
class CandidateDatasetCache:
    datasets: dict[tuple, LstmDataset] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    def build(self, config: ModelFamilyTrainingConfig) -> LstmDataset:
        key = _dataset_cache_key(config)
        with self.lock:
            if key not in self.datasets:
                self.datasets[key] = build_lstm_training_dataset(config)
            return self.datasets[key]

    def clear(self) -> None:
        with self.lock:
            self.datasets.clear()
        gc.collect()


def train_candidate_reports(
    configs: list[ModelFamilyTrainingConfig],
    profile: str,
    workers: int,
    dataset_builder,
    *,
    stage: str,
    record_configs: list[ModelFamilyTrainingConfig] | None = None,
) -> Iterator[CandidateTrainingResult]:
    pairs = _training_pairs(configs, record_configs)
    if workers <= 1 or len(pairs) <= 1:
        yield from _train_candidate_reports_serial(pairs, profile, dataset_builder, stage)
        return
    if _uses_process_executor(configs, dataset_builder):
        count = _process_worker_count(workers, family=configs[0].family)
        yield from _train_candidate_reports_in_processes(pairs, profile, count, stage)
        return
    yield from _train_candidate_reports_in_threads(
        pairs,
        profile,
        _thread_worker_count(configs, workers),
        dataset_builder,
        stage,
    )


def _train_candidate_reports_serial(
    pairs: list[tuple[ModelFamilyTrainingConfig, ModelFamilyTrainingConfig]],
    profile: str,
    dataset_builder,
    stage: str,
) -> Iterator[CandidateTrainingResult]:
    for train_config, record_config in pairs:
        report = train_candidate(train_config, profile, dataset_builder, stage=stage, record_config=record_config)
        yield CandidateTrainingResult(record_config, report)


def train_candidate(
    train_config: ModelFamilyTrainingConfig,
    profile: str,
    dataset_builder,
    *,
    stage: str,
    record_config: ModelFamilyTrainingConfig,
) -> dict[str, Any]:
    try:
        _candidate_event("candidate_start", train_config, stage)
        report = train_model_family(
            train_config,
            dataset_builder=dataset_builder,
            publish_shadow_active=False,
            publish_trade_active=False,
            write_attempt=False,
            persist_artifacts=False,
            evaluate_test=False,
        )
        _candidate_event("candidate_finish", train_config, stage, status=str(report.get("status") or "unknown"))
    except Exception as exc:
        _candidate_event("candidate_failed", train_config, stage, status=type(exc).__name__)
        report = _failed_report(record_config, profile, exc)
    finally:
        _clear_dataset_builder_cache(dataset_builder)
    report = _stage_report(record_config, profile, report, stage)
    record_model_candidate(record_config, profile, report)
    return report


def _train_candidate_reports_in_threads(
    pairs: list[tuple[ModelFamilyTrainingConfig, ModelFamilyTrainingConfig]],
    profile: str,
    workers: int,
    dataset_builder,
    stage: str,
) -> Iterator[CandidateTrainingResult]:
    max_workers = min(int(workers), len(pairs))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(train_candidate, train_config, profile, dataset_builder, stage=stage, record_config=record_config): record_config
            for train_config, record_config in pairs
        }
        for future in as_completed(future_map):
            yield CandidateTrainingResult(future_map[future], future.result())


def _train_candidate_reports_in_processes(
    pairs: list[tuple[ModelFamilyTrainingConfig, ModelFamilyTrainingConfig]],
    profile: str,
    workers: int,
    stage: str,
) -> Iterator[CandidateTrainingResult]:
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(train_candidate_in_process, train_config, profile): record_config for train_config, record_config in pairs}
        for future in as_completed(future_map):
            record_config = future_map[future]
            report = _stage_report(record_config, profile, future.result(), stage)
            record_model_candidate(record_config, profile, report)
            yield CandidateTrainingResult(record_config, report)


def _training_pairs(configs, record_configs) -> list[tuple[ModelFamilyTrainingConfig, ModelFamilyTrainingConfig]]:
    selected_records = record_configs or configs
    if len(configs) != len(selected_records):
        raise ValueError("record_configs length must match configs length")
    return list(zip(configs, selected_records))


def _stage_report(config: ModelFamilyTrainingConfig, profile: str, report: dict[str, Any], stage: str) -> dict[str, Any]:
    return {**report, "searchKey": model_search_key(config, profile), "searchStage": stage}


def _uses_process_executor(configs: list[ModelFamilyTrainingConfig], dataset_builder) -> bool:
    return (
        bool(configs)
        and configs[0].family in PROCESS_EXECUTOR_FAMILIES
        and _is_default_dataset_builder(dataset_builder)
        and train_model_family is REAL_TRAIN_MODEL_FAMILY
    )


def _process_worker_count(max_workers: int, *, family: str | None = None) -> int:
    raw = os.environ.get(PROCESS_WORKERS_ENV)
    source = PROCESS_WORKERS_ENV
    if raw is None and (family is None or family == "xgboost"):
        raw = os.environ.get(XGBOOST_PROCESS_WORKERS_ENV)
        source = XGBOOST_PROCESS_WORKERS_ENV
    if raw is None:
        return max_workers
    selected = int(raw)
    if selected <= 0:
        raise ValueError(f"{source} must be positive")
    return min(max_workers, selected)


def _thread_worker_count(configs: list[ModelFamilyTrainingConfig], max_workers: int) -> int:
    if not configs or configs[0].family not in TORCH_MODEL_FAMILIES:
        return max_workers
    raw = os.environ.get(TORCH_JOBS_ENV)
    if raw is None:
        return max_workers
    selected = int(raw)
    if selected <= 0:
        raise ValueError(f"{TORCH_JOBS_ENV} must be positive")
    return min(max_workers, selected)


def _is_default_dataset_builder(dataset_builder) -> bool:
    owner = getattr(dataset_builder, "__self__", None)
    return dataset_builder is build_lstm_training_dataset or isinstance(owner, CandidateDatasetCache)


def _clear_dataset_builder_cache(dataset_builder) -> None:
    owner = getattr(dataset_builder, "__self__", None)
    if isinstance(owner, CandidateDatasetCache):
        owner.clear()


def _failed_report(config: ModelFamilyTrainingConfig, profile: str, exc: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "candidateStatus": "failed",
        "modelFamily": config.family,
        "symbol": config.symbol,
        "duration": config.duration,
        "modelVersion": None,
        "searchKey": model_search_key(config, profile),
        "validationFailureReason": str(exc),
        "failure": {
            "stage": "candidate_training",
            "exceptionType": type(exc).__name__,
            "error": str(exc),
        },
    }


def _candidate_event(event: str, config: ModelFamilyTrainingConfig, stage: str, *, status: str | None = None) -> None:
    payload = {
        "event": event,
        "family": config.family,
        "symbol": config.symbol,
        "duration": config.duration,
        "stage": stage,
        "featureWindow": config.feature_window,
        "minMoveBps": config.min_move_bps,
        "params": config.params,
    }
    print(json.dumps(payload if status is None else {**payload, "status": status}, ensure_ascii=False), flush=True)


def _dataset_cache_key(config: ModelFamilyTrainingConfig) -> tuple:
    return (
        config.symbol,
        config.duration,
        config.feature_window,
        config.horizon_minutes,
        config.min_samples,
        config.train_ratio,
        config.val_ratio,
        config.min_move_bps,
    )
