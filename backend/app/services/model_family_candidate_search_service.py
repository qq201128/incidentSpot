from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from collections.abc import Iterator
from threading import Lock
from typing import Any

from app.services.experiment_profiles import lstm_training_config_for_profile, normalize_experiment_profile
from app.services.lstm_feature_builder import LstmDataset, build_lstm_training_dataset
from app.services.model_family_candidates import (
    attempted_model_search_keys,
    complete_model_candidate_progress,
    finish_model_candidate_progress,
    finish_model_candidate_progress_from_library,
    model_search_key,
    model_search_space_size,
    next_model_candidate_configs,
    read_model_candidate_library,
    read_model_candidate_progress,
    record_model_candidate,
    start_model_candidate_progress,
)
from app.services.model_family_candidate_process import train_candidate_in_process
from app.services.model_family_config import ModelFamilyTrainingConfig, normalize_model_family
from app.services.model_family_search_rules import (
    DEFAULT_PARALLEL_WORKERS,
    model_family_training_rules,
)
from app.services.model_family_status_service import model_family_status
from app.services.model_family_training_service import train_model_family

PROCESS_EXECUTOR_FAMILIES = frozenset({"xgboost"})
XGBOOST_PROCESS_WORKERS_ENV = "MODEL_FAMILY_XGBOOST_PROCESS_WORKERS"


@dataclass(frozen=True)
class ModelCandidateSearchConfig:
    family: str
    symbol: str
    duration: str
    profile: str
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS


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


def run_model_candidate_search(config: ModelCandidateSearchConfig) -> dict[str, Any]:
    cfg = _validated(config)
    base = model_training_config_for_profile(cfg.family, cfg.symbol, cfg.duration, cfg.profile)
    attempted = attempted_model_search_keys(cfg.family, cfg.symbol, cfg.duration)
    candidates = next_model_candidate_configs(base, cfg.profile, attempted)
    if not candidates:
        current = read_model_candidate_progress(cfg.family, cfg.symbol, cfg.duration)
        if _preserve_exhausted_progress(current):
            status = str(current.get("status"))
            return {"status": status, "reason": "candidate_search_exhausted", "family": cfg.family}
        status = _exhausted_candidate_status(cfg.family, cfg.symbol, cfg.duration)
        finish_model_candidate_progress_from_library(
            cfg.family,
            symbol=cfg.symbol,
            duration=cfg.duration,
            profile=cfg.profile,
            parallel_workers=cfg.parallel_workers,
            status=status,
        )
        return {"status": status, "reason": "candidate_search_exhausted", "family": cfg.family}
    start_model_candidate_progress(
        cfg.family,
        symbol=cfg.symbol,
        duration=cfg.duration,
        profile=cfg.profile,
        total=len(candidates),
        parallel_workers=cfg.parallel_workers,
    )
    trainings: list[CandidateTrainingResult] = []
    dataset_cache = CandidateDatasetCache()
    try:
        for result in _train_candidate_reports(candidates, cfg.profile, cfg.parallel_workers, dataset_cache.build):
            trainings.append(result)
            complete_model_candidate_progress(result.config, cfg.profile, result.report, len(trainings), len(candidates))
        _publish_best_trade_candidate(trainings)
        reports = [item.report for item in trainings]
        status = _batch_status(reports)
        finish_model_candidate_progress(cfg.family, symbol=cfg.symbol, duration=cfg.duration, status=status)
        return {"status": status, "family": cfg.family, "reports": reports, "trainingRules": model_family_training_rules(cfg.family)}
    except Exception:
        finish_model_candidate_progress(cfg.family, symbol=cfg.symbol, duration=cfg.duration, status="failed")
        raise


def model_training_config_for_profile(
    family: str,
    symbol: str,
    duration: str,
    profile: str,
    **overrides,
) -> ModelFamilyTrainingConfig:
    selected = normalize_model_family(family)
    base = lstm_training_config_for_profile(symbol, duration, normalize_experiment_profile(profile), **overrides)
    params = _family_default_params(selected)
    return ModelFamilyTrainingConfig(
        family=selected,
        symbol=base.symbol,
        duration=base.duration,
        feature_window=base.feature_window,
        horizon_minutes=base.horizon_minutes,
        min_samples=base.min_samples,
        epochs=base.epochs,
        batch_size=base.batch_size,
        hidden_size=base.hidden_size,
        num_layers=base.num_layers,
        learning_rate=base.learning_rate,
        train_ratio=base.train_ratio,
        val_ratio=base.val_ratio,
        min_move_bps=base.min_move_bps,
        seed=base.seed,
        params=params,
    )


def queue_total_for_family(family: str) -> int:
    return model_search_space_size(normalize_model_family(family))


def training_rules_for_family(family: str) -> dict[str, Any]:
    return model_family_training_rules(family)


def _train_candidate(config: ModelFamilyTrainingConfig, profile: str, dataset_builder) -> dict[str, Any]:
    try:
        report = train_model_family(
            config,
            dataset_builder=dataset_builder,
            publish_shadow_active=False,
            publish_trade_active=False,
            write_attempt=False,
            persist_artifacts=False,
        )
    except Exception as exc:
        report = _failed_report(config, profile, exc)
    report = {**report, "searchKey": model_search_key(config, profile)}
    record_model_candidate(config, profile, report)
    return report


def _train_candidate_reports(
    configs: list[ModelFamilyTrainingConfig],
    profile: str,
    workers: int,
    dataset_builder,
) -> Iterator[CandidateTrainingResult]:
    if workers <= 1 or len(configs) <= 1:
        for config in configs:
            yield CandidateTrainingResult(config, _train_candidate(config, profile, dataset_builder))
        return
    max_workers = min(int(workers), len(configs))
    if _uses_process_executor(configs):
        yield from _train_candidate_reports_in_processes(configs, profile, _process_worker_count(max_workers))
        return
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_train_candidate, config, profile, dataset_builder): config for config in configs}
        for future in as_completed(future_map):
            yield CandidateTrainingResult(future_map[future], future.result())


def _train_candidate_reports_in_processes(
    configs: list[ModelFamilyTrainingConfig],
    profile: str,
    workers: int,
) -> Iterator[CandidateTrainingResult]:
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(train_candidate_in_process, config, profile): config for config in configs}
        for future in as_completed(future_map):
            config = future_map[future]
            report = future.result()
            record_model_candidate(config, profile, report)
            yield CandidateTrainingResult(config, report)


def _uses_process_executor(configs: list[ModelFamilyTrainingConfig]) -> bool:
    return bool(configs) and configs[0].family in PROCESS_EXECUTOR_FAMILIES


def _process_worker_count(max_workers: int) -> int:
    raw = os.environ.get(XGBOOST_PROCESS_WORKERS_ENV)
    if raw is None:
        return max_workers
    selected = int(raw)
    if selected <= 0:
        raise ValueError(f"{XGBOOST_PROCESS_WORKERS_ENV} must be positive")
    return min(max_workers, selected)


def _validated(config: ModelCandidateSearchConfig) -> ModelCandidateSearchConfig:
    if config.parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")
    return ModelCandidateSearchConfig(
        family=normalize_model_family(config.family),
        symbol=config.symbol.strip().upper(),
        duration=config.duration,
        profile=normalize_experiment_profile(config.profile),
        parallel_workers=int(config.parallel_workers),
    )


def _batch_status(reports: list[dict[str, Any]]) -> str:
    if any(str(item.get("status")) in {"trade_active", "trained"} for item in reports):
        return "trade_active"
    if any(str(item.get("status")) == "shadow_active" for item in reports):
        return "shadow_active"
    return str(reports[-1].get("status") or "failed") if reports else "skipped"


def _exhausted_candidate_status(family: str, symbol: str, duration: str) -> str:
    records = list(read_model_candidate_library(family, symbol, duration)["records"])
    return _batch_status(records)


def _preserve_exhausted_progress(progress: dict[str, Any]) -> bool:
    status = str(progress.get("status") or "")
    completed = int(progress.get("completed") or 0)
    return completed > 0 and status not in {"failed", "idle", "queued", "running"}


def _publish_best_trade_candidate(trainings: list[CandidateTrainingResult]) -> None:
    selected = _best_trade_candidate(trainings)
    if selected is not None:
        train_model_family(selected.config)


def _best_trade_candidate(trainings: list[CandidateTrainingResult]) -> CandidateTrainingResult | None:
    eligible = [
        item
        for item in trainings
        if str(item.report.get("status") or "") in {"trade_active", "trained"}
    ]
    return max(eligible, key=lambda item: _trade_candidate_score(item.report)) if eligible else None


def _trade_candidate_score(report: dict[str, Any]) -> tuple[float, float, int]:
    validation = report.get("validation") or {}
    test = report.get("test") or {}
    return (
        min(float(validation.get("winRate") or 0.0), float(test.get("winRate") or 0.0)),
        min(float(validation.get("profitFactor") or 0.0), float(test.get("profitFactor") or 0.0)),
        int((report.get("sampleCounts") or {}).get("test") or 0),
    )


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
    }


def _family_default_params(family: str) -> dict[str, Any]:
    return {
        "random_forest": {"n_estimators": 120, "max_depth": None, "min_samples_leaf": 2},
        "xgboost": {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.08},
        "svm": {"C": 1.0, "gamma": "scale", "kernel": "rbf"},
        "bayesian": {"var_smoothing": 1e-9},
        "knn": {"n_neighbors": 7, "weights": "distance"},
        "rl_strategy": {"state_bins": 5, "alpha": 0.2, "gamma": 0.8, "epsilon": 0.1, "episodes": 20},
    }.get(family, {})


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
