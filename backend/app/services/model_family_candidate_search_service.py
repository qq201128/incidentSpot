from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.exception_payloads import exception_details
from app.services.experiment_profiles import lstm_training_config_for_profile, normalize_experiment_profile
from app.services.model_family_candidate_executor import (
    CandidateDatasetCache,
    CandidateTrainingResult,
    train_candidate_reports,
)
from app.services.model_family_candidate_batches import candidate_job_batch, job_batch_payload
from app.services.model_family_candidate_halving import (
    close_halving_stage,
    coarse_candidate_config,
    walk_forward_stage_payload,
)
from app.services.model_family_candidates import (
    attempted_model_search_keys,
    complete_model_candidate_progress,
    finish_model_candidate_progress,
    finish_model_candidate_progress_from_library,
    model_search_space_size,
    next_model_candidate_configs,
    read_model_candidate_library,
    read_model_candidate_progress,
    record_model_candidate,
    start_model_candidate_progress,
)
from app.services.model_family_candidate_publisher import publish_best_model_candidate
from app.services.model_family_config import ModelFamilyTrainingConfig, normalize_model_family
from app.services.model_family_search_rules import (
    DEFAULT_PARALLEL_WORKERS,
    model_family_training_rules,
)
from app.services.model_family_walk_forward import run_walk_forward_stage


@dataclass(frozen=True)
class ModelCandidateSearchConfig:
    family: str
    symbol: str
    duration: str
    profile: str
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS
    reset_history: bool = False
    candidates_per_job: int | None = None


def run_model_candidate_search(config: ModelCandidateSearchConfig) -> dict[str, Any]:
    cfg = _validated(config)
    base = model_training_config_for_profile(cfg.family, cfg.symbol, cfg.duration, profile=cfg.profile)
    attempted = frozenset() if cfg.reset_history else attempted_model_search_keys(cfg.family, cfg.symbol, cfg.duration)
    available = next_model_candidate_configs(base, cfg.profile, attempted)
    if not available:
        return _exhausted_search_result(cfg)
    candidates = candidate_job_batch(available, cfg.candidates_per_job)
    return _run_candidate_batch(cfg, candidates, available_count=len(available))


def _exhausted_search_result(cfg: ModelCandidateSearchConfig) -> dict[str, Any]:
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


def _run_candidate_batch(
    cfg: ModelCandidateSearchConfig,
    candidates: list[ModelFamilyTrainingConfig],
    *,
    available_count: int,
) -> dict[str, Any]:
    start_model_candidate_progress(
        cfg.family,
        symbol=cfg.symbol,
        duration=cfg.duration,
        profile=cfg.profile,
        total=len(candidates),
        parallel_workers=cfg.parallel_workers,
    )
    dataset_cache = CandidateDatasetCache()
    try:
        evaluation = _run_successive_halving(candidates, cfg, dataset_cache.build)
        published = publish_best_model_candidate(
            evaluation["finalists"],
            observation_trainings=evaluation["observationCandidates"],
        )
        reports = [item.report for item in evaluation["reports"]]
        status = _batch_status(reports, published)
        finish_model_candidate_progress(cfg.family, symbol=cfg.symbol, duration=cfg.duration, status=status)
        return {
            "status": status,
            "family": cfg.family,
            "reports": reports,
            "trainingRules": model_family_training_rules(cfg.family),
            "successiveHalvingStages": evaluation["stages"],
            "jobBatch": job_batch_payload(candidates, available_count),
        }
    except Exception as exc:
        finish_model_candidate_progress(
            cfg.family,
            symbol=cfg.symbol,
            duration=cfg.duration,
            status="failed",
            failure=exception_details(exc, "candidate_search"),
        )
        raise


def model_training_config_for_profile(
    family: str,
    symbol: str,
    duration: str,
    *,
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


def _run_successive_halving(candidates, cfg: ModelCandidateSearchConfig, dataset_builder) -> dict[str, Any]:
    reports: list[CandidateTrainingResult] = []
    stages = []
    completed = 0
    coarse_configs = [coarse_candidate_config(item) for item in candidates]
    coarse = _collect_stage(
        coarse_configs,
        candidates,
        cfg=cfg,
        dataset_builder=dataset_builder,
        stage="coarse",
        completed=completed,
        total=len(candidates),
    )
    completed += len(coarse)
    coarse_closed = close_halving_stage(coarse, "coarse")
    _record_stage_reports(coarse_closed.reports, cfg.profile)
    reports.extend(coarse_closed.reports)
    stages.append(coarse_closed.payload)
    total = completed + len(coarse_closed.survivors)
    full = _collect_stage(
        _configs(coarse_closed.survivors),
        None,
        cfg=cfg,
        dataset_builder=dataset_builder,
        stage="full",
        completed=completed,
        total=total,
    )
    full_closed = close_halving_stage(full, "full")
    _record_stage_reports(full_closed.reports, cfg.profile)
    reports.extend(full_closed.reports)
    stages.append(full_closed.payload)
    if full_closed.survivors:
        walk_forward_survivors, walk_forward_payload = run_walk_forward_stage(full_closed.survivors, dataset_builder)
    else:
        walk_forward_survivors, walk_forward_payload = [], walk_forward_stage_payload([])
    _record_stage_reports(walk_forward_survivors, cfg.profile)
    reports.extend(walk_forward_survivors)
    stages.append(walk_forward_payload)
    return {
        "reports": reports,
        "finalists": walk_forward_survivors,
        "observationCandidates": full_closed.reports,
        "stages": stages,
    }


def _collect_stage(
    train_configs,
    record_configs,
    *,
    cfg,
    dataset_builder,
    stage: str,
    completed: int,
    total: int,
):
    results = []
    for result in train_candidate_reports(
        train_configs,
        cfg.profile,
        cfg.parallel_workers,
        dataset_builder,
        stage=stage,
        record_configs=record_configs,
    ):
        results.append(result)
        completed += 1
        complete_model_candidate_progress(
            result.config,
            profile=cfg.profile,
            report=result.report,
            completed=completed,
            total=max(total, completed),
        )
    return results


def _record_stage_reports(results: list[CandidateTrainingResult], profile: str) -> None:
    for item in results:
        record_model_candidate(item.config, profile, item.report)


def _configs(results: list[CandidateTrainingResult]) -> list[ModelFamilyTrainingConfig]:
    return [item.config for item in results]


def _validated(config: ModelCandidateSearchConfig) -> ModelCandidateSearchConfig:
    if config.parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")
    if config.candidates_per_job is not None and config.candidates_per_job <= 0:
        raise ValueError("candidates_per_job must be positive")
    return ModelCandidateSearchConfig(
        family=normalize_model_family(config.family),
        symbol=config.symbol.strip().upper(),
        duration=config.duration,
        profile=normalize_experiment_profile(config.profile),
        parallel_workers=int(config.parallel_workers),
        reset_history=bool(config.reset_history),
        candidates_per_job=None if config.candidates_per_job is None else int(config.candidates_per_job),
    )


def _batch_status(reports: list[dict[str, Any]], published: dict[str, Any] | None = None) -> str:
    if published is not None:
        return str(published.get("status") or "failed")
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


def _family_default_params(family: str) -> dict[str, Any]:
    return {
        "random_forest": {"n_estimators": 120, "max_depth": None, "min_samples_leaf": 2},
        "extra_trees": {"n_estimators": 120, "max_depth": None, "min_samples_leaf": 2},
        "xgboost": {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.08},
        "lightgbm": {"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.08},
        "catboost": {"iterations": 100, "depth": 4, "learning_rate": 0.08, "l2_leaf_reg": 3.0},
        "logistic_elasticnet": {"C": 1.0, "l1_ratio": 0.5},
        "svm": {"C": 1.0, "gamma": "scale", "kernel": "rbf"},
        "bayesian": {"var_smoothing": 1e-9},
        "knn": {"n_neighbors": 7, "weights": "distance"},
        "rl_strategy": {"state_bins": 5, "alpha": 0.2, "gamma": 0.8, "epsilon": 0.1, "episodes": 20},
    }.get(family, {})
