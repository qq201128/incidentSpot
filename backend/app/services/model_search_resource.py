from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from app.services.model_family_candidate_executor import TORCH_JOBS_ENV, XGBOOST_PROCESS_WORKERS_ENV

DEFAULT_INTERNAL_THREADS = 4
DEFAULT_PARALLEL_WORKERS = 1
DEFAULT_XGBOOST_PROCESS_WORKERS = 1
DEFAULT_TORCH_JOBS = 1
DEFAULT_RESOURCE_PROFILE = "local_safe"
THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class ModelSearchResourceConfig:
    resource_profile: str = DEFAULT_RESOURCE_PROFILE
    internal_threads: int = DEFAULT_INTERNAL_THREADS
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS
    xgboost_process_workers: int = DEFAULT_XGBOOST_PROCESS_WORKERS
    torch_jobs: int = DEFAULT_TORCH_JOBS


def apply_model_search_resource_config(config: ModelSearchResourceConfig) -> dict[str, Any]:
    selected = validated_resource_config(config)
    for key in THREAD_ENV_VARS:
        os.environ[key] = str(selected.internal_threads)
    os.environ[XGBOOST_PROCESS_WORKERS_ENV] = str(selected.xgboost_process_workers)
    os.environ[TORCH_JOBS_ENV] = str(selected.torch_jobs)
    return resource_payload(selected)


def validated_resource_config(config: ModelSearchResourceConfig) -> ModelSearchResourceConfig:
    if config.internal_threads <= 0:
        raise ValueError("internal_threads must be positive")
    if config.parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")
    if config.xgboost_process_workers <= 0:
        raise ValueError("xgboost_process_workers must be positive")
    if config.torch_jobs <= 0:
        raise ValueError("torch_jobs must be positive")
    return ModelSearchResourceConfig(
        resource_profile=str(config.resource_profile or DEFAULT_RESOURCE_PROFILE),
        internal_threads=int(config.internal_threads),
        parallel_workers=int(config.parallel_workers),
        xgboost_process_workers=int(config.xgboost_process_workers),
        torch_jobs=int(config.torch_jobs),
    )


def resource_payload(config: ModelSearchResourceConfig) -> dict[str, Any]:
    return {
        "resourceProfile": config.resource_profile,
        "maxRunningJobs": 1,
        "internalThreads": config.internal_threads,
        "parallelWorkers": config.parallel_workers,
        "xgboostProcessWorkers": config.xgboost_process_workers,
        "torchJobs": config.torch_jobs,
        "threadEnv": {key: str(config.internal_threads) for key in THREAD_ENV_VARS},
        XGBOOST_PROCESS_WORKERS_ENV: str(config.xgboost_process_workers),
        TORCH_JOBS_ENV: str(config.torch_jobs),
    }
