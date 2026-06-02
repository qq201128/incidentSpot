from __future__ import annotations

import pytest

from app.services.model_search_adaptive_parallelism import (
    AdaptiveParallelismConfig,
    HostResourceSample,
    decide_adaptive_parallelism,
)
from app.services.model_search_resource import ModelSearchResourceConfig


def test_adaptive_parallelism_scales_up_when_cpu_is_available() -> None:
    decision = decide_adaptive_parallelism(
        AdaptiveParallelismConfig(max_jobs=0, cpu_headroom_cores=1),
        _sample(logical_cpus=9, cpu_percent=30.0, available_memory_mb=32768, total_memory_mb=32768),
        running_jobs=1,
        resource=ModelSearchResourceConfig(internal_threads=2, parallel_workers=1),
    )

    assert decision.target_jobs == 4
    assert decision.capacity_jobs == 4
    assert decision.cpu_per_job == 2
    assert decision.reason == "cpu_available"


def test_adaptive_parallelism_limits_scale_up_by_memory_budget() -> None:
    decision = decide_adaptive_parallelism(
        AdaptiveParallelismConfig(memory_per_job_mb=4096, min_available_memory_mb=4096),
        _sample(logical_cpus=12, cpu_percent=30.0, available_memory_mb=12288),
        running_jobs=1,
        resource=ModelSearchResourceConfig(internal_threads=1),
    )

    assert decision.target_jobs == 3
    assert decision.capacity_jobs == 3
    assert decision.cpu_capacity_jobs == 11
    assert decision.memory_capacity_jobs == 3
    assert decision.memory_per_job_mb == 4096
    assert decision.reason == "memory_limited"


def test_adaptive_parallelism_reduces_target_under_cpu_pressure() -> None:
    decision = decide_adaptive_parallelism(
        AdaptiveParallelismConfig(max_jobs=6, cpu_high_percent=90.0),
        _sample(logical_cpus=12, cpu_percent=95.0),
        running_jobs=4,
        resource=ModelSearchResourceConfig(internal_threads=1),
    )

    assert decision.target_jobs == 3
    assert decision.reason == "cpu_pressure"


def test_adaptive_parallelism_reduces_target_under_memory_pressure() -> None:
    decision = decide_adaptive_parallelism(
        AdaptiveParallelismConfig(min_available_memory_mb=4096),
        _sample(logical_cpus=8, cpu_percent=30.0, available_memory_mb=1024),
        running_jobs=3,
        resource=ModelSearchResourceConfig(internal_threads=1),
    )

    assert decision.target_jobs == 2
    assert decision.reason == "memory_pressure"


def test_adaptive_parallelism_pauses_new_jobs_under_memory_pressure() -> None:
    decision = decide_adaptive_parallelism(
        AdaptiveParallelismConfig(min_available_memory_mb=4096),
        _sample(logical_cpus=8, cpu_percent=30.0, available_memory_mb=1024),
        running_jobs=0,
        resource=ModelSearchResourceConfig(internal_threads=1),
    )

    assert decision.target_jobs == 0
    assert decision.capacity_jobs == 0
    assert decision.reason == "memory_pressure"


def test_adaptive_parallelism_validates_threshold_order() -> None:
    with pytest.raises(ValueError, match="cpu_low_percent"):
        decide_adaptive_parallelism(
            AdaptiveParallelismConfig(cpu_low_percent=90.0, cpu_high_percent=80.0),
            _sample(logical_cpus=8, cpu_percent=30.0),
            running_jobs=1,
            resource=ModelSearchResourceConfig(),
        )


def _sample(
    *,
    logical_cpus: int,
    cpu_percent: float,
    available_memory_mb: int = 8192,
    total_memory_mb: int = 16384,
) -> HostResourceSample:
    return HostResourceSample(
        logical_cpus=logical_cpus,
        cpu_percent=cpu_percent,
        available_memory_mb=available_memory_mb,
        total_memory_mb=total_memory_mb,
        load_average_1m=None,
        sampled_at="2026-06-02T00:00:00+00:00",
    )
