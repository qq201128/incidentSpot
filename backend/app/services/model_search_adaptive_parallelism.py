from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import psutil

AUTO_MAX_RUNNING_JOBS = 0
AUTO_CPU_PER_JOB = 0
AUTO_MEMORY_PER_JOB_MB = 0
BYTES_PER_MIB = 1024 * 1024
AUTO_MEMORY_JOB_DIVISOR = 4
DEFAULT_CPU_HEADROOM_CORES = 1
DEFAULT_CPU_HIGH_PERCENT = 90.0
DEFAULT_CPU_LOW_PERCENT = 65.0
DEFAULT_CPU_SAMPLE_SECONDS = 0.1
DEFAULT_MIN_AVAILABLE_MEMORY_MB = 4096
DEFAULT_MIN_RUNNING_JOBS = 1
MIN_AUTO_MEMORY_PER_JOB_MB = 6144


@dataclass(frozen=True)
class HostResourceSample:
    logical_cpus: int
    cpu_percent: float
    available_memory_mb: int
    total_memory_mb: int
    load_average_1m: float | None
    sampled_at: str


@dataclass(frozen=True)
class AdaptiveParallelismConfig:
    min_jobs: int = DEFAULT_MIN_RUNNING_JOBS
    max_jobs: int = AUTO_MAX_RUNNING_JOBS
    cpu_per_job: int = AUTO_CPU_PER_JOB
    memory_per_job_mb: int = AUTO_MEMORY_PER_JOB_MB
    cpu_headroom_cores: int = DEFAULT_CPU_HEADROOM_CORES
    cpu_low_percent: float = DEFAULT_CPU_LOW_PERCENT
    cpu_high_percent: float = DEFAULT_CPU_HIGH_PERCENT
    min_available_memory_mb: int = DEFAULT_MIN_AVAILABLE_MEMORY_MB
    cpu_sample_seconds: float = DEFAULT_CPU_SAMPLE_SECONDS


@dataclass(frozen=True)
class AdaptiveCapacityPlan:
    capacity_jobs: int
    cpu_capacity_jobs: int
    memory_capacity_jobs: int
    cpu_per_job: int
    memory_per_job_mb: int


@dataclass(frozen=True)
class AdaptiveParallelismDecision:
    target_jobs: int
    capacity_jobs: int
    cpu_capacity_jobs: int
    memory_capacity_jobs: int
    cpu_per_job: int
    memory_per_job_mb: int
    reason: str
    sample: HostResourceSample

    def to_payload(self) -> dict[str, Any]:
        return {
            "targetJobs": self.target_jobs,
            "capacityJobs": self.capacity_jobs,
            "cpuCapacityJobs": self.cpu_capacity_jobs,
            "memoryCapacityJobs": self.memory_capacity_jobs,
            "cpuPerJob": self.cpu_per_job,
            "memoryPerJobMb": self.memory_per_job_mb,
            "reason": self.reason,
            "sample": {
                "logicalCpus": self.sample.logical_cpus,
                "cpuPercent": self.sample.cpu_percent,
                "availableMemoryMb": self.sample.available_memory_mb,
                "totalMemoryMb": self.sample.total_memory_mb,
                "loadAverage1m": self.sample.load_average_1m,
                "sampledAt": self.sample.sampled_at,
            },
        }


def sample_host_resources(cpu_sample_seconds: float = DEFAULT_CPU_SAMPLE_SECONDS) -> HostResourceSample:
    if cpu_sample_seconds <= 0:
        raise ValueError("cpu_sample_seconds must be positive")
    logical_cpus = psutil.cpu_count(logical=True)
    if logical_cpus is None or logical_cpus <= 0:
        raise RuntimeError("psutil could not detect logical CPU count")
    memory = psutil.virtual_memory()
    return HostResourceSample(
        logical_cpus=int(logical_cpus),
        cpu_percent=float(psutil.cpu_percent(interval=cpu_sample_seconds)),
        available_memory_mb=int(memory.available // BYTES_PER_MIB),
        total_memory_mb=int(memory.total // BYTES_PER_MIB),
        load_average_1m=_load_average_1m(),
        sampled_at=datetime.now(timezone.utc).isoformat(),
    )


def decide_adaptive_parallelism(
    config: AdaptiveParallelismConfig,
    sample: HostResourceSample,
    *,
    running_jobs: int,
    resource: Any,
) -> AdaptiveParallelismDecision:
    selected = _validated_config(config)
    cpu_per_job = _cpu_per_job(selected, resource)
    capacity = _capacity_plan(selected, sample, running_jobs=running_jobs, cpu_per_job=cpu_per_job)
    target, reason = _target_jobs(selected, sample, running_jobs=running_jobs, capacity=capacity)
    return AdaptiveParallelismDecision(
        target_jobs=target,
        capacity_jobs=capacity.capacity_jobs,
        cpu_capacity_jobs=capacity.cpu_capacity_jobs,
        memory_capacity_jobs=capacity.memory_capacity_jobs,
        cpu_per_job=cpu_per_job,
        memory_per_job_mb=capacity.memory_per_job_mb,
        reason=reason,
        sample=sample,
    )


def max_adaptive_jobs(config: AdaptiveParallelismConfig, sample: HostResourceSample, resource: Any) -> int:
    selected = _validated_config(config)
    cpu_capacity = _cpu_capacity_jobs(selected, sample.logical_cpus, _cpu_per_job(selected, resource))
    cap = cpu_capacity if selected.max_jobs == AUTO_MAX_RUNNING_JOBS else selected.max_jobs
    return max(1, min(cap, cpu_capacity))


def _target_jobs(
    config: AdaptiveParallelismConfig,
    sample: HostResourceSample,
    *,
    running_jobs: int,
    capacity: AdaptiveCapacityPlan,
) -> tuple[int, str]:
    running = max(int(running_jobs), 0)
    if sample.available_memory_mb < config.min_available_memory_mb:
        return _reduced_target(running, capacity.capacity_jobs), "memory_pressure"
    if sample.cpu_percent >= config.cpu_high_percent:
        return _reduced_target(running, capacity.capacity_jobs), "cpu_pressure"
    if sample.cpu_percent <= config.cpu_low_percent:
        return capacity.capacity_jobs, _capacity_reason(capacity, running)
    if capacity.capacity_jobs <= 0:
        return 0, _capacity_reason(capacity, running)
    target = min(max(running, config.min_jobs), capacity.capacity_jobs)
    return target, _steady_reason(capacity, running)


def _reduced_target(running_jobs: int, capacity: int) -> int:
    if running_jobs <= 0:
        return 0
    return max(0, min(running_jobs - 1, capacity))


def _capacity_plan(
    config: AdaptiveParallelismConfig,
    sample: HostResourceSample,
    *,
    running_jobs: int,
    cpu_per_job: int,
) -> AdaptiveCapacityPlan:
    cpu_capacity = _cpu_capacity_jobs(config, sample.logical_cpus, cpu_per_job)
    memory_per_job = _memory_per_job_mb(config, sample)
    memory_capacity = _memory_capacity_jobs(
        config,
        sample,
        running_jobs=running_jobs,
        memory_per_job_mb=memory_per_job,
    )
    configured_capacity = _configured_capacity(config, cpu_capacity, memory_capacity)
    return AdaptiveCapacityPlan(
        capacity_jobs=configured_capacity,
        cpu_capacity_jobs=cpu_capacity,
        memory_capacity_jobs=memory_capacity,
        cpu_per_job=cpu_per_job,
        memory_per_job_mb=memory_per_job,
    )


def _cpu_capacity_jobs(config: AdaptiveParallelismConfig, logical_cpus: int, cpu_per_job: int) -> int:
    core_budget = max(int(logical_cpus) - config.cpu_headroom_cores, 1)
    cpu_limited = max(config.min_jobs, core_budget // cpu_per_job)
    return max(config.min_jobs, cpu_limited)


def _memory_capacity_jobs(
    config: AdaptiveParallelismConfig,
    sample: HostResourceSample,
    *,
    running_jobs: int,
    memory_per_job_mb: int,
) -> int:
    usable_memory = max(sample.available_memory_mb - config.min_available_memory_mb, 0)
    new_slots = usable_memory // memory_per_job_mb
    return max(int(running_jobs), 0) + int(new_slots)


def _configured_capacity(config: AdaptiveParallelismConfig, cpu_capacity: int, memory_capacity: int) -> int:
    resource_capacity = min(cpu_capacity, memory_capacity)
    cap = resource_capacity if config.max_jobs == AUTO_MAX_RUNNING_JOBS else config.max_jobs
    return max(0, min(cap, resource_capacity))


def _memory_per_job_mb(config: AdaptiveParallelismConfig, sample: HostResourceSample) -> int:
    if config.memory_per_job_mb != AUTO_MEMORY_PER_JOB_MB:
        return config.memory_per_job_mb
    auto_budget = sample.total_memory_mb // AUTO_MEMORY_JOB_DIVISOR
    return max(MIN_AUTO_MEMORY_PER_JOB_MB, auto_budget)


def _cpu_per_job(config: AdaptiveParallelismConfig, resource: Any) -> int:
    if config.cpu_per_job != AUTO_CPU_PER_JOB:
        return config.cpu_per_job
    internal_threads = _positive_resource_int(resource, "internal_threads")
    parallel_workers = _positive_resource_int(resource, "parallel_workers")
    xgboost_workers = _positive_resource_int(resource, "xgboost_process_workers")
    torch_jobs = _positive_resource_int(resource, "torch_jobs")
    return max(1, internal_threads * max(parallel_workers, xgboost_workers, torch_jobs))


def _positive_resource_int(resource: Any, field: str) -> int:
    value = int(getattr(resource, field))
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _capacity_reason(capacity: AdaptiveCapacityPlan, running_jobs: int) -> str:
    if capacity.capacity_jobs < capacity.cpu_capacity_jobs and capacity.capacity_jobs < capacity.memory_capacity_jobs:
        return "max_jobs_limited"
    if capacity.capacity_jobs == capacity.memory_capacity_jobs and capacity.memory_capacity_jobs < capacity.cpu_capacity_jobs:
        return "memory_limited"
    return "cpu_available"


def _steady_reason(capacity: AdaptiveCapacityPlan, running_jobs: int) -> str:
    if capacity.capacity_jobs <= running_jobs:
        return _capacity_reason(capacity, running_jobs)
    return "steady"


def _validated_config(config: AdaptiveParallelismConfig) -> AdaptiveParallelismConfig:
    if config.min_jobs <= 0:
        raise ValueError("min_jobs must be positive")
    if config.max_jobs < AUTO_MAX_RUNNING_JOBS:
        raise ValueError("max_jobs must be zero for auto or positive")
    if config.max_jobs and config.max_jobs < config.min_jobs:
        raise ValueError("max_jobs must be greater than or equal to min_jobs")
    if config.cpu_per_job < AUTO_CPU_PER_JOB:
        raise ValueError("cpu_per_job must be zero for auto or positive")
    if config.memory_per_job_mb < AUTO_MEMORY_PER_JOB_MB:
        raise ValueError("memory_per_job_mb must be zero for auto or positive")
    if config.cpu_headroom_cores < 0:
        raise ValueError("cpu_headroom_cores must be non-negative")
    if config.cpu_low_percent < 0 or config.cpu_high_percent > 100:
        raise ValueError("CPU thresholds must be within 0..100")
    if config.cpu_low_percent >= config.cpu_high_percent:
        raise ValueError("cpu_low_percent must be lower than cpu_high_percent")
    if config.min_available_memory_mb < 0:
        raise ValueError("min_available_memory_mb must be non-negative")
    if config.cpu_sample_seconds <= 0:
        raise ValueError("cpu_sample_seconds must be positive")
    return config


def _load_average_1m() -> float | None:
    if not hasattr(os, "getloadavg"):
        return None
    return float(os.getloadavg()[0])
