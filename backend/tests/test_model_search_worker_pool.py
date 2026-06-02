from __future__ import annotations

from collections import deque

import pytest

from app.services import model_search_worker_pool as pool
from app.services.model_search_adaptive_parallelism import AdaptiveParallelismConfig, HostResourceSample
from app.services.model_search_job_runner import ModelSearchWorkerConfig
from app.services.model_search_resource import ModelSearchResourceConfig


def test_worker_pool_starts_parallel_jobs_and_releases_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = FakeLauncher(
        reports=[
            _job_report("job-1"),
            _job_report("job-2"),
            _idle_report(),
        ]
    )
    monkeypatch.setattr(pool, "sample_host_resources", lambda _seconds: _sample(8, 20.0))
    monkeypatch.setattr(pool.time, "sleep", lambda _seconds: None)

    reports = list(
        pool.run_adaptive_worker_pool(
            ModelSearchWorkerConfig(resource=ModelSearchResourceConfig(internal_threads=2)),
            AdaptiveParallelismConfig(max_jobs=2, cpu_sample_seconds=0.01),
            poll_seconds=0.01,
            run_until_empty=True,
            launcher_factory=lambda max_workers: launcher.record_max(max_workers),
        )
    )

    assert launcher.max_workers == 2
    assert len(launcher.started_configs) == 3
    assert [report["job"]["job_id"] for report in reports[:2]] == ["job-1", "job-2"]
    assert reports[-1]["status"] == "idle"
    assert reports[0]["adaptiveParallelism"]["targetJobs"] == 2


def test_worker_pool_reduces_new_starts_after_cpu_pressure(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = FakeLauncher(
        reports=[
            _job_report("job-1"),
            _job_report("job-2"),
            _idle_report(),
        ],
        initially_done=0,
    )
    samples = deque([_sample(8, 20.0), _sample(8, 20.0), _sample(8, 20.0), _sample(8, 95.0), _sample(8, 20.0)])
    monkeypatch.setattr(pool, "sample_host_resources", lambda _seconds: samples.popleft())

    def sleep_after_two_starts(seconds: float) -> None:
        if len(launcher.handles) >= 2:
            launcher.finish_one(seconds)

    monkeypatch.setattr(pool.time, "sleep", sleep_after_two_starts)

    reports = list(
        pool.run_adaptive_worker_pool(
            ModelSearchWorkerConfig(resource=ModelSearchResourceConfig(internal_threads=1)),
            AdaptiveParallelismConfig(max_jobs=3, cpu_sample_seconds=0.01),
            poll_seconds=0.01,
            run_until_empty=True,
            launcher_factory=lambda max_workers: launcher.record_max(max_workers),
        )
    )

    assert len(launcher.started_configs) == 3
    assert launcher.started_configs[-1].max_running_jobs == 3
    assert reports[-1]["status"] == "idle"


def test_worker_pool_continues_when_partial_and_idle_complete_together(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = FakeLauncher(
        reports=[
            _partial_report("job-partial"),
            _job_report("job-requeued"),
            _idle_report(),
        ]
    )
    monkeypatch.setattr(pool, "sample_host_resources", lambda _seconds: _sample(8, 20.0))
    monkeypatch.setattr(pool.time, "sleep", lambda _seconds: None)

    reports = list(
        pool.run_adaptive_worker_pool(
            ModelSearchWorkerConfig(resource=ModelSearchResourceConfig(internal_threads=1)),
            AdaptiveParallelismConfig(max_jobs=2, cpu_sample_seconds=0.01),
            poll_seconds=0.01,
            run_until_empty=True,
            launcher_factory=lambda max_workers: launcher.record_max(max_workers),
        )
    )

    assert [report["status"] for report in reports] == ["partial", "succeeded", "idle"]
    assert len(launcher.started_configs) == 3


def test_worker_pool_waits_when_memory_blocks_new_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = FakeLauncher(reports=[_idle_report()])
    samples = deque([_sample(8, 20.0), _sample(8, 20.0, available_memory_mb=1024), _sample(8, 20.0)])
    monkeypatch.setattr(pool, "sample_host_resources", lambda _seconds: samples.popleft())
    monkeypatch.setattr(pool.time, "sleep", lambda _seconds: None)

    reports = list(
        pool.run_adaptive_worker_pool(
            ModelSearchWorkerConfig(resource=ModelSearchResourceConfig(internal_threads=1)),
            AdaptiveParallelismConfig(max_jobs=2, cpu_sample_seconds=0.01),
            poll_seconds=0.01,
            run_until_empty=True,
            launcher_factory=lambda max_workers: launcher.record_max(max_workers),
            pending_job_counter=lambda: 1,
        )
    )

    assert reports[0]["status"] == "waiting"
    assert reports[0]["reason"] == "memory_pressure"
    assert reports[0]["adaptiveParallelism"]["targetJobs"] == 0
    assert len(launcher.started_configs) == 1


def test_process_worker_returns_ipc_safe_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    large_payload = "x" * 10000
    monkeypatch.setattr(
        pool,
        "run_one_model_search_job",
        lambda _config: {
            "status": "succeeded",
            "job": {
                "job_id": "job-big",
                "symbol": "BTCUSDT",
                "duration": "10m",
                "model_family": "lstm",
                "status": "succeeded",
                "trainingReport": {"raw": large_payload},
                "log_path": "runtime/job-big.log",
            },
            "result": {
                "status": "trained",
                "reports": [{"raw": large_payload}],
                "trainingRules": {"raw": large_payload},
                "successiveHalvingStages": [{"raw": large_payload}],
                "jobBatch": {
                    "selectedCandidates": 1,
                    "availableCandidatesBeforeJob": 4,
                    "remainingCandidatesAfterJob": 3,
                    "hasMoreCandidates": True,
                },
            },
        },
    )

    report = pool._run_one_model_search_job(ModelSearchWorkerConfig())

    assert report["summaryOnly"] is True
    assert report["job"] == {
        "job_id": "job-big",
        "symbol": "BTCUSDT",
        "duration": "10m",
        "model_family": "lstm",
        "status": "succeeded",
        "log_path": "runtime/job-big.log",
    }
    assert report["result"]["status"] == "trained"
    assert report["result"]["jobBatch"]["hasMoreCandidates"] is True
    assert report["result"]["omittedKeys"] == ["reports", "trainingRules", "successiveHalvingStages"]
    assert "reports" not in report["result"]


class FakeLauncher:
    def __init__(self, *, reports: list[dict], initially_done: int | None = None) -> None:
        self.reports = deque(reports)
        self.initially_done = len(reports) if initially_done is None else initially_done
        self.handles: list[FakeHandle] = []
        self.started_configs: list[ModelSearchWorkerConfig] = []
        self.max_workers = 0

    def __enter__(self) -> "FakeLauncher":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        return None

    def record_max(self, max_workers: int) -> "FakeLauncher":
        self.max_workers = max_workers
        return self

    def start(self, config: ModelSearchWorkerConfig) -> "FakeHandle":
        if not self.reports:
            raise AssertionError("unexpected extra worker start")
        handle = FakeHandle(self.reports.popleft(), done=len(self.handles) < self.initially_done)
        self.handles.append(handle)
        self.started_configs.append(config)
        return handle

    def finish_one(self, _seconds: float) -> None:
        for handle in self.handles:
            if not handle.done():
                handle.mark_done()
                return


class FakeHandle:
    def __init__(self, report: dict, *, done: bool) -> None:
        self.report = report
        self._done = done

    def done(self) -> bool:
        return self._done

    def result(self) -> dict:
        return self.report

    def mark_done(self) -> None:
        self._done = True


def _sample(logical_cpus: int, cpu_percent: float, *, available_memory_mb: int = 32768) -> HostResourceSample:
    return HostResourceSample(
        logical_cpus=logical_cpus,
        cpu_percent=cpu_percent,
        available_memory_mb=available_memory_mb,
        total_memory_mb=32768,
        load_average_1m=None,
        sampled_at="2026-06-02T00:00:00+00:00",
    )


def _job_report(job_id: str) -> dict:
    return {"status": "succeeded", "job": {"job_id": job_id}, "result": {"status": "trained"}}


def _partial_report(job_id: str) -> dict:
    return {"status": "partial", "job": {"job_id": job_id}, "result": {"status": "partial_batch"}}


def _idle_report() -> dict:
    return {"status": "idle", "reason": "no_pending_job", "queue": {"total": 0}}
