from __future__ import annotations

import sys

import pytest

from scripts import run_model_search_worker as worker_script


def test_worker_script_runs_once_by_default(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    reports = [_job_report("succeeded"), _idle_report()]
    calls = _patch_worker_script(monkeypatch, reports)

    assert worker_script.main() == 0

    captured = capsys.readouterr()
    assert len(calls) == 1
    assert '"status": "succeeded"' in captured.out


def test_worker_script_print_report_streams_without_json_dumps(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(worker_script.json, "dumps", lambda *_args, **_kwargs: (_ for _ in ()).throw(MemoryError()))

    worker_script._print_report(_job_report("succeeded"), compact=False)

    captured = capsys.readouterr()
    assert '"status": "succeeded"' in captured.out


def test_worker_script_run_until_empty_exits_on_idle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    reports = [_job_report("succeeded"), _job_report("rejected"), _idle_report()]
    calls = _patch_worker_script(monkeypatch, reports, "--run-until-empty", "--no-adaptive-parallelism")

    assert worker_script.main() == 0

    captured = capsys.readouterr()
    assert len(calls) == 3
    assert '"status": "rejected"' in captured.out
    assert '"reason": "no_pending_job"' in captured.out


def test_worker_script_run_until_empty_continues_after_partial(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    reports = [_job_report("partial"), _job_report("succeeded"), _idle_report()]
    calls = _patch_worker_script(monkeypatch, reports, "--run-until-empty", "--no-adaptive-parallelism")

    assert worker_script.main() == 0

    captured = capsys.readouterr()
    assert len(calls) == 3
    assert '"status": "partial"' in captured.out
    assert '"status": "succeeded"' in captured.out
    assert '"reason": "no_pending_job"' in captured.out


def test_worker_script_loop_polls_after_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    reports = [_idle_report()]
    calls = _patch_worker_script(
        monkeypatch,
        reports,
        "--loop",
        "--poll-seconds",
        "0.01",
        "--no-adaptive-parallelism",
    )
    sleeps = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(worker_script.time, "sleep", fake_sleep)

    with pytest.raises(KeyboardInterrupt):
        worker_script.main()

    assert len(calls) == 1
    assert sleeps == [0.01]


def test_worker_script_surfaces_runner_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_model_search_worker.py", "--run-until-empty", "--no-adaptive-parallelism"],
    )

    def fake_run_one(_config):
        raise RuntimeError("training crashed")

    monkeypatch.setattr(worker_script, "run_one_model_search_job", fake_run_one)

    with pytest.raises(RuntimeError, match="training crashed"):
        worker_script.main()


def test_worker_script_run_until_empty_uses_adaptive_pool_by_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_model_search_worker.py", "--run-until-empty", "--poll-seconds", "0.01"],
    )

    def fake_pool(config, adaptive, *, poll_seconds: float, run_until_empty: bool):
        calls.append((config, adaptive, poll_seconds, run_until_empty))
        return iter([_job_report("succeeded"), _idle_report()])

    monkeypatch.setattr(worker_script, "run_adaptive_worker_pool", fake_pool)

    assert worker_script.main() == 0

    captured = capsys.readouterr()
    assert len(calls) == 1
    assert calls[0][0].max_running_jobs == 1
    assert calls[0][0].candidates_per_job == worker_script.DEFAULT_CANDIDATES_PER_JOB
    assert calls[0][0].candidate_budget == worker_script.DEFAULT_CANDIDATE_BUDGET
    assert calls[0][1].max_jobs == 0
    assert calls[0][2] == 0.01
    assert calls[0][3] is True
    assert '"status": "idle"' in captured.out


def test_worker_script_passes_memory_budget_to_adaptive_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_search_worker.py",
            "--run-until-empty",
            "--memory-per-job-mb",
            "8192",
            "--min-available-memory-mb",
            "6144",
        ],
    )

    def fake_pool(config, adaptive, *, poll_seconds: float, run_until_empty: bool):
        calls.append((config, adaptive, poll_seconds, run_until_empty))
        return iter([_idle_report()])

    monkeypatch.setattr(worker_script, "run_adaptive_worker_pool", fake_pool)

    assert worker_script.main() == 0
    assert calls[0][1].memory_per_job_mb == 8192
    assert calls[0][1].min_available_memory_mb == 6144


def test_worker_script_candidate_budget_zero_disables_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_model_search_worker.py", "--run-until-empty", "--candidate-budget", "0"],
    )

    def fake_pool(config, adaptive, *, poll_seconds: float, run_until_empty: bool):
        calls.append((config, adaptive, poll_seconds, run_until_empty))
        return iter([_idle_report()])

    monkeypatch.setattr(worker_script, "run_adaptive_worker_pool", fake_pool)

    assert worker_script.main() == 0
    assert calls[0][0].candidate_budget is None


def test_worker_script_passes_claim_filters_to_worker_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_search_worker.py",
            "--run-until-empty",
            "--symbols",
            "btcusdt",
            "--durations",
            "10m",
            "--families",
            "knn",
        ],
    )

    def fake_pool(config, adaptive, *, poll_seconds: float, run_until_empty: bool):
        calls.append((config, adaptive, poll_seconds, run_until_empty))
        return iter([_idle_report()])

    monkeypatch.setattr(worker_script, "run_adaptive_worker_pool", fake_pool)

    assert worker_script.main() == 0
    assert calls[0][0].filters == {
        "symbols": ("BTCUSDT",),
        "durations": ("10m",),
        "families": ("knn",),
    }


def _patch_worker_script(monkeypatch: pytest.MonkeyPatch, reports: list[dict], *args: str) -> list:
    calls = []
    monkeypatch.setattr(sys, "argv", ["run_model_search_worker.py", *args])

    def fake_run_one(config):
        calls.append(config)
        if not reports:
            raise AssertionError("unexpected extra worker call")
        return reports.pop(0)

    monkeypatch.setattr(worker_script, "run_one_model_search_job", fake_run_one)
    return calls


def _job_report(status: str) -> dict:
    return {"status": status, "job": {"job_id": f"job-{status}"}}


def _idle_report() -> dict:
    return {"status": "idle", "reason": "no_pending_job", "queue": {"total": 0}}
