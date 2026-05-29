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


def test_worker_script_run_until_empty_exits_on_idle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    reports = [_job_report("succeeded"), _job_report("rejected"), _idle_report()]
    calls = _patch_worker_script(monkeypatch, reports, "--run-until-empty")

    assert worker_script.main() == 0

    captured = capsys.readouterr()
    assert len(calls) == 3
    assert '"status": "rejected"' in captured.out
    assert '"reason": "no_pending_job"' in captured.out


def test_worker_script_loop_polls_after_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    reports = [_idle_report()]
    calls = _patch_worker_script(monkeypatch, reports, "--loop", "--poll-seconds", "0.01")
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
    monkeypatch.setattr(sys, "argv", ["run_model_search_worker.py", "--run-until-empty"])

    def fake_run_one(_config):
        raise RuntimeError("training crashed")

    monkeypatch.setattr(worker_script, "run_one_model_search_job", fake_run_one)

    with pytest.raises(RuntimeError, match="training crashed"):
        worker_script.main()


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
