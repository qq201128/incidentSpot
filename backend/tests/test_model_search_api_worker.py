from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest

from app.services import model_search_api_worker as api_worker


def test_api_worker_starts_real_worker_command_once(monkeypatch: pytest.MonkeyPatch) -> None:
    launched = []
    runtime_path = _runtime_path("start")
    monkeypatch.setattr(api_worker, "API_WORKER_LOG_DIR", runtime_path / "api-worker")
    monkeypatch.setattr(api_worker, "JOB_LOG_DIR", runtime_path / "jobs")
    monkeypatch.setattr(api_worker.threading, "Thread", FakeThread)

    def launcher(command: list[str], **kwargs: object) -> FakeProcess:
        launched.append({"command": command, "kwargs": kwargs})
        return FakeProcess()

    first = api_worker.ensure_api_model_search_worker(_resource(), launcher=launcher)
    second = api_worker.ensure_api_model_search_worker(_resource(), launcher=launcher)

    assert first["started"] is True
    assert first["running"] is True
    assert first["managedByApi"] is True
    assert second["started"] is False
    assert len(launched) == 1
    command = launched[0]["command"]
    assert command[1].endswith("backend\\scripts\\run_model_search_worker.py") or command[1].endswith(
        "backend/scripts/run_model_search_worker.py"
    )
    assert "--run-until-empty" in command
    assert command[command.index("--max-running-jobs") + 1] == "0"
    assert command[command.index("--internal-threads") + 1] == "2"
    assert command[command.index("--parallel-workers") + 1] == "3"
    assert command[command.index("--xgboost-process-workers") + 1] == "1"
    assert command[command.index("--torch-jobs") + 1] == "1"
    assert "--compact" in command
    assert launched[0]["kwargs"]["cwd"] == str(api_worker.PROJECT_ROOT)


def test_api_worker_startup_failure_is_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_worker, "API_WORKER_LOG_DIR", _runtime_path("failure") / "api-worker")

    def launcher(_command: list[str], **_kwargs: object) -> FakeProcess:
        raise OSError("python missing")

    with pytest.raises(RuntimeError, match="startup failed"):
        api_worker.ensure_api_model_search_worker(_resource(), launcher=launcher)

    status = api_worker.api_model_search_worker_status()
    assert status["running"] is False
    assert status["lastFailureReason"] == "python missing"
    assert status["logPath"].endswith("_api_worker.log")


class FakeThread:
    def __init__(self, *, target: object, args: tuple[object, ...], name: str, daemon: bool) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon

    def start(self) -> None:
        return None


class FakeProcess:
    def poll(self) -> None:
        return None

    def wait(self) -> int:
        return 0


def _resource() -> dict:
    return {
        "resourceProfile": "local_safe",
        "internalThreads": 2,
        "parallelWorkers": 3,
        "xgboostProcessWorkers": 1,
        "torchJobs": 1,
    }


def _runtime_path(name: str) -> Path:
    path = Path(tempfile.gettempdir()) / "incidentSpot-pytest-temp" / f"api-worker-{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
