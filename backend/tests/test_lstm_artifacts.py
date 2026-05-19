from __future__ import annotations

import threading
import time
from pathlib import Path

from app.services import lstm_artifacts

CONCURRENT_WRITE_WINDOW_SECONDS = 0.1
THREAD_JOIN_TIMEOUT_SECONDS = 1.0


def test_write_json_uses_unique_temp_paths(tmp_path, monkeypatch) -> None:
    path = tmp_path / "payload.json"
    sources = []
    replace = lstm_artifacts.os.replace

    def track_replace(source: Path, target: Path) -> None:
        sources.append(Path(source).name)
        replace(source, target)

    monkeypatch.setattr(lstm_artifacts.os, "replace", track_replace)

    lstm_artifacts.write_json(path, {"value": 1})
    lstm_artifacts.write_json(path, {"value": 2})

    assert len(set(sources)) == 2
    assert "payload.json.tmp" not in sources


def test_write_json_waits_for_active_reader(tmp_path, monkeypatch) -> None:
    path = tmp_path / "payload.json"
    path.write_text('{"value": 1}', encoding="utf-8")
    original_load = lstm_artifacts.json.load
    writer_errors = []
    holder = {}

    def write_during_read() -> None:
        try:
            lstm_artifacts.write_json(path, {"value": 2})
        except Exception as exc:  # pragma: no cover - asserted below
            writer_errors.append(exc)

    def slow_load(handle):
        thread = threading.Thread(target=write_during_read)
        holder["thread"] = thread
        thread.start()
        time.sleep(CONCURRENT_WRITE_WINDOW_SECONDS)
        return original_load(handle)

    monkeypatch.setattr(lstm_artifacts.json, "load", slow_load)

    assert lstm_artifacts.read_json(path) == {"value": 1}
    holder["thread"].join(timeout=THREAD_JOIN_TIMEOUT_SECONDS)

    assert writer_errors == []
    assert holder["thread"].is_alive() is False
    assert lstm_artifacts.read_json(path) == {"value": 2}
