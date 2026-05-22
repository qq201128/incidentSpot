from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.json_atomic_io import load_json_object, save_json_object


def test_save_json_object_replaces_target_atomically(tmp_path: Path) -> None:
    target = tmp_path / "memory.json"
    save_json_object(target, {"version": 1, "symbol": "BTCUSDT"})
    payload = load_json_object(target)
    assert payload["symbol"] == "BTCUSDT"


def test_load_json_object_retries_while_file_is_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "memory.json"
    save_json_object(target, {"ok": True})

    attempts = {"count": 0}
    real_load = json.load

    def flaky_load(handle):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise json.JSONDecodeError("Expecting ':' delimiter", "", 0)
        return real_load(handle)

    monkeypatch.setattr(json, "load", flaky_load)
    payload = load_json_object(target, retries=3, retry_delay_sec=0.01)
    assert payload["ok"] is True
    assert attempts["count"] == 2
