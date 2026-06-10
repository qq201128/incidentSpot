from __future__ import annotations

from app.api import event_final_decision as api


def test_read_latest_returns_null_envelope_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(api, "latest_event_final_decision", lambda *_args: None)

    result = api.read_latest_event_final_decision(symbol="btcusdt", duration="10m")

    assert result == {"symbol": "BTCUSDT", "duration": "10m", "latest": None}


def test_read_latest_returns_payload_when_present(monkeypatch) -> None:
    payload = {"symbol": "BTCUSDT", "duration": "10m", "openTime": 1, "decision": "SKIP"}
    monkeypatch.setattr(api, "latest_event_final_decision", lambda *_args: payload)

    result = api.read_latest_event_final_decision(symbol="BTCUSDT", duration="10m")

    assert result == {"symbol": "BTCUSDT", "duration": "10m", "latest": payload}
