from __future__ import annotations

from app.services import auto_trade_status


def test_auto_trade_status_exposes_auto_predict_loop(monkeypatch) -> None:
    monkeypatch.setattr(auto_trade_status, "list_auto_trade_settings", lambda: [])
    monkeypatch.setattr(
        auto_trade_status,
        "auto_predict_loop_status",
        lambda: {"status": "failed", "error": "predict failed"},
    )

    payload = auto_trade_status.get_auto_trade_status()

    assert payload["autoPredictLoop"] == {"status": "failed", "error": "predict failed"}
