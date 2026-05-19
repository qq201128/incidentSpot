from __future__ import annotations

from app.services import lstm_candidate_search_notification as notification


def test_lstm_candidate_notification_skips_without_wxpusher_config(monkeypatch) -> None:
    monkeypatch.delenv("WXPUSHER_APP_TOKEN", raising=False)
    monkeypatch.delenv("WXPUSHER_UIDS", raising=False)
    monkeypatch.delenv("WXPUSHER_TOPIC_IDS", raising=False)

    result = notification.notify_lstm_candidate_search_finished(_report())

    assert result == {"sent": False, "reason": "wxpusher_app_not_configured"}


def test_lstm_candidate_notification_sends_plain_language_summary(monkeypatch) -> None:
    sent = []

    class Client:
        def send_markdown(self, *, summary, content):
            sent.append({"summary": summary, "content": content})
            return {"code": 1000}

    monkeypatch.setenv("WXPUSHER_APP_TOKEN", "AT_test")
    monkeypatch.setenv("WXPUSHER_UIDS", "UID_a")
    monkeypatch.setattr(notification, "WxPusherAppClient", lambda: Client())
    monkeypatch.setattr(
        notification,
        "lstm_candidate_library_summary",
        lambda *_args, **_kwargs: {
            "total": 1,
            "latest": _record("trade_active"),
            "bestTradeCandidate": _record("trade_active"),
            "bestShadowCandidate": None,
        },
    )

    result = notification.notify_lstm_candidate_search_finished(_report())

    assert result["sent"] is True
    assert sent[0]["summary"] == "LSTM候选训练完成：BTCUSDT"
    assert "可下单模型 1 个" in sent[0]["content"]
    assert "找到可下单模型" in sent[0]["content"]


def _report() -> dict:
    return {
        "results": [
            {
                "symbol": "BTCUSDT",
                "duration": "10m",
                "candidates": [{"status": "trade_active"}],
            }
        ]
    }


def _record(status: str) -> dict:
    return {
        "status": status,
        "modelVersion": "lstm_test",
        "config": {"featureWindow": 32, "minMoveBps": 8, "epochs": 8},
        "validation": {"winRate": 0.72},
        "test": {"winRate": 0.71},
    }
