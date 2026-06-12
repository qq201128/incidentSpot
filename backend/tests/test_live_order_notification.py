from __future__ import annotations

from datetime import datetime, timezone

from app.services import live_order_notification as notification
from app.services.auto_trade_types import AutoTradeSettings


def test_live_order_success_notification_contains_order_details(monkeypatch) -> None:
    sent = []
    notification._LAST_SENT_AT.clear()
    monkeypatch.setenv("WXPUSHER_APP_TOKEN", "AT_test")
    monkeypatch.setenv("WXPUSHER_UIDS", "UID_test")
    monkeypatch.setattr(notification, "WxPusherAppClient", lambda: _Client(sent))

    result = notification.notify_live_order_success(
        _settings(),
        _prediction(),
        {"eventId": 7, "orderId": 8, "externalOrderId": "ex-1", "externalStatus": "PLACED"},
        entry_price=67890.5,
        order_time=datetime(2026, 6, 11, 1, 2, tzinfo=timezone.utc),
    )

    content = sent[0]["content"]
    assert result["sent"] is True
    assert sent[0]["summary"] == "实盘下单成功：BTCUSDT 30m"
    assert "下单时间：2026-06-11T01:02:00+00:00" in content
    assert "下单金额：12.5" in content
    assert "开仓价：67890.5" in content
    assert "币种：BTCUSDT" in content
    assert "周期：30m" in content
    assert "候选：combo__alpha__beta" in content


def test_live_order_failure_notification_contains_error_details(monkeypatch) -> None:
    sent = []
    notification._LAST_SENT_AT.clear()
    monkeypatch.setenv("WXPUSHER_APP_TOKEN", "AT_test")
    monkeypatch.setenv("WXPUSHER_UIDS", "UID_test")
    monkeypatch.setattr(notification, "WxPusherAppClient", lambda: _Client(sent))

    result = notification.notify_live_order_failure(
        _settings(),
        _prediction(),
        RuntimeError("binance order rejected"),
        entry_price=67890.5,
        order_time=datetime(2026, 6, 11, 1, 3, tzinfo=timezone.utc),
    )

    content = sent[0]["content"]
    assert result["sent"] is True
    assert sent[0]["summary"] == "实盘下单失败：BTCUSDT 30m"
    assert "失败类型：RuntimeError" in content
    assert "失败内容：binance order rejected" in content
    assert "候选：combo__alpha__beta" in content


def test_live_order_notification_skips_without_wxpusher_config(monkeypatch) -> None:
    notification._LAST_SENT_AT.clear()
    monkeypatch.delenv("WXPUSHER_APP_TOKEN", raising=False)
    monkeypatch.delenv("WXPUSHER_UIDS", raising=False)
    monkeypatch.delenv("WXPUSHER_TOPIC_IDS", raising=False)

    result = notification.notify_live_order_success(
        _settings(),
        _prediction(),
        {"eventId": 7},
        entry_price=67890.5,
    )

    assert result == {"sent": False, "reason": "wxpusher_app_not_configured"}


def test_live_order_failure_notification_dedupes_same_prediction(monkeypatch) -> None:
    sent = []
    notification._LAST_SENT_AT.clear()
    monkeypatch.setenv("WXPUSHER_APP_TOKEN", "AT_test")
    monkeypatch.setenv("WXPUSHER_UIDS", "UID_test")
    monkeypatch.setenv("LIVE_ORDER_NOTIFICATION_DEDUPE_SECONDS", "3600")
    monkeypatch.setattr(notification, "WxPusherAppClient", lambda: _Client(sent))

    first = notification.notify_live_order_failure(
        _settings(),
        _prediction(),
        RuntimeError("binance 401"),
        entry_price=67890.5,
        order_time=datetime(2026, 6, 11, 1, 3, tzinfo=timezone.utc),
    )
    second = notification.notify_live_order_failure(
        _settings(),
        _prediction(),
        RuntimeError("binance 401"),
        entry_price=67891.0,
        order_time=datetime(2026, 6, 11, 1, 4, tzinfo=timezone.utc),
    )

    assert first["sent"] is True
    assert second == {
        "sent": False,
        "reason": "duplicate_live_order_notification",
        "dedupeSeconds": 3600.0,
    }
    assert len(sent) == 1


def test_live_order_notification_allows_repeat_after_dedupe_window(monkeypatch) -> None:
    sent = []
    notification._LAST_SENT_AT.clear()
    monkeypatch.setenv("WXPUSHER_APP_TOKEN", "AT_test")
    monkeypatch.setenv("WXPUSHER_UIDS", "UID_test")
    monkeypatch.setenv("LIVE_ORDER_NOTIFICATION_DEDUPE_SECONDS", "60")
    monkeypatch.setattr(notification, "WxPusherAppClient", lambda: _Client(sent))

    notification.notify_live_order_failure(
        _settings(),
        _prediction(),
        RuntimeError("binance 401"),
        entry_price=67890.5,
        order_time=datetime(2026, 6, 11, 1, 3, tzinfo=timezone.utc),
    )
    result = notification.notify_live_order_failure(
        _settings(),
        _prediction(),
        RuntimeError("binance 401"),
        entry_price=67891.0,
        order_time=datetime(2026, 6, 11, 1, 5, tzinfo=timezone.utc),
    )

    assert result["sent"] is True
    assert len(sent) == 2


def test_live_order_success_notification_dedupes_same_order(monkeypatch) -> None:
    sent = []
    notification._LAST_SENT_AT.clear()
    monkeypatch.setenv("WXPUSHER_APP_TOKEN", "AT_test")
    monkeypatch.setenv("WXPUSHER_UIDS", "UID_test")
    monkeypatch.setenv("LIVE_ORDER_NOTIFICATION_DEDUPE_SECONDS", "3600")
    monkeypatch.setattr(notification, "WxPusherAppClient", lambda: _Client(sent))

    first = notification.notify_live_order_success(
        _settings(),
        _prediction(),
        {"eventId": 7, "orderId": 8, "externalOrderId": "ex-1", "externalStatus": "PLACED"},
        entry_price=67890.5,
        order_time=datetime(2026, 6, 11, 1, 2, tzinfo=timezone.utc),
    )
    second = notification.notify_live_order_success(
        _settings(),
        _prediction(),
        {"eventId": 7, "orderId": 8, "externalOrderId": "ex-1", "externalStatus": "PLACED"},
        entry_price=67890.5,
        order_time=datetime(2026, 6, 11, 1, 3, tzinfo=timezone.utc),
    )

    assert first["sent"] is True
    assert second["reason"] == "duplicate_live_order_notification"
    assert len(sent) == 1


def _settings() -> AutoTradeSettings:
    return AutoTradeSettings(
        strategy_key="factor_combo_ranker_v1_combo_abcd",
        enabled=True,
        symbol="BTCUSDT",
        duration="30m",
        duration_minutes=30,
        qty=12.5,
        live_trading_enabled=True,
    )


def _prediction() -> dict:
    return {
        "direction": "up",
        "high_winrate_rule": "combo__alpha__beta",
        "open_time": 1_700_000_000_000,
        "probability_up": 0.72,
    }


class _Client:
    def __init__(self, sent: list) -> None:
        self.sent = sent

    def send_markdown(self, *, summary: str, content: str) -> dict:
        self.sent.append({"summary": summary, "content": content})
        return {"code": 1000}
