from __future__ import annotations

import pytest

from app.services import auto_trade_execution as execution
from app.services.auto_trade_types import AutoTradeSettings


def test_live_auto_trade_success_sends_notification(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(execution, "_fetch_latest_entry_price", lambda _symbol: 68000.0)
    monkeypatch.setattr(execution, "create_quick_trade_record", lambda _ctx: _trade_result())
    monkeypatch.setattr(execution, "notify_live_order_success", lambda *args, **kwargs: sent.append((args, kwargs)) or {"sent": True})

    result = execution.create_trade_from_prediction(_settings(live=True), _prediction())

    assert result["externalOrderId"] == "ex-1"
    assert sent[0][0][0].symbol == "BTCUSDT"
    assert sent[0][0][1]["high_winrate_rule"] == "combo__alpha__beta"
    assert sent[0][0][2]["orderId"] == 8
    assert sent[0][1]["entry_price"] == 68000.0


def test_live_auto_trade_failure_sends_notification(monkeypatch) -> None:
    sent = []
    error = RuntimeError("binance order rejected")
    monkeypatch.setattr(execution, "_fetch_latest_entry_price", lambda _symbol: 68000.0)
    monkeypatch.setattr(execution, "create_quick_trade_record", lambda _ctx: (_ for _ in ()).throw(error))
    monkeypatch.setattr(execution, "notify_live_order_failure", lambda *args, **kwargs: sent.append((args, kwargs)) or {"sent": True})

    with pytest.raises(RuntimeError, match="binance order rejected"):
        execution.create_trade_from_prediction(_settings(live=True), _prediction())

    assert sent[0][0][0].duration == "30m"
    assert sent[0][0][2] is error
    assert sent[0][1]["entry_price"] == 68000.0


def test_simulated_auto_trade_does_not_send_live_notification(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(execution, "_fetch_latest_entry_price", lambda _symbol: 68000.0)
    monkeypatch.setattr(execution, "create_quick_trade_record", lambda _ctx: _trade_result())
    monkeypatch.setattr(execution, "notify_live_order_success", lambda *_args, **_kwargs: sent.append(True))

    execution.create_trade_from_prediction(_settings(live=False), _prediction())

    assert sent == []


def _settings(*, live: bool) -> AutoTradeSettings:
    return AutoTradeSettings(
        strategy_key="factor_combo_ranker_v1_combo_abcd",
        enabled=True,
        symbol="BTCUSDT",
        duration="30m",
        duration_minutes=30,
        qty=12.5,
        live_trading_enabled=live,
    )


def _prediction() -> dict:
    return {
        "direction": "up",
        "high_winrate_rule": "combo__alpha__beta",
        "open_time": 1_700_000_000_000,
        "probability_up": 0.72,
        "trade_quality_passed": True,
    }


def _trade_result() -> dict:
    return {
        "eventId": 7,
        "orderId": 8,
        "externalOrderId": "ex-1",
        "externalStatus": "PLACED",
    }
