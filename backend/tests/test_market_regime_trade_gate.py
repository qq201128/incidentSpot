from __future__ import annotations

import pytest

from app.services import market_regime_trade_gate as gate


def test_trend_up_allows_up_direction(monkeypatch) -> None:
    monkeypatch.setattr(gate, "market_regime_status", lambda *_args: _regime("trend_up"))

    decision = gate.evaluate_market_regime_trade_gate(
        symbol="BTCUSDT",
        duration="10m",
        open_time=1,
        direction="up",
    )

    assert decision.allowed is True
    assert decision.mode == "trend"
    assert decision.reason == "trend_up_aligned"


def test_trend_up_blocks_down_direction(monkeypatch) -> None:
    monkeypatch.setattr(gate, "market_regime_status", lambda *_args: _regime("trend_up"))

    decision = gate.evaluate_market_regime_trade_gate(
        symbol="BTCUSDT",
        duration="10m",
        open_time=1,
        direction="down",
    )

    assert decision.allowed is False
    assert decision.mode == "skip"
    assert decision.reason == "counter_trend_down_vs_up"


def test_trend_down_allows_sell_direction(monkeypatch) -> None:
    monkeypatch.setattr(gate, "market_regime_status", lambda *_args: _regime("trend_down"))

    decision = gate.evaluate_market_regime_trade_gate(
        symbol="BTCUSDT",
        duration="10m",
        open_time=1,
        direction="SELL",
    )

    assert decision.allowed is True
    assert decision.reason == "trend_down_aligned"


def test_range_allows_prediction_direction_as_range_mode(monkeypatch) -> None:
    monkeypatch.setattr(gate, "market_regime_status", lambda *_args: _regime("range"))

    decision = gate.evaluate_market_regime_trade_gate(
        symbol="BTCUSDT",
        duration="10m",
        open_time=1,
        direction="down",
    )

    assert decision.allowed is True
    assert decision.mode == "range"
    assert decision.reason == "range_environment_allowed"


def test_not_ready_regime_skips_order(monkeypatch) -> None:
    monkeypatch.setattr(
        gate,
        "market_regime_status",
        lambda *_args: {"ready": False, "reason": "insufficient_regime_data"},
    )

    decision = gate.evaluate_market_regime_trade_gate(
        symbol="BTCUSDT",
        duration="10m",
        open_time=1,
        direction="up",
    )

    assert decision.allowed is False
    assert decision.reason == "market_regime_not_ready"


def test_uncertain_regime_skips_order(monkeypatch) -> None:
    monkeypatch.setattr(gate, "market_regime_status", lambda *_args: _regime("uncertain"))

    decision = gate.evaluate_market_regime_trade_gate(
        symbol="BTCUSDT",
        duration="10m",
        open_time=1,
        direction="up",
    )

    assert decision.allowed is False
    assert decision.reason == "market_regime_uncertain_skip"


def test_unsupported_direction_raises(monkeypatch) -> None:
    monkeypatch.setattr(gate, "market_regime_status", lambda *_args: _regime("range"))

    with pytest.raises(ValueError, match="unsupported trade direction"):
        gate.evaluate_market_regime_trade_gate(
            symbol="BTCUSDT",
            duration="10m",
            open_time=1,
            direction="flat",
        )


def _regime(trend: str) -> dict:
    return {
        "ready": True,
        "trendState": trend,
        "volatilityState": "normal_vol",
        "regimeLabel": f"{trend}:normal_vol",
        "confidence": 0.8,
        "reasonCodes": [trend, "normal_vol"],
        "metrics": {},
    }
