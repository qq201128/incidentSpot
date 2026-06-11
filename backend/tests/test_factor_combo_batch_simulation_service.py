from __future__ import annotations

from app.services import factor_combo_batch_simulation_service as service
from app.services.auto_trade_types import AutoTradeSettings


def test_create_batch_combo_simulation_trade_observes_backtest_failed_prediction(monkeypatch) -> None:
    created = []
    parent = AutoTradeSettings(
        strategy_key="factor_combo_ranker_v1",
        enabled=True,
        symbol="BTCUSDT",
        duration="10m",
        duration_minutes=10,
        qty=5.0,
        live_trading_enabled=False,
    )
    prediction = {
        "strategy_key": "paper_combo_low_winrate",
        "symbol": "BTCUSDT",
        "duration": "10m",
        "open_time": 1_700_000_000_000,
        "direction": "up",
        "probability_up": 0.55,
        "trade_quality_passed": False,
    }

    def create_trade(settings: AutoTradeSettings, row: dict) -> dict:
        created.append((settings, row))
        return {"eventId": 1}

    monkeypatch.setattr(service, "_has_open_position", lambda _settings: False)
    monkeypatch.setattr(service, "_live_trading_enabled", lambda _settings: False)
    monkeypatch.setattr(service, "evaluate_market_regime_trade_gate", _allowed_regime_gate)
    monkeypatch.setattr(service, "create_trade_from_prediction", create_trade)

    result = service.create_batch_combo_simulation_trade(parent, prediction)

    assert result == {"eventId": 1}
    assert created[0][0].strategy_key == "paper_combo_low_winrate"
    assert created[0][0].live_trading_enabled is False
    assert created[0][1]["trade_quality_passed"] is False


def test_create_batch_combo_simulation_trade_skips_when_market_regime_blocks(monkeypatch) -> None:
    created = []
    failures = []
    parent = AutoTradeSettings(
        strategy_key="factor_combo_ranker_v1",
        enabled=True,
        symbol="BTCUSDT",
        duration="10m",
        duration_minutes=10,
        qty=5.0,
        live_trading_enabled=False,
    )
    prediction = {
        "signal_key": "combo_alpha",
        "strategy_key": "paper_combo_alpha",
        "symbol": "BTCUSDT",
        "duration": "10m",
        "open_time": 1_700_000_000_000,
        "direction": "down",
        "probability_up": 0.2,
    }

    monkeypatch.setattr(service, "_has_open_position", lambda _settings: False)
    monkeypatch.setattr(service, "_live_trading_enabled", lambda _settings: False)
    monkeypatch.setattr(service, "evaluate_market_regime_trade_gate", _blocked_regime_gate)
    monkeypatch.setattr(service, "log_prediction_failure", lambda **kwargs: failures.append(kwargs))
    monkeypatch.setattr(service, "create_trade_from_prediction", lambda *_args: created.append(True))

    result = service.create_batch_combo_simulation_trade(parent, prediction)

    assert result is None
    assert created == []
    assert failures[0]["candidate_key"] == "combo_alpha"
    assert failures[0]["stage"] == service.MARKET_REGIME_GATE_STAGE
    assert failures[0]["reason"] == "counter_trend_down_vs_up"
    assert failures[0]["details"]["mode"] == "skip"


def test_create_batch_combo_simulation_trade_skips_live_enabled_candidate(monkeypatch) -> None:
    created = []
    parent = AutoTradeSettings(
        strategy_key="factor_combo_ranker_v1",
        enabled=True,
        symbol="BTCUSDT",
        duration="10m",
        duration_minutes=10,
        qty=5.0,
        live_trading_enabled=False,
    )
    prediction = {
        "strategy_key": "paper_combo_live",
        "symbol": "BTCUSDT",
        "duration": "10m",
        "open_time": 1_700_000_000_000,
        "direction": "up",
        "probability_up": 0.7,
    }

    monkeypatch.setattr(service, "_live_trading_enabled", lambda _settings: True)
    monkeypatch.setattr(service, "_has_open_position", lambda _settings: False)
    monkeypatch.setattr(service, "evaluate_market_regime_trade_gate", _allowed_regime_gate)
    monkeypatch.setattr(service, "create_trade_from_prediction", lambda *_args: created.append(True))

    result = service.create_batch_combo_simulation_trade(parent, prediction)

    assert result is None
    assert created == []


class _RegimeDecision:
    def __init__(self, allowed: bool, reason: str, mode: str) -> None:
        self.allowed = allowed
        self.reason = reason
        self.mode = mode
        self.regime = {"ready": True, "trendState": "trend_up"}


def _allowed_regime_gate(**_kwargs):
    return _RegimeDecision(True, "range_environment_allowed", "range")


def _blocked_regime_gate(**_kwargs):
    return _RegimeDecision(False, "counter_trend_down_vs_up", "skip")
