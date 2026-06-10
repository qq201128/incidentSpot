from __future__ import annotations

from app.services import auto_trade_status
from app.services.auto_trade_types import AutoTradeSettings


def test_auto_trade_status_exposes_auto_predict_loop(monkeypatch) -> None:
    monkeypatch.setattr(auto_trade_status, "list_auto_trade_settings", lambda: [])
    monkeypatch.setattr(
        auto_trade_status,
        "auto_predict_loop_status",
        lambda: {"status": "failed", "error": "predict failed"},
    )

    payload = auto_trade_status.get_auto_trade_status()

    assert payload["autoPredictLoop"] == {"status": "failed", "error": "predict failed"}


def test_auto_trade_status_exposes_market_regime(monkeypatch) -> None:
    settings = _settings()
    prediction = _prediction()
    regime = {
        "ready": True,
        "trendState": "trend_down",
        "volatilityState": "normal_vol",
        "regimeLabel": "trend_down:normal_vol",
        "confidence": 0.82,
        "reasonCodes": ["trend_down", "normal_vol"],
        "metrics": {"regime_slow_slope": -0.001},
    }

    monkeypatch.setattr(auto_trade_status, "list_auto_trade_settings", lambda: [settings])
    monkeypatch.setattr(auto_trade_status, "_runtime_data", lambda _settings: _runtime(settings, prediction))
    monkeypatch.setattr(auto_trade_status, "auto_predict_loop_status", lambda: {})
    monkeypatch.setattr(auto_trade_status, "market_regime_status", lambda *_args: regime)

    payload = auto_trade_status.get_auto_trade_status()

    assert payload["latestPrediction"]["marketRegime"] == regime


def test_auto_trade_status_exposes_market_regime_data_gap(monkeypatch) -> None:
    settings = _settings()

    monkeypatch.setattr(auto_trade_status, "list_auto_trade_settings", lambda: [settings])
    monkeypatch.setattr(auto_trade_status, "_runtime_data", lambda _settings: _runtime(settings, _prediction()))
    monkeypatch.setattr(auto_trade_status, "auto_predict_loop_status", lambda: {})
    monkeypatch.setattr(
        auto_trade_status,
        "market_regime_status",
        lambda *_args: {"ready": False, "reason": "insufficient_regime_data"},
    )

    payload = auto_trade_status.get_auto_trade_status()

    assert payload["latestPrediction"]["marketRegime"] == {
        "ready": False,
        "reason": "insufficient_regime_data",
    }


def test_auto_trade_status_skips_disabled_runtime_queries(monkeypatch) -> None:
    settings = AutoTradeSettings(
        strategy_key="factor_combo_ranker_v1",
        enabled=False,
        symbol="BTCUSDT",
        duration="10m",
        duration_minutes=10,
        qty=5.0,
        live_trading_enabled=False,
    )

    monkeypatch.setattr(auto_trade_status, "list_auto_trade_settings", lambda: [settings])
    monkeypatch.setattr(auto_trade_status, "auto_predict_loop_status", lambda: {})
    monkeypatch.setattr(
        auto_trade_status,
        "_runtime_data",
        lambda _settings: auto_trade_status.StatusRuntimeData({}, set()),
    )
    monkeypatch.setattr(
        auto_trade_status,
        "_latest_prediction",
        lambda _settings: (_ for _ in ()).throw(AssertionError("prediction query should be skipped")),
    )
    monkeypatch.setattr(
        auto_trade_status,
        "_has_open_position",
        lambda _settings: (_ for _ in ()).throw(AssertionError("position query should be skipped")),
    )

    payload = auto_trade_status.get_auto_trade_status()

    assert payload["latestPrediction"] is None
    assert payload["reason"] == "disabled"


def _settings() -> AutoTradeSettings:
    return AutoTradeSettings(
        strategy_key="factor_combo_ranker_v1",
        enabled=True,
        symbol="BTCUSDT",
        duration="10m",
        duration_minutes=10,
        qty=5.0,
        live_trading_enabled=False,
    )


def _prediction() -> dict:
    return {
        "id": 1,
        "created_at": "2026-06-09T00:00:00+00:00",
        "open_time": 123,
        "direction": "down",
        "probability_up": 0.45,
        "trade_quality_score": 0.7,
        "trade_quality_passed": 1,
        "high_winrate_gate": "factor_combo_cached_ranking_v1",
        "high_winrate_gate_passed": 1,
        "high_winrate_gate_value": 0.58,
    }


def _runtime(settings: AutoTradeSettings, prediction: dict) -> auto_trade_status.StatusRuntimeData:
    key = (settings.strategy_key, settings.symbol.upper(), settings.duration)
    return auto_trade_status.StatusRuntimeData({key: prediction}, set())
