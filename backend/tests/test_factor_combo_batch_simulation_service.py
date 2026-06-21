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

    def create_trade(settings: AutoTradeSettings, row: dict, **_kwargs) -> dict:
        created.append((settings, row))
        return {"eventId": 1}

    monkeypatch.setattr(service, "_has_open_position", lambda _settings: False)
    monkeypatch.setattr(service, "_live_trading_enabled", lambda _settings: False)
    monkeypatch.setattr(service, "evaluate_candidate_regime_admission", _allowed_regime_admission)
    monkeypatch.setattr(service, "create_trade_from_prediction", create_trade)

    result = service.create_batch_combo_simulation_trade(parent, prediction)

    assert result == {"eventId": 1}
    assert created[0][0].strategy_key == "paper_combo_low_winrate"
    assert created[0][0].live_trading_enabled is False
    assert created[0][1]["trade_quality_passed"] is False


def test_create_batch_combo_simulation_trade_marks_prediction_event(monkeypatch) -> None:
    created = []
    marked = []
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
        "id": 42,
        "strategy_key": "paper_combo_alpha",
        "symbol": "BTCUSDT",
        "duration": "10m",
        "open_time": 1_700_000_000_000,
        "direction": "up",
        "probability_up": 0.7,
    }

    def create_trade(settings: AutoTradeSettings, row: dict, **kwargs) -> dict:
        created.append((settings, row, kwargs))
        return {"eventId": 99}

    monkeypatch.setattr(service, "_has_open_position", lambda _settings: False)
    monkeypatch.setattr(service, "_live_trading_enabled", lambda _settings: False)
    monkeypatch.setattr(service, "evaluate_candidate_regime_admission", _allowed_regime_admission)
    monkeypatch.setattr(service, "create_trade_from_prediction", create_trade)
    monkeypatch.setattr(service, "mark_prediction_execution", lambda *args, **kwargs: marked.append((args, kwargs)))

    result = service.create_batch_combo_simulation_trade(parent, prediction)

    assert result == {"eventId": 99}
    assert created[0][1]["id"] == 42
    assert created[0][2]["regime_decision"].reason == "regime_exploration_sample_count_below_50"
    assert marked == [
        (
            (42,),
            {
                "status": service.EXECUTION_EVENT_CREATED,
                "reason": "regime_exploration_sample_count_below_50",
                "event_id": 99,
            },
        )
    ]


def test_create_batch_combo_simulation_trade_skips_when_candidate_regime_bucket_blocks(monkeypatch) -> None:
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
    monkeypatch.setattr(service, "evaluate_candidate_regime_admission", _blocked_regime_admission)
    monkeypatch.setattr(service, "log_prediction_failure", lambda **kwargs: failures.append(kwargs))
    monkeypatch.setattr(service, "create_trade_from_prediction", lambda *_args: created.append(True))

    result = service.create_batch_combo_simulation_trade(parent, prediction)

    assert result is None
    assert created == []
    assert failures[0]["candidate_key"] == "combo_alpha"
    assert failures[0]["stage"] == service.CANDIDATE_REGIME_ADMISSION_STAGE
    assert failures[0]["reason"] == "regime_bucket_win_rate_below_min"
    assert failures[0]["details"]["mode"] == "evaluable"
    assert failures[0]["details"]["sampleCount"] == 120


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
    monkeypatch.setattr(service, "evaluate_candidate_regime_admission", _allowed_regime_admission)
    monkeypatch.setattr(service, "create_trade_from_prediction", lambda *_args: created.append(True))

    result = service.create_batch_combo_simulation_trade(parent, prediction)

    assert result is None
    assert created == []


class _RegimeAdmission:
    def __init__(self, allowed: bool, reason: str, mode: str, sample_count: int = 0) -> None:
        self.allowed = allowed
        self.reason = reason
        self.mode = mode
        self.regime = {"ready": True, "trendState": "trend_down", "regimeLabel": "trend_down:normal_vol"}
        self.sample_count = sample_count
        self.metrics = {"sampleCount": sample_count}
        self.version = "candidate_regime_admission_v1"


def _allowed_regime_admission(_prediction):
    return _RegimeAdmission(True, "regime_exploration_sample_count_below_50", "exploration")


def _blocked_regime_admission(_prediction):
    return _RegimeAdmission(False, "regime_bucket_win_rate_below_min", "evaluable", sample_count=120)
