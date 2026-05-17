from __future__ import annotations

import asyncio

from app.services import auto_predict_service as service
from app.services.auto_trade_types import AutoTradeSettings
from app.services.factor_combo_simulation_keys import (
    factor_combo_shadow_strategy_key,
    simulation_strategy_key_for_factor_name,
)
from app.services.lstm_config import lstm_shadow_strategy_key
from app.services.rule_config import DURATION_TO_MINUTES
from app.services.strategy_prediction_readiness import PredictionReadiness
from app.services.strategy_registry import (
    FACTOR_COMBO_STRATEGY_KEY,
    HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
)

ASYNC_TEST_TIMEOUT_SECONDS = 1.0
DEFAULT_DURATION = "10m"
DEFAULT_QTY = 5.0
ENTRY_OPEN_TIME = 1778121600000


def test_prepare_prediction_inputs_deduplicates_shared_work(monkeypatch) -> None:
    refresh_1m_calls = []
    refresh_duration_calls = []
    settlement_calls = []
    strategy_settings = [
        _settings(FACTOR_COMBO_STRATEGY_KEY),
        _settings(HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY),
        _settings(FACTOR_COMBO_STRATEGY_KEY, symbol="ETHUSDT"),
    ]

    monkeypatch.setattr(service, "_refresh_1m_prediction_input", lambda *args: refresh_1m_calls.append(args))
    monkeypatch.setattr(
        service,
        "_refresh_duration_prediction_input",
        lambda *args: refresh_duration_calls.append(args),
    )
    monkeypatch.setattr(service, "settle_due_predictions", lambda *args: settlement_calls.append(args))
    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )

    asyncio.run(service._prepare_prediction_inputs(strategy_settings))

    assert refresh_1m_calls == [
        ("BTCUSDT", ENTRY_OPEN_TIME),
        ("ETHUSDT", ENTRY_OPEN_TIME),
    ]
    assert refresh_duration_calls == [
        ("BTCUSDT", DEFAULT_DURATION, ENTRY_OPEN_TIME),
        ("ETHUSDT", DEFAULT_DURATION, ENTRY_OPEN_TIME),
    ]
    assert sorted(settlement_calls) == [("BTCUSDT", DEFAULT_DURATION), ("ETHUSDT", DEFAULT_DURATION)]


def test_should_predict_entry_backfills_missing_current_bucket_prediction(monkeypatch) -> None:
    existing_calls = []

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(service, "prediction_exists", lambda **kwargs: existing_calls.append(kwargs) or False)

    assert service._should_predict_entry(_settings(FACTOR_COMBO_STRATEGY_KEY))

    assert existing_calls[0]["strategy_key"] == FACTOR_COMBO_STRATEGY_KEY


def test_should_predict_entry_backfills_ready_lstm_shadow(monkeypatch) -> None:
    calls = []
    lstm_key = lstm_shadow_strategy_key(DEFAULT_DURATION)

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(service, "lstm_model_status", lambda *_args: {"shadowPredictionReady": True})
    monkeypatch.setattr(
        service,
        "eligible_factor_combo_rows",
        lambda *_args: [{"factorName": "combo__a"}, {"factorName": "combo__b"}],
    )
    monkeypatch.setattr(
        service,
        "prediction_exists",
        lambda **kwargs: calls.append(kwargs["strategy_key"]) or kwargs["strategy_key"] != lstm_key,
    )

    assert service._should_predict_entry(_settings(FACTOR_COMBO_STRATEGY_KEY))
    assert calls == [
        FACTOR_COMBO_STRATEGY_KEY,
        simulation_strategy_key_for_factor_name("combo__a"),
        simulation_strategy_key_for_factor_name("combo__b"),
        lstm_key,
    ]


def test_factor_combo_prediction_saves_top_two_and_three_shadow_rows(monkeypatch) -> None:
    saved = []
    trades = []

    async def save_prediction(result: dict, _write_lock: asyncio.Lock, *, allow_existing: bool = False) -> bool:
        saved.append((result["strategy_key"], allow_existing))
        return True

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(
        service,
        "predict_rule_direction",
        lambda symbol, duration, **kwargs: _prediction(kwargs["strategy_key"], symbol=symbol, duration=duration),
    )
    monkeypatch.setattr(service, "predict_eligible_factor_combo_rows", _batch_predictions)
    monkeypatch.setattr(service, "_save_prediction", save_prediction)
    monkeypatch.setattr(service, "create_batch_combo_simulation_trade", lambda settings, result: trades.append(result["strategy_key"]))
    monkeypatch.setattr(service, "prediction_response", lambda result: result)
    monkeypatch.setattr(service, "_broadcast", _noop_broadcast)
    monkeypatch.setattr(
        service,
        "lstm_model_status",
        lambda *_args: {"shadowPredictionReady": False, "shadowPredictionBlockedReason": "torch_unavailable"},
    )

    asyncio.run(service._run_prediction(_settings(FACTOR_COMBO_STRATEGY_KEY), write_lock=asyncio.Lock()))

    assert saved == [
        (FACTOR_COMBO_STRATEGY_KEY, False),
        (simulation_strategy_key_for_factor_name("combo__a"), False),
        (simulation_strategy_key_for_factor_name("combo__b"), False),
    ]
    assert trades == [simulation_strategy_key_for_factor_name("combo__a"), simulation_strategy_key_for_factor_name("combo__b")]


def test_factor_combo_existing_primary_still_saves_lstm_shadow(monkeypatch) -> None:
    saved = []
    broadcasts = []
    lstm_key = lstm_shadow_strategy_key(DEFAULT_DURATION)

    async def save_prediction(result: dict, _write_lock: asyncio.Lock, *, allow_existing: bool = False) -> bool:
        saved.append((result["strategy_key"], allow_existing))
        return result["strategy_key"] != FACTOR_COMBO_STRATEGY_KEY

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(
        service,
        "predict_rule_direction",
        lambda symbol, duration, **kwargs: _prediction(kwargs["strategy_key"], symbol=symbol, duration=duration),
    )
    monkeypatch.setattr(service, "predict_eligible_factor_combo_rows", _batch_predictions)
    monkeypatch.setattr(
        service,
        "predict_lstm_shadow_prediction",
        lambda *_args, **_kwargs: _prediction(lstm_key, symbol="BTCUSDT", duration=DEFAULT_DURATION),
    )
    monkeypatch.setattr(service, "lstm_model_status", lambda *_args: {"shadowPredictionReady": True})
    monkeypatch.setattr(service, "_save_prediction", save_prediction)
    monkeypatch.setattr(service, "_broadcast", lambda result: broadcasts.append(result))

    asyncio.run(service._run_prediction(_settings(FACTOR_COMBO_STRATEGY_KEY), write_lock=asyncio.Lock()))

    assert saved == [
        (FACTOR_COMBO_STRATEGY_KEY, False),
        (simulation_strategy_key_for_factor_name("combo__a"), False),
        (simulation_strategy_key_for_factor_name("combo__b"), False),
        (lstm_key, False),
    ]
    assert broadcasts == []


def test_lstm_strategy_prediction_saves_own_simulation_row(monkeypatch) -> None:
    saved = []
    lstm_key = lstm_shadow_strategy_key("10m")

    async def save_prediction(result: dict, _write_lock: asyncio.Lock, *, allow_existing: bool = False) -> bool:
        saved.append((result["strategy_key"], allow_existing))
        return True

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(
        service,
        "predict_rule_direction",
        lambda symbol, duration, **kwargs: _prediction(kwargs["strategy_key"], symbol=symbol, duration=duration),
    )
    monkeypatch.setattr(service, "_save_prediction", save_prediction)
    monkeypatch.setattr(service, "prediction_response", lambda result: result)
    monkeypatch.setattr(service, "_broadcast", _noop_broadcast)

    asyncio.run(service._run_prediction(_settings(lstm_key), write_lock=asyncio.Lock()))

    assert saved == [(lstm_key, False)]


def test_prediction_targets_include_all_enabled_slots(monkeypatch) -> None:
    mixed = [
        _settings(FACTOR_COMBO_STRATEGY_KEY, duration="30m"),
        _settings(HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, duration="10m"),
    ]
    monkeypatch.setattr(service, "list_auto_trade_settings", lambda: mixed)
    targets = service._prediction_targets()
    keys = {(target.strategy_key, target.duration) for target in targets}
    assert keys == {
        (FACTOR_COMBO_STRATEGY_KEY, "30m"),
        (HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, "10m"),
    }


def test_ready_due_prediction_targets_skip_empty_ranking_cache(monkeypatch) -> None:
    mixed = [
        _settings(FACTOR_COMBO_STRATEGY_KEY, duration="10m"),
        _settings(FACTOR_COMBO_STRATEGY_KEY, duration="30m"),
    ]
    readiness = {
        (FACTOR_COMBO_STRATEGY_KEY, "10m"): _readiness(False, "ranking_cache_empty"),
        (FACTOR_COMBO_STRATEGY_KEY, "30m"): _readiness(True),
    }
    recovery_flags = []

    monkeypatch.setattr(
        service,
        "strategy_prediction_readiness",
        lambda strategy_key, _symbol, duration, **kwargs: recovery_flags.append(kwargs["attempt_recovery"])
        or readiness[(strategy_key, duration)],
    )
    monkeypatch.setattr(service, "_due_prediction_targets", lambda targets: targets)

    targets = service._ready_due_prediction_targets(mixed)

    assert [(target.strategy_key, target.duration) for target in targets] == [
        (FACTOR_COMBO_STRATEGY_KEY, "30m")
    ]
    assert recovery_flags == [True, True]


def test_prediction_targets_do_not_fallback_to_default_when_enabled_targets_invalid(monkeypatch) -> None:
    mixed = [_settings(FACTOR_COMBO_STRATEGY_KEY, duration="10m")]

    monkeypatch.setattr(service, "list_auto_trade_settings", lambda: mixed)

    assert service._prediction_targets() == mixed


def test_prediction_targets_skip_default_when_default_cache_empty(monkeypatch) -> None:
    disabled = [_settings(FACTOR_COMBO_STRATEGY_KEY, duration="10m", enabled=False)]
    recovery_flags = []

    monkeypatch.setattr(service, "list_auto_trade_settings", lambda: disabled)
    monkeypatch.setattr(service, "get_auto_trade_settings", lambda _key: _settings(FACTOR_COMBO_STRATEGY_KEY))
    monkeypatch.setattr(
        service,
        "strategy_prediction_readiness",
        lambda *_args, **kwargs: recovery_flags.append(kwargs["attempt_recovery"])
        or _readiness(False, "ranking_cache_empty"),
    )

    assert service._prediction_targets() == []
    assert recovery_flags == [True]


def test_ready_lstm_shadow_due_syncs_snapshot_mismatch(monkeypatch) -> None:
    calls = []
    statuses = [
        {"shadowPredictionReady": False, "shadowPredictionBlockedReason": "combo_snapshot_mismatch"},
        {"shadowPredictionReady": True, "shadowPredictionBlockedReason": "passed"},
    ]
    settings = _settings(FACTOR_COMBO_STRATEGY_KEY)

    monkeypatch.setattr(service, "prediction_exists", lambda **_kwargs: False)
    monkeypatch.setattr(service, "lstm_model_status", lambda *_args: statuses.pop(0))
    monkeypatch.setattr(service, "get_cached_combination_ranking", lambda *_args: {"ranking": [{"factorName": "combo__a"}]})
    monkeypatch.setattr(
        service,
        "sync_lstm_model_to_combo_ranking",
        lambda symbol, duration, *, ranking_report: calls.append((symbol, duration, ranking_report)) or {"status": "trained"},
    )

    assert service._ready_lstm_shadow_due(settings, ENTRY_OPEN_TIME) is True
    assert calls == [("BTCUSDT", "10m", {"ranking": [{"factorName": "combo__a"}]})]


def test_run_prediction_batch_starts_strategies_concurrently(monkeypatch) -> None:
    strategy_settings = [
        _settings(FACTOR_COMBO_STRATEGY_KEY),
        _settings(HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY),
        _settings(lstm_shadow_strategy_key(DEFAULT_DURATION)),
    ]
    started = []
    completed = []
    expected_count = len(strategy_settings)
    start_gate = {}

    async def run_prediction(setting: AutoTradeSettings, *, write_lock: asyncio.Lock) -> None:
        del write_lock
        started.append(setting.strategy_key)
        if len(started) == expected_count:
            start_gate["event"].set()
        await start_gate["event"].wait()
        completed.append(setting.strategy_key)

    async def run_batch() -> None:
        start_gate["event"] = asyncio.Event()
        monkeypatch.setattr(service, "_run_prediction", run_prediction)
        await asyncio.wait_for(
            service._run_prediction_batch(strategy_settings),
            timeout=ASYNC_TEST_TIMEOUT_SECONDS,
        )

    asyncio.run(run_batch())

    assert set(started) == {settings.strategy_key for settings in strategy_settings}
    assert set(completed) == {settings.strategy_key for settings in strategy_settings}


def _settings(
    strategy_key: str,
    *,
    symbol: str = "BTCUSDT",
    duration: str = DEFAULT_DURATION,
    enabled: bool = True,
) -> AutoTradeSettings:
    return AutoTradeSettings(
        strategy_key=strategy_key,
        enabled=enabled,
        symbol=symbol,
        duration=duration,
        duration_minutes=int(DURATION_TO_MINUTES[duration]),
        qty=DEFAULT_QTY,
        live_trading_enabled=False,
    )


def _prediction(strategy_key: str, *, symbol: str, duration: str) -> dict:
    return {
        "strategy_key": strategy_key,
        "symbol": symbol,
        "duration": duration,
        "open_time": ENTRY_OPEN_TIME,
        "direction": "up",
        "probability_up": 0.55,
        "confidence": 0.55,
        "certainty_label": "FACTOR_COMBO_WAIT",
        "trade_quality_score": 0.5,
        "trade_quality_passed": True,
    }


def _batch_predictions(symbol: str, duration: str, **_kwargs) -> list[dict]:
    return [
        _prediction(simulation_strategy_key_for_factor_name("combo__a"), symbol=symbol, duration=duration),
        _prediction(simulation_strategy_key_for_factor_name("combo__b"), symbol=symbol, duration=duration),
    ]


async def _noop_broadcast(_result: dict) -> None:
    return None


def _readiness(ready: bool, reason: str = "ready"):
    return PredictionReadiness(ready, reason)
