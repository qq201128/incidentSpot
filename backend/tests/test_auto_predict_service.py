from __future__ import annotations

import asyncio

import pytest

from app.services import auto_predict_service as service
from app.services.auto_trade_types import AutoTradeSettings
from app.services.factor_combo_simulation_keys import factor_combo_shadow_strategy_key
from app.services.kline_timing import N_BAR_10M_RM_ENTRY_GRACE_MS
from app.services.lstm_config import lstm_shadow_strategy_key
from app.services.rule_config import DURATION_TO_MINUTES
from app.services.strategy_registry import (
    FACTOR_COMBO_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS,
    ORDERBOOK_NOTIONAL_STRATEGY_KEY,
    ORDERBOOK_TRADE_FLOW_STRATEGY_KEY,
    THREE_BAR_10M_RM_STRATEGY_KEY,
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
        _settings(ORDERBOOK_NOTIONAL_STRATEGY_KEY),
        _settings(ORDERBOOK_TRADE_FLOW_STRATEGY_KEY),
        _settings(ORDERBOOK_NOTIONAL_STRATEGY_KEY, symbol="ETHUSDT"),
    ]

    def refresh_1m(symbol: str, entry_open_time: int) -> None:
        refresh_1m_calls.append((symbol, entry_open_time))

    def refresh_duration(symbol: str, duration: str, entry_open_time: int) -> None:
        refresh_duration_calls.append((symbol, duration, entry_open_time))

    def settle(symbol: str, duration: str) -> None:
        settlement_calls.append((symbol, duration))

    monkeypatch.setattr(service, "_refresh_1m_prediction_input", refresh_1m)
    monkeypatch.setattr(service, "_refresh_duration_prediction_input", refresh_duration)
    monkeypatch.setattr(service, "settle_due_predictions", settle)
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


def test_should_predict_entry_uses_strategy_entry_grace(monkeypatch) -> None:
    calls = []
    existing_calls = []
    passed_calls = []

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )

    def is_within_entry_grace(open_time: int, *, grace_ms: int) -> bool:
        calls.append((open_time, grace_ms))
        return True

    def prediction_exists(**kwargs) -> bool:
        existing_calls.append(kwargs)
        return False

    def prediction_passed_exists(**kwargs) -> bool:
        passed_calls.append(kwargs)
        return False

    monkeypatch.setattr(service, "is_within_entry_grace", is_within_entry_grace)
    monkeypatch.setattr(service, "prediction_exists", prediction_exists)
    monkeypatch.setattr(service, "prediction_passed_exists", prediction_passed_exists)

    assert service._should_predict_entry(_settings(ORDERBOOK_NOTIONAL_STRATEGY_KEY))
    assert service._should_predict_entry(_settings(THREE_BAR_10M_RM_STRATEGY_KEY))
    assert calls == [
        (ENTRY_OPEN_TIME, ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS),
        (ENTRY_OPEN_TIME, N_BAR_10M_RM_ENTRY_GRACE_MS),
    ]
    assert [call["strategy_key"] for call in passed_calls] == [ORDERBOOK_NOTIONAL_STRATEGY_KEY]
    assert [call["strategy_key"] for call in existing_calls] == [THREE_BAR_10M_RM_STRATEGY_KEY]


def test_should_predict_entry_retries_orderbook_until_trade_signal(monkeypatch) -> None:
    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(service, "is_within_entry_grace", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(service, "prediction_passed_exists", lambda **_kwargs: True)
    monkeypatch.setattr(service, "prediction_exists", _raise_prediction_exists_call)

    assert not service._should_predict_entry(_settings(ORDERBOOK_NOTIONAL_STRATEGY_KEY))


def test_should_predict_entry_backfills_ready_lstm_shadow(monkeypatch) -> None:
    calls = []
    lstm_key = lstm_shadow_strategy_key(DEFAULT_DURATION)

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(service, "is_within_entry_grace", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(service, "lstm_model_status", lambda *_args: {"shadowPredictionReady": True})

    def prediction_exists(**kwargs) -> bool:
        calls.append(kwargs["strategy_key"])
        return kwargs["strategy_key"] != lstm_key

    monkeypatch.setattr(service, "prediction_exists", prediction_exists)

    assert service._should_predict_entry(_settings(FACTOR_COMBO_STRATEGY_KEY))
    assert calls == [
        FACTOR_COMBO_STRATEGY_KEY,
        factor_combo_shadow_strategy_key(2),
        factor_combo_shadow_strategy_key(3),
        lstm_key,
    ]


def test_orderbook_prediction_allows_existing_attempt_rows(monkeypatch) -> None:
    allow_existing_flags = []

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )

    async def run_prediction(strategy_key: str) -> None:
        await service._run_prediction(
            _settings(strategy_key),
            write_lock=asyncio.Lock(),
        )

    def predict_rule_direction(symbol: str, duration: str, **kwargs) -> dict:
        return _prediction(kwargs["strategy_key"], symbol=symbol, duration=duration)

    async def save_prediction(_result: dict, _write_lock: asyncio.Lock, *, allow_existing: bool = False) -> bool:
        allow_existing_flags.append(allow_existing)
        return True

    monkeypatch.setattr(service, "predict_rule_direction", predict_rule_direction)
    monkeypatch.setattr(service, "_save_prediction", save_prediction)
    monkeypatch.setattr(service, "prediction_response", lambda result: result)
    monkeypatch.setattr(service, "_broadcast", _noop_broadcast)

    asyncio.run(run_prediction(ORDERBOOK_NOTIONAL_STRATEGY_KEY))
    asyncio.run(run_prediction(THREE_BAR_10M_RM_STRATEGY_KEY))

    assert allow_existing_flags == [True, False]


def test_factor_combo_prediction_saves_top_two_and_three_shadow_rows(monkeypatch) -> None:
    saved = []

    async def save_prediction(result: dict, _write_lock: asyncio.Lock, *, allow_existing: bool = False) -> bool:
        saved.append((result["strategy_key"], allow_existing))
        return True

    def predict_rule_direction(symbol: str, duration: str, **kwargs) -> dict:
        return _prediction(kwargs["strategy_key"], symbol=symbol, duration=duration)

    def predict_factor_combo_rank_direction(symbol: str, duration: str, **kwargs) -> dict:
        return _prediction(kwargs["result_strategy_key"], symbol=symbol, duration=duration)

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(service, "predict_rule_direction", predict_rule_direction)
    monkeypatch.setattr(service, "predict_factor_combo_rank_direction", predict_factor_combo_rank_direction)
    monkeypatch.setattr(service, "_save_prediction", save_prediction)
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
        (factor_combo_shadow_strategy_key(2), False),
        (factor_combo_shadow_strategy_key(3), False),
    ]


def test_factor_combo_existing_primary_still_saves_lstm_shadow(monkeypatch) -> None:
    saved = []
    broadcasts = []
    lstm_key = lstm_shadow_strategy_key(DEFAULT_DURATION)

    async def save_prediction(result: dict, _write_lock: asyncio.Lock, *, allow_existing: bool = False) -> bool:
        saved.append((result["strategy_key"], allow_existing))
        return result["strategy_key"] != FACTOR_COMBO_STRATEGY_KEY

    def predict_rule_direction(symbol: str, duration: str, **kwargs) -> dict:
        return _prediction(kwargs["strategy_key"], symbol=symbol, duration=duration)

    def predict_factor_combo_rank_direction(symbol: str, duration: str, **kwargs) -> dict:
        return _prediction(kwargs["result_strategy_key"], symbol=symbol, duration=duration)

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(service, "predict_rule_direction", predict_rule_direction)
    monkeypatch.setattr(service, "predict_factor_combo_rank_direction", predict_factor_combo_rank_direction)
    monkeypatch.setattr(
        service,
        "predict_lstm_shadow_prediction",
        lambda *_a, **_k: _prediction(lstm_key, symbol="BTCUSDT", duration=DEFAULT_DURATION),
    )
    monkeypatch.setattr(service, "lstm_model_status", lambda *_args: {"shadowPredictionReady": True})
    monkeypatch.setattr(service, "_save_prediction", save_prediction)
    monkeypatch.setattr(service, "_broadcast", lambda result: broadcasts.append(result))

    asyncio.run(service._run_prediction(_settings(FACTOR_COMBO_STRATEGY_KEY), write_lock=asyncio.Lock()))

    assert saved == [
        (FACTOR_COMBO_STRATEGY_KEY, False),
        (factor_combo_shadow_strategy_key(2), False),
        (factor_combo_shadow_strategy_key(3), False),
        (lstm_key, False),
    ]
    assert broadcasts == []


def test_lstm_strategy_prediction_saves_own_simulation_row(monkeypatch) -> None:
    saved = []
    lstm_key = lstm_shadow_strategy_key("10m")

    async def save_prediction(result: dict, _write_lock: asyncio.Lock, *, allow_existing: bool = False) -> bool:
        saved.append((result["strategy_key"], allow_existing))
        return True

    def predict_rule_direction(symbol: str, duration: str, **kwargs) -> dict:
        return _prediction(kwargs["strategy_key"], symbol=symbol, duration=duration)

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(service, "predict_rule_direction", predict_rule_direction)
    monkeypatch.setattr(service, "_save_prediction", save_prediction)
    monkeypatch.setattr(service, "prediction_response", lambda result: result)
    monkeypatch.setattr(service, "_broadcast", _noop_broadcast)

    asyncio.run(service._run_prediction(_settings(lstm_key), write_lock=asyncio.Lock()))

    assert saved == [(lstm_key, False)]


def test_unready_lstm_strategy_is_not_due(monkeypatch) -> None:
    calls = []
    lstm_key = lstm_shadow_strategy_key("10m")

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(service, "is_within_entry_grace", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        service,
        "lstm_model_status",
        lambda *_args: {
            "shadowPredictionReady": False,
            "shadowPredictionBlockedReason": "validation_gate_missing",
        },
    )

    def prediction_exists(**kwargs) -> bool:
        calls.append(kwargs["strategy_key"])
        return False

    monkeypatch.setattr(service, "prediction_exists", prediction_exists)

    assert not service._should_predict_entry(_settings(lstm_key))
    assert calls == [lstm_key]


def test_prediction_targets_include_all_enabled_slots(monkeypatch) -> None:
    mixed = [
        _settings(THREE_BAR_10M_RM_STRATEGY_KEY, duration="30m"),
        _settings(ORDERBOOK_NOTIONAL_STRATEGY_KEY, duration="10m"),
    ]
    monkeypatch.setattr(service, "list_auto_trade_settings", lambda: mixed)
    targets = service._prediction_targets()
    assert len(targets) == 2
    keys = {(t.strategy_key, t.duration) for t in targets}
    assert keys == {
        (THREE_BAR_10M_RM_STRATEGY_KEY, "30m"),
        (ORDERBOOK_NOTIONAL_STRATEGY_KEY, "10m"),
    }


def test_next_predict_wait_polls_during_entry_window(monkeypatch) -> None:
    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(service, "is_within_entry_grace", lambda *_a, **_k: True)

    assert service._next_predict_wait([_settings(ORDERBOOK_NOTIONAL_STRATEGY_KEY)], 1) == 1.0


def test_run_prediction_batch_starts_strategies_concurrently(monkeypatch) -> None:
    strategy_settings = [
        _settings(ORDERBOOK_NOTIONAL_STRATEGY_KEY),
        _settings(ORDERBOOK_TRADE_FLOW_STRATEGY_KEY),
        _settings(THREE_BAR_10M_RM_STRATEGY_KEY),
    ]
    started = []
    completed = []
    expected_count = len(strategy_settings)

    async def run_batch() -> None:
        all_started = asyncio.Event()

        async def run_prediction(
            setting: AutoTradeSettings,
            *,
            write_lock: asyncio.Lock,
        ) -> None:
            started.append(setting.strategy_key)
            if len(started) == expected_count:
                all_started.set()
            await all_started.wait()
            completed.append(setting.strategy_key)

        monkeypatch.setattr(service, "_run_prediction", run_prediction)
        await asyncio.wait_for(
            service._run_prediction_batch(strategy_settings),
            timeout=ASYNC_TEST_TIMEOUT_SECONDS,
        )

    asyncio.run(run_batch())

    assert set(started) == {settings.strategy_key for settings in strategy_settings}
    assert set(completed) == {settings.strategy_key for settings in strategy_settings}


def test_run_prediction_batch_reports_failures_after_batch_finishes(monkeypatch) -> None:
    strategy_settings = [_settings("good_one"), _settings("bad_one"), _settings("good_two")]
    completed = []

    async def run_batch() -> None:
        async def run_prediction(
            setting: AutoTradeSettings,
            *,
            write_lock: asyncio.Lock,
        ) -> None:
            await asyncio.sleep(0)
            completed.append(setting.strategy_key)
            if setting.strategy_key == "bad_one":
                raise ValueError("boom")

        monkeypatch.setattr(service, "_run_prediction", run_prediction)
        with pytest.raises(RuntimeError, match="bad_one"):
            await asyncio.wait_for(
                service._run_prediction_batch(strategy_settings),
                timeout=ASYNC_TEST_TIMEOUT_SECONDS,
            )

    asyncio.run(run_batch())

    assert set(completed) == {settings.strategy_key for settings in strategy_settings}


def _settings(
    strategy_key: str,
    *,
    symbol: str = "BTCUSDT",
    duration: str = DEFAULT_DURATION,
) -> AutoTradeSettings:
    return AutoTradeSettings(
        strategy_key=strategy_key,
        enabled=True,
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
        "certainty_label": "ORDERBOOK_NOTIONAL_WAIT",
        "trade_quality_score": 0.5,
        "trade_quality_passed": False,
    }


async def _noop_broadcast(_result: dict) -> None:
    return None


def _raise_prediction_exists_call(**_kwargs) -> bool:
    raise AssertionError("continuous orderbook strategy must check passed predictions only")
