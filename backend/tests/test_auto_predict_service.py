from __future__ import annotations

import asyncio

import pytest

from app.services import auto_predict_service as service
from app.services.auto_trade_types import AutoTradeSettings
from app.services.kline_timing import KLINE_ENTRY_GRACE_MS
from app.services.strategy_registry import (
    ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS,
    ORDERBOOK_NOTIONAL_STRATEGY_KEY,
)

ASYNC_TEST_TIMEOUT_SECONDS = 1.0
DEFAULT_DURATION = "10m"
DEFAULT_DURATION_MINUTES = 10
DEFAULT_QTY = 5.0
ENTRY_OPEN_TIME = 1778121600000


def test_prepare_prediction_inputs_deduplicates_shared_work(monkeypatch) -> None:
    refresh_calls = []
    settlement_calls = []
    strategy_settings = [
        _settings("vegas_fib_resonance"),
        _settings(ORDERBOOK_NOTIONAL_STRATEGY_KEY),
        _settings("daily_trade_floor_tree"),
        _settings("high_winrate_rules", symbol="ETHUSDT"),
    ]

    def refresh(symbol: str, entry_open_time: int) -> None:
        refresh_calls.append((symbol, entry_open_time))

    def settle(symbol: str, duration: str) -> None:
        settlement_calls.append((symbol, duration))

    monkeypatch.setattr(service, "_refresh_prediction_input", refresh)
    monkeypatch.setattr(service, "settle_due_predictions", settle)

    asyncio.run(service._prepare_prediction_inputs(strategy_settings, ENTRY_OPEN_TIME))

    assert sorted(refresh_calls) == [
        ("BTCUSDT", ENTRY_OPEN_TIME),
        ("ETHUSDT", ENTRY_OPEN_TIME),
    ]
    assert sorted(settlement_calls) == [("BTCUSDT", DEFAULT_DURATION), ("ETHUSDT", DEFAULT_DURATION)]


def test_should_predict_entry_uses_strategy_entry_grace(monkeypatch) -> None:
    calls = []
    existing_calls = []
    passed_calls = []

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

    assert service._should_predict_entry(_settings(ORDERBOOK_NOTIONAL_STRATEGY_KEY), ENTRY_OPEN_TIME)
    assert service._should_predict_entry(_settings("vegas_fib_resonance"), ENTRY_OPEN_TIME)
    assert calls == [
        (ENTRY_OPEN_TIME, ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS),
        (ENTRY_OPEN_TIME, KLINE_ENTRY_GRACE_MS),
    ]
    assert [call["strategy_key"] for call in passed_calls] == [ORDERBOOK_NOTIONAL_STRATEGY_KEY]
    assert [call["strategy_key"] for call in existing_calls] == ["vegas_fib_resonance"]


def test_should_predict_entry_retries_orderbook_until_trade_signal(monkeypatch) -> None:
    monkeypatch.setattr(service, "is_within_entry_grace", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(service, "prediction_passed_exists", lambda **_kwargs: True)
    monkeypatch.setattr(service, "prediction_exists", _raise_prediction_exists_call)

    assert not service._should_predict_entry(_settings(ORDERBOOK_NOTIONAL_STRATEGY_KEY), ENTRY_OPEN_TIME)


def test_orderbook_prediction_allows_existing_attempt_rows(monkeypatch) -> None:
    allow_existing_flags = []

    async def run_prediction(strategy_key: str) -> None:
        await service._run_prediction(
            _settings(strategy_key),
            ENTRY_OPEN_TIME,
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
    asyncio.run(run_prediction("vegas_fib_resonance"))

    assert allow_existing_flags == [True, False]


def test_next_predict_wait_polls_during_entry_window(monkeypatch) -> None:
    monkeypatch.setattr(service, "current_rule_entry_open_time", lambda: ENTRY_OPEN_TIME)
    monkeypatch.setattr(service, "is_within_entry_grace", lambda _open_time: True)

    assert service._next_predict_wait(1) == 1.0


def test_run_prediction_batch_starts_strategies_concurrently(monkeypatch) -> None:
    strategy_settings = [
        _settings("vegas_fib_resonance"),
        _settings("high_winrate_rules"),
        _settings("daily_trade_floor_tree"),
    ]
    started = []
    completed = []
    expected_count = len(strategy_settings)

    async def run_batch() -> None:
        all_started = asyncio.Event()

        async def run_prediction(
            setting: AutoTradeSettings,
            entry_open_time: int,
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
            service._run_prediction_batch(strategy_settings, ENTRY_OPEN_TIME),
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
            entry_open_time: int,
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
                service._run_prediction_batch(strategy_settings, ENTRY_OPEN_TIME),
                timeout=ASYNC_TEST_TIMEOUT_SECONDS,
            )

    asyncio.run(run_batch())

    assert set(completed) == {settings.strategy_key for settings in strategy_settings}


def _settings(strategy_key: str, *, symbol: str = "BTCUSDT") -> AutoTradeSettings:
    return AutoTradeSettings(
        strategy_key=strategy_key,
        enabled=True,
        symbol=symbol,
        duration=DEFAULT_DURATION,
        duration_minutes=DEFAULT_DURATION_MINUTES,
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
