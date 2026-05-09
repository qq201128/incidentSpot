from __future__ import annotations

from datetime import datetime, timezone

from app.services import auto_trade_service
from app.services.kline_timing import KLINE_ENTRY_GRACE_MS
from app.services.auto_trade_types import AutoTradeSettings
from app.services.orderbook_notional_strategy import (
    ORDERBOOK_NOTIONAL_DIFFERENCE_THRESHOLD,
    ORDERBOOK_NOTIONAL_LEVELS_PER_SIDE,
    OrderbookNotionalConfig,
    OrderbookNotionalDependencies,
    evaluate_orderbook_notional,
    predict_orderbook_notional_direction,
)
from app.services.strategy_registry import (
    ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS,
    ORDERBOOK_NOTIONAL_MG_RULE_NAME,
    ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_STRATEGY_KEY,
    strategy_definition,
)

ENTRY_OPEN_TIME = 1778121600000
TEST_LEVELS_PER_SIDE = 3
TEST_MIN_QTY = 1.0
TEST_THRESHOLD = 100.0
TEST_INDEX_PRICE = 50_000.0
DEFAULT_DURATION = "10m"
DEFAULT_DURATION_MINUTES = 10
DEFAULT_QTY = 5.0
MS_PER_SECOND = 1000
ORDERBOOK_NOTIONAL_THRESHOLD = 8_000_000.0
ORDERBOOK_LEVELS_PER_SIDE = 1_000
WITHIN_ORDERBOOK_GRACE_MS = 5_000
OUTSIDE_ORDERBOOK_GRACE_MS = ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS + 5_000


def test_orderbook_notional_strategy_declares_no_policy_gates() -> None:
    strategy = strategy_definition(ORDERBOOK_NOTIONAL_STRATEGY_KEY)

    assert strategy.requires_kline_features is False
    assert strategy.uses_trade_policy_gates is False
    assert strategy.requires_vegas_confirmation is False
    assert strategy.entry_grace_ms == ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS
    assert strategy.entry_grace_ms == KLINE_ENTRY_GRACE_MS


def test_orderbook_notional_default_difference_threshold_is_8m() -> None:
    assert ORDERBOOK_NOTIONAL_DIFFERENCE_THRESHOLD == ORDERBOOK_NOTIONAL_THRESHOLD


def test_orderbook_notional_default_levels_match_binance_depth_limit() -> None:
    assert ORDERBOOK_NOTIONAL_LEVELS_PER_SIDE == ORDERBOOK_LEVELS_PER_SIDE


def test_evaluate_orderbook_notional_filters_quantity_and_selects_bid_side() -> None:
    config = OrderbookNotionalConfig(
        levels_per_side=TEST_LEVELS_PER_SIDE,
        min_qty=TEST_MIN_QTY,
        difference_threshold=TEST_THRESHOLD,
    )
    bids = [["100", "2"], ["90", "1"], ["50", "5"]]
    asks = [["120", "1.1"], ["110", "0.5"], ["100", "1"]]

    evaluation = evaluate_orderbook_notional(bids, asks, config=config)

    assert evaluation.bid_notional == 450.0
    assert evaluation.ask_notional == 132.0
    assert evaluation.direction == "up"
    assert evaluation.dominant_side == "bid"
    assert evaluation.passed is True


def test_predict_orderbook_notional_direction_places_down_when_ask_value_is_larger() -> None:
    config = OrderbookNotionalConfig(
        levels_per_side=TEST_LEVELS_PER_SIDE,
        min_qty=TEST_MIN_QTY,
        difference_threshold=TEST_THRESHOLD,
    )
    dependencies = OrderbookNotionalDependencies(
        fetch_depth=_depth,
        fetch_price=lambda symbol: {"symbol": symbol, "indexPrice": TEST_INDEX_PRICE},
    )

    result = predict_orderbook_notional_direction(
        "btcusdt",
        entry_open_time=ENTRY_OPEN_TIME,
        now_ms=ENTRY_OPEN_TIME + WITHIN_ORDERBOOK_GRACE_MS,
        config=config,
        dependencies=dependencies,
    )

    assert result["strategy_key"] == ORDERBOOK_NOTIONAL_STRATEGY_KEY
    assert result["direction"] == "down"
    assert result["trade_quality_passed"] is True
    assert result["open_time"] == ENTRY_OPEN_TIME
    assert result["entry_price"] == TEST_INDEX_PRICE
    assert result["orderbook"]["dominantSide"] == "ask"
    assert result["orderbook"]["entryWindowPassed"] is True


def test_predict_orderbook_notional_mg_shares_signal_but_distinct_registry() -> None:
    config = OrderbookNotionalConfig(
        levels_per_side=TEST_LEVELS_PER_SIDE,
        min_qty=TEST_MIN_QTY,
        difference_threshold=TEST_THRESHOLD,
    )
    dependencies = OrderbookNotionalDependencies(
        fetch_depth=_depth,
        fetch_price=lambda symbol: {"symbol": symbol, "indexPrice": TEST_INDEX_PRICE},
    )

    result = predict_orderbook_notional_direction(
        "btcusdt",
        entry_open_time=ENTRY_OPEN_TIME,
        now_ms=ENTRY_OPEN_TIME + WITHIN_ORDERBOOK_GRACE_MS,
        result_strategy_key=ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY,
        config=config,
        dependencies=dependencies,
    )

    assert result["strategy_key"] == ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY
    assert result["trade_quality_gate"] == ORDERBOOK_NOTIONAL_MG_RULE_NAME
    assert result["direction"] == "down"


def test_predict_orderbook_notional_requires_entry_window_to_trade() -> None:
    config = OrderbookNotionalConfig(
        levels_per_side=TEST_LEVELS_PER_SIDE,
        min_qty=TEST_MIN_QTY,
        difference_threshold=TEST_THRESHOLD,
    )
    dependencies = OrderbookNotionalDependencies(
        fetch_depth=_depth,
        fetch_price=lambda symbol: {"symbol": symbol, "indexPrice": TEST_INDEX_PRICE},
    )

    result = predict_orderbook_notional_direction(
        "btcusdt",
        entry_open_time=ENTRY_OPEN_TIME,
        now_ms=ENTRY_OPEN_TIME + OUTSIDE_ORDERBOOK_GRACE_MS,
        config=config,
        dependencies=dependencies,
    )

    assert result["trade_quality_passed"] is False
    assert result["certainty_label"] == "ORDERBOOK_NOTIONAL_WAIT"
    assert result["orderbook"]["entryWindowPassed"] is False


def test_auto_trade_uses_orderbook_rule_condition_without_policy_gates(monkeypatch) -> None:
    settings = AutoTradeSettings(
        strategy_key=ORDERBOOK_NOTIONAL_STRATEGY_KEY,
        enabled=True,
        symbol="BTCUSDT",
        duration=DEFAULT_DURATION,
        duration_minutes=DEFAULT_DURATION_MINUTES,
        qty=DEFAULT_QTY,
        live_trading_enabled=False,
    )
    monkeypatch.setattr(auto_trade_service, "trade_policy_payload", _raise_policy_call)

    assert auto_trade_service._is_prediction_tradable(_prediction(True), settings) is True
    assert auto_trade_service._is_prediction_tradable(_prediction(False), settings) is False


def test_auto_trade_limits_orderbook_prediction_to_entry_window() -> None:
    assert auto_trade_service._is_fresh_prediction(
        _prediction_at_entry_age(WITHIN_ORDERBOOK_GRACE_MS),
        ORDERBOOK_NOTIONAL_STRATEGY_KEY,
    )
    assert not auto_trade_service._is_fresh_prediction(
        _prediction_at_entry_age(OUTSIDE_ORDERBOOK_GRACE_MS),
        ORDERBOOK_NOTIONAL_STRATEGY_KEY,
    )


def _depth(symbol: str, levels: int) -> dict:
    return {
        "symbol": symbol,
        "bids": [["100", "2"], ["90", "2"], ["80", "2"]][:levels],
        "asks": [["200", "3"], ["190", "3"], ["180", "3"]][:levels],
        "timestamp": ENTRY_OPEN_TIME,
    }


def _prediction(passed: bool) -> dict:
    return {"probability_up": 0.51, "trade_quality_passed": passed}


def _prediction_at_entry_age(age_ms: int) -> dict:
    now_ms = int(datetime.now(timezone.utc).timestamp() * MS_PER_SECOND)
    return {"open_time": now_ms - age_ms}


def _raise_policy_call(*_args, **_kwargs) -> None:
    raise AssertionError("orderbook notional strategy must not read policy gates")
