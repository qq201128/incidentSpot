from __future__ import annotations

from app.services import auto_trade_service
from app.services.auto_trade_types import AutoTradeSettings
from app.services.kline_timing import KLINE_ENTRY_GRACE_MS
from app.services.orderbook_trade_flow_strategy import (
    OrderbookTradeFlowConfig,
    OrderbookTradeFlowDependencies,
    predict_orderbook_trade_flow_direction,
)
from app.services.strategy_registry import (
    BLIND_REVERSE_MARTINGALE_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY,
    ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY,
    ORDERBOOK_TRADE_FLOW_STRATEGY_KEY,
    is_continuous_orderbook_strategy,
    strategy_definition,
)

ENTRY_OPEN_TIME = 1778121600000
TEST_INDEX_PRICE = 50_000.0
DEFAULT_DURATION = "10m"
DEFAULT_DURATION_MINUTES = 10
DEFAULT_QTY = 5.0
WITHIN_GRACE_MS = 5_000
OUTSIDE_GRACE_MS = KLINE_ENTRY_GRACE_MS + 5_000


def test_trade_flow_strategy_registry_flags() -> None:
    strategy = strategy_definition(ORDERBOOK_TRADE_FLOW_STRATEGY_KEY)

    assert strategy.requires_kline_features is False
    assert strategy.uses_trade_policy_gates is False
    assert strategy.requires_vegas_confirmation is False
    assert strategy.entry_grace_ms == KLINE_ENTRY_GRACE_MS


def test_is_continuous_orderbook_strategy_includes_trade_flow() -> None:
    assert is_continuous_orderbook_strategy(ORDERBOOK_TRADE_FLOW_STRATEGY_KEY)
    assert is_continuous_orderbook_strategy(ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY)
    assert is_continuous_orderbook_strategy(ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY)
    assert is_continuous_orderbook_strategy(BLIND_REVERSE_MARTINGALE_STRATEGY_KEY)
    assert not is_continuous_orderbook_strategy("vegas_fib_resonance")


def test_predict_trade_flow_combines_book_and_flow() -> None:
    cfg = OrderbookTradeFlowConfig(
        levels_per_side=3,
        snapshot_interval_sec=0.0,
        agg_trades_limit=10,
        min_total_quote_flow=100.0,
        combined_threshold=0.05,
    )
    dependencies = OrderbookTradeFlowDependencies(
        fetch_depth=_make_depth_pair(),
        fetch_agg_trades=_agg_trades_buy_heavy,
        fetch_price=lambda _s: {"indexPrice": TEST_INDEX_PRICE},
    )

    result = predict_orderbook_trade_flow_direction(
        "btcusdt",
        entry_open_time=ENTRY_OPEN_TIME,
        now_ms=ENTRY_OPEN_TIME + WITHIN_GRACE_MS,
        config=cfg,
        dependencies=dependencies,
    )

    assert result["strategy_key"] == ORDERBOOK_TRADE_FLOW_STRATEGY_KEY
    assert result["direction"] == "up"
    assert result["open_time"] == ENTRY_OPEN_TIME
    assert result["trade_quality_passed"] is True
    assert result["orderbook"]["combinedScore"] > 0


def test_predict_trade_flow_invert_flips_direction() -> None:
    cfg = OrderbookTradeFlowConfig(
        levels_per_side=3,
        snapshot_interval_sec=0.0,
        agg_trades_limit=10,
        min_total_quote_flow=100.0,
        combined_threshold=0.05,
    )
    dependencies = OrderbookTradeFlowDependencies(
        fetch_depth=_make_depth_pair(),
        fetch_agg_trades=_agg_trades_buy_heavy,
        fetch_price=lambda _s: {"indexPrice": TEST_INDEX_PRICE},
    )

    result = predict_orderbook_trade_flow_direction(
        "btcusdt",
        entry_open_time=ENTRY_OPEN_TIME,
        now_ms=ENTRY_OPEN_TIME + WITHIN_GRACE_MS,
        config=cfg,
        dependencies=dependencies,
        result_strategy_key=ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY,
    )

    assert result["strategy_key"] == ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY
    assert result["direction"] == "down"
    assert result["orderbook"]["invertDirection"] is True


def test_predict_trade_flow_requires_entry_window_for_trade_flag() -> None:
    cfg = OrderbookTradeFlowConfig(
        levels_per_side=3,
        snapshot_interval_sec=0.0,
        agg_trades_limit=10,
        min_total_quote_flow=100.0,
        combined_threshold=0.05,
    )
    dependencies = OrderbookTradeFlowDependencies(
        fetch_depth=_make_depth_pair(),
        fetch_agg_trades=_agg_trades_buy_heavy,
        fetch_price=lambda _s: {"indexPrice": TEST_INDEX_PRICE},
    )

    result = predict_orderbook_trade_flow_direction(
        "btcusdt",
        entry_open_time=ENTRY_OPEN_TIME,
        now_ms=ENTRY_OPEN_TIME + OUTSIDE_GRACE_MS,
        config=cfg,
        dependencies=dependencies,
    )

    assert result["trade_quality_passed"] is False
    assert result["orderbook"]["entryWindowPassed"] is False


def test_auto_trade_skips_policy_payload_for_trade_flow(monkeypatch) -> None:
    settings = AutoTradeSettings(
        strategy_key=ORDERBOOK_TRADE_FLOW_STRATEGY_KEY,
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


def _make_depth_pair():
    calls = {"n": 0}

    def depth(symbol: str, levels: int) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            bids = [["100", "10"], ["99", "10"], ["98", "10"]][:levels]
            asks = [["101", "10"], ["102", "10"], ["103", "10"]][:levels]
        else:
            bids = [["100", "14"], ["99", "10"], ["98", "10"]][:levels]
            asks = [["101", "6"], ["102", "10"], ["103", "10"]][:levels]
        return {"symbol": symbol, "bids": bids, "asks": asks, "timestamp": ENTRY_OPEN_TIME}

    return depth


def _agg_trades_buy_heavy(symbol: str, limit: int) -> list[dict]:
    rows = []
    for i in range(min(20, limit)):
        rows.append({"quoteQty": 5000.0 + i, "side": "buy"})
        rows.append({"quoteQty": 1000.0, "side": "sell"})
    return rows


def _prediction(passed: bool) -> dict:
    return {"probability_up": 0.51, "trade_quality_passed": passed}


def _raise_policy_call(*_args, **_kwargs) -> None:
    raise AssertionError("trade flow strategy must not read policy gates")
