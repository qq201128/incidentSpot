from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from app.services.binance_orderbook_depth import fetch_orderbook_depth_levels
from app.services.binance_service import fetch_agg_trades_display, fetch_premium_index
from app.services.kline_timing import is_within_entry_grace
from app.services.rule_config import RULE_DURATION, SUPPORTED_RULE_DURATIONS
from app.services.strategy_registry import (
    ORDERBOOK_TRADE_FLOW_ENTRY_GRACE_MS,
    ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY,
    ORDERBOOK_TRADE_FLOW_RULE_NAME,
    ORDERBOOK_TRADE_FLOW_STRATEGY_KEY,
    StrategyDefinition,
    strategy_definition,
)

LEVELS_PER_SIDE = 1_000
SNAPSHOT_INTERVAL_SEC = 0.25
AGG_TRADES_LIMIT = 500
MIN_TOTAL_QUOTE_FLOW = 10_000.0
BOOK_WEIGHT = 0.5
FLOW_WEIGHT = 0.5
COMBINED_THRESHOLD = 0.12
PRICE_DECIMALS = 8
PROBABILITY_DECIMALS = 4
SCORE_DECIMALS = 6

DepthFetcher = Callable[[str, int], dict[str, Any]]
AggTradesFetcher = Callable[[str, int], list[dict[str, Any]]]
PriceFetcher = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class OrderbookTradeFlowConfig:
    levels_per_side: int = LEVELS_PER_SIDE
    snapshot_interval_sec: float = SNAPSHOT_INTERVAL_SEC
    agg_trades_limit: int = AGG_TRADES_LIMIT
    min_total_quote_flow: float = MIN_TOTAL_QUOTE_FLOW
    book_weight: float = BOOK_WEIGHT
    flow_weight: float = FLOW_WEIGHT
    combined_threshold: float = COMBINED_THRESHOLD
    entry_grace_ms: int = ORDERBOOK_TRADE_FLOW_ENTRY_GRACE_MS


@dataclass(frozen=True)
class OrderbookTradeFlowDependencies:
    fetch_depth: DepthFetcher = fetch_orderbook_depth_levels
    fetch_agg_trades: AggTradesFetcher = fetch_agg_trades_display
    fetch_price: PriceFetcher = fetch_premium_index


@dataclass(frozen=True)
class OrderbookTradeFlowEvaluation:
    imbalance_before: float
    imbalance_after: float
    book_delta: float
    book_component: float
    flow_ratio: float
    buy_quote: float
    sell_quote: float
    combined_score: float
    direction: str
    confidence: float
    passed: bool


DEFAULT_CONFIG = OrderbookTradeFlowConfig()
DEFAULT_DEPENDENCIES = OrderbookTradeFlowDependencies()


def predict_orderbook_trade_flow_direction(
    symbol: str,
    duration: str = RULE_DURATION,
    *,
    entry_open_time: int | None = None,
    now_ms: int | None = None,
    result_strategy_key: str | None = None,
    config: OrderbookTradeFlowConfig = DEFAULT_CONFIG,
    dependencies: OrderbookTradeFlowDependencies = DEFAULT_DEPENDENCIES,
) -> dict[str, Any]:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(
            f"orderbook trade flow strategy supports only {sorted(SUPPORTED_RULE_DURATIONS)}, got {duration}"
        )
    cfg = _validated_config(config)
    sym = symbol.upper()
    depth_a = dependencies.fetch_depth(sym, cfg.levels_per_side)
    time.sleep(cfg.snapshot_interval_sec)
    depth_b = dependencies.fetch_depth(sym, cfg.levels_per_side)

    imb_a = _qty_imbalance(depth_a["bids"], depth_a["asks"], cfg.levels_per_side)
    imb_b = _qty_imbalance(depth_b["bids"], depth_b["asks"], cfg.levels_per_side)
    book_delta = imb_b - imb_a
    book_component = max(-1.0, min(1.0, book_delta / 2.0))

    trades = dependencies.fetch_agg_trades(sym, cfg.agg_trades_limit)
    flow_ratio, buy_q, sell_q = _flow_ratio(trades)
    total_flow = buy_q + sell_q
    flow_ok = total_flow >= cfg.min_total_quote_flow

    w_b = cfg.book_weight
    w_f = cfg.flow_weight
    combined = w_b * book_component + w_f * flow_ratio
    if not flow_ok:
        combined = book_component
        w_b, w_f = 1.0, 0.0

    direction = "up" if combined >= 0 else "down"
    confidence = max(abs(book_component), abs(flow_ratio), abs(combined)) if flow_ok else abs(book_component)
    confidence = min(1.0, confidence)

    passed = flow_ok and abs(combined) >= cfg.combined_threshold
    if not flow_ok:
        passed = abs(book_component) >= cfg.combined_threshold

    evaluation = OrderbookTradeFlowEvaluation(
        imbalance_before=imb_a,
        imbalance_after=imb_b,
        book_delta=book_delta,
        book_component=book_component,
        flow_ratio=flow_ratio,
        buy_quote=buy_q,
        sell_quote=sell_q,
        combined_score=combined,
        direction=direction,
        confidence=confidence,
        passed=passed,
    )

    entry_price = _entry_price(dependencies.fetch_price(sym))
    open_time = int(entry_open_time if entry_open_time is not None else depth_b.get("timestamp", 0))
    if open_time <= 0:
        raise ValueError("entry open time missing for orderbook trade flow prediction")
    entry_window_passed = is_within_entry_grace(
        open_time,
        now_ms,
        grace_ms=int(cfg.entry_grace_ms),
    )

    out_key = result_strategy_key or ORDERBOOK_TRADE_FLOW_STRATEGY_KEY
    invert_direction = out_key == ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY
    strategy_def = strategy_definition(out_key)
    return _prediction_payload(
        sym,
        duration,
        open_time,
        entry_price,
        evaluation,
        cfg,
        entry_window_passed,
        strategy=strategy_def,
        invert_direction=invert_direction,
        book_weight_used=w_b,
        flow_weight_used=w_f,
    )


def _qty_imbalance(bids: Sequence[Sequence[Any]], asks: Sequence[Sequence[Any]], levels: int) -> float:
    bid_sum = sum(float(level[1]) for level in bids[:levels])
    ask_sum = sum(float(level[1]) for level in asks[:levels])
    total = bid_sum + ask_sum
    if total <= 0:
        raise ValueError("order book has no quantity on bids/asks")
    return (bid_sum - ask_sum) / total


def _flow_ratio(trades: list[dict[str, Any]]) -> tuple[float, float, float]:
    buy_q = 0.0
    sell_q = 0.0
    for row in trades:
        qq = float(row.get("quoteQty", 0) or 0)
        side = str(row.get("side", ""))
        if side == "buy":
            buy_q += qq
        elif side == "sell":
            sell_q += qq
    total = buy_q + sell_q
    if total <= 0:
        return 0.0, 0.0, 0.0
    return (buy_q - sell_q) / total, buy_q, sell_q


def _prediction_payload(
    symbol: str,
    duration: str,
    open_time: int,
    entry_price: float,
    evaluation: OrderbookTradeFlowEvaluation,
    config: OrderbookTradeFlowConfig,
    entry_window_passed: bool,
    *,
    strategy: StrategyDefinition,
    invert_direction: bool,
    book_weight_used: float,
    flow_weight_used: float,
) -> dict[str, Any]:
    confidence = round(min(1.0, evaluation.confidence), PROBABILITY_DECIMALS)
    direction_out = evaluation.direction
    if invert_direction:
        direction_out = "down" if direction_out == "up" else "up"
    probability_up = confidence if direction_out == "up" else 1.0 - confidence
    trade_quality_passed = evaluation.passed and entry_window_passed
    rule_gate_name = strategy.rule_names[0] if strategy.rule_names else ORDERBOOK_TRADE_FLOW_RULE_NAME
    return {
        "symbol": symbol,
        "strategy_key": strategy.key,
        "duration": duration,
        "open_time": int(open_time),
        "entry_price": round(float(entry_price), PRICE_DECIMALS),
        "direction": direction_out,
        "probability_up": round(probability_up, PROBABILITY_DECIMALS),
        "confidence": confidence,
        "certainty_label": _certainty_label(trade_quality_passed, invert_direction),
        "threshold": config.combined_threshold,
        "trade_quality_score": round(min(1.0, abs(evaluation.combined_score) / config.combined_threshold), SCORE_DECIMALS),
        "trade_quality_passed": trade_quality_passed,
        "trade_quality_gate": rule_gate_name,
        "high_winrate_gate": None,
        "high_winrate_rule": rule_gate_name,
        "high_winrate_gate_passed": None,
        "high_winrate_gate_value": None,
        "high_winrate_gate_min": None,
        "signal_source": strategy.signal_source,
        "rule_score": round(evaluation.combined_score, SCORE_DECIMALS),
        "rule_reasons": _rule_reasons(
            evaluation,
            config,
            entry_window_passed,
            book_weight_used,
            flow_weight_used,
            invert_direction=invert_direction,
        ),
        "orderbook": _orderbook_payload(
            evaluation,
            config,
            entry_window_passed,
            strategy,
            invert_direction,
            book_weight_used,
            flow_weight_used,
        ),
        "timeframe_votes": [],
    }


def _orderbook_payload(
    evaluation: OrderbookTradeFlowEvaluation,
    config: OrderbookTradeFlowConfig,
    entry_window_passed: bool,
    strategy: StrategyDefinition,
    invert_direction: bool,
    book_weight_used: float,
    flow_weight_used: float,
) -> dict[str, Any]:
    rule_label = strategy.rule_names[0] if strategy.rule_names else ORDERBOOK_TRADE_FLOW_RULE_NAME
    return {
        "strategy": rule_label,
        "levelsPerSide": config.levels_per_side,
        "snapshotIntervalSec": config.snapshot_interval_sec,
        "aggTradesLimit": config.agg_trades_limit,
        "minTotalQuoteFlow": config.min_total_quote_flow,
        "bookWeight": book_weight_used,
        "flowWeight": flow_weight_used,
        "entryWindowPassed": entry_window_passed,
        "imbalanceBefore": round(evaluation.imbalance_before, SCORE_DECIMALS),
        "imbalanceAfter": round(evaluation.imbalance_after, SCORE_DECIMALS),
        "bookDelta": round(evaluation.book_delta, SCORE_DECIMALS),
        "bookComponent": round(evaluation.book_component, SCORE_DECIMALS),
        "flowRatio": round(evaluation.flow_ratio, SCORE_DECIMALS),
        "buyQuote": round(evaluation.buy_quote, SCORE_DECIMALS),
        "sellQuote": round(evaluation.sell_quote, SCORE_DECIMALS),
        "combinedScore": round(evaluation.combined_score, SCORE_DECIMALS),
        "combinedThreshold": config.combined_threshold,
        "invertDirection": invert_direction,
    }


def _rule_reasons(
    evaluation: OrderbookTradeFlowEvaluation,
    config: OrderbookTradeFlowConfig,
    entry_window_passed: bool,
    book_weight_used: float,
    flow_weight_used: float,
    *,
    invert_direction: bool,
) -> list[str]:
    return [
        f"rule={ORDERBOOK_TRADE_FLOW_RULE_NAME}",
        f"invert_direction={invert_direction}",
        f"levels_per_side={config.levels_per_side}",
        f"snapshot_interval_sec={config.snapshot_interval_sec}",
        f"entry_window_passed={entry_window_passed}",
        f"imbalance_before={evaluation.imbalance_before:.6f}",
        f"imbalance_after={evaluation.imbalance_after:.6f}",
        f"book_delta={evaluation.book_delta:.6f}",
        f"flow_buy_quote={evaluation.buy_quote:.2f}",
        f"flow_sell_quote={evaluation.sell_quote:.2f}",
        f"flow_ratio={evaluation.flow_ratio:.6f}",
        f"weights_book_flow={book_weight_used:.2f}/{flow_weight_used:.2f}",
        f"combined={evaluation.combined_score:.6f}",
        f"threshold={config.combined_threshold:.4f}",
    ]


def _entry_price(row: dict[str, Any]) -> float:
    price = float(row.get("indexPrice") or 0)
    if price <= 0:
        raise ValueError("latest index price unavailable")
    return price


def _validated_config(config: OrderbookTradeFlowConfig) -> OrderbookTradeFlowConfig:
    if config.levels_per_side <= 0:
        raise ValueError("levels_per_side must be > 0")
    if config.snapshot_interval_sec < 0:
        raise ValueError("snapshot_interval_sec must be >= 0")
    if config.agg_trades_limit < 1:
        raise ValueError("agg_trades_limit must be >= 1")
    if config.min_total_quote_flow < 0:
        raise ValueError("min_total_quote_flow must be >= 0")
    if config.combined_threshold <= 0:
        raise ValueError("combined_threshold must be > 0")
    if config.entry_grace_ms < 0:
        raise ValueError("entry_grace_ms must be >= 0")
    return config


def _certainty_label(passed: bool, invert_direction: bool = False) -> str:
    if invert_direction:
        return "ORDERBOOK_TRADE_FLOW_INVERT_TRADE" if passed else "ORDERBOOK_TRADE_FLOW_INVERT_WAIT"
    return "ORDERBOOK_TRADE_FLOW_TRADE" if passed else "ORDERBOOK_TRADE_FLOW_WAIT"
