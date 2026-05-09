from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from app.services.binance_orderbook_depth import fetch_orderbook_depth_levels
from app.services.binance_service import fetch_premium_index
from app.services.kline_timing import is_within_entry_grace
from app.services.rule_config import RULE_DURATION
from app.services.strategy_registry import (
    ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS,
    ORDERBOOK_NOTIONAL_RULE_NAME,
    ORDERBOOK_NOTIONAL_STRATEGY_KEY,
    StrategyDefinition,
    strategy_definition,
)

ORDERBOOK_NOTIONAL_LEVELS_PER_SIDE = 1_000
ORDERBOOK_NOTIONAL_MIN_QTY = 1.0
ORDERBOOK_NOTIONAL_DIFFERENCE_THRESHOLD = 8_000_000.0
NOTIONAL_SCORE_LIMIT = 1.0
PRICE_DECIMALS = 8
PROBABILITY_DECIMALS = 4
SCORE_DECIMALS = 6

DepthFetcher = Callable[[str, int], dict[str, Any]]
PriceFetcher = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class OrderbookNotionalConfig:
    levels_per_side: int = ORDERBOOK_NOTIONAL_LEVELS_PER_SIDE
    min_qty: float = ORDERBOOK_NOTIONAL_MIN_QTY
    difference_threshold: float = ORDERBOOK_NOTIONAL_DIFFERENCE_THRESHOLD
    entry_grace_ms: int = ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS


@dataclass(frozen=True)
class OrderbookNotionalDependencies:
    fetch_depth: DepthFetcher = fetch_orderbook_depth_levels
    fetch_price: PriceFetcher = fetch_premium_index


@dataclass(frozen=True)
class OrderbookNotionalEvaluation:
    bid_notional: float
    ask_notional: float
    difference: float
    direction: str
    dominant_side: str
    confidence: float
    passed: bool


DEFAULT_CONFIG = OrderbookNotionalConfig()
DEFAULT_DEPENDENCIES = OrderbookNotionalDependencies()


def predict_orderbook_notional_direction(
    symbol: str,
    duration: str = RULE_DURATION,
    *,
    entry_open_time: int | None = None,
    now_ms: int | None = None,
    result_strategy_key: str | None = None,
    config: OrderbookNotionalConfig = DEFAULT_CONFIG,
    dependencies: OrderbookNotionalDependencies = DEFAULT_DEPENDENCIES,
) -> dict[str, Any]:
    if duration != RULE_DURATION:
        raise ValueError(f"orderbook notional strategy supports only {RULE_DURATION}, got {duration}")
    cfg = _validated_config(config)
    sym = symbol.upper()
    depth = dependencies.fetch_depth(sym, cfg.levels_per_side)
    evaluation = evaluate_orderbook_notional(depth["bids"], depth["asks"], config=cfg)
    entry_price = _entry_price(dependencies.fetch_price(sym))
    open_time = _open_time(entry_open_time, depth)
    entry_window_passed = is_within_entry_grace(open_time, now_ms, grace_ms=cfg.entry_grace_ms)
    out_key = result_strategy_key or ORDERBOOK_NOTIONAL_STRATEGY_KEY
    strategy = strategy_definition(out_key)
    return _prediction_payload(
        sym,
        duration,
        open_time,
        entry_price,
        evaluation,
        cfg,
        entry_window_passed,
        strategy=strategy,
    )


def evaluate_orderbook_notional(
    bids: Sequence[Sequence[Any]],
    asks: Sequence[Sequence[Any]],
    *,
    config: OrderbookNotionalConfig = DEFAULT_CONFIG,
) -> OrderbookNotionalEvaluation:
    cfg = _validated_config(config)
    bid_notional = _side_notional(bids, cfg)
    ask_notional = _side_notional(asks, cfg)
    total_notional = bid_notional + ask_notional
    if total_notional <= 0:
        raise ValueError("orderbook notional has no levels with quantity above threshold")
    direction = "up" if bid_notional > ask_notional else "down"
    dominant_side = "bid" if direction == "up" else "ask"
    difference = abs(bid_notional - ask_notional)
    confidence = max(bid_notional, ask_notional) / total_notional
    return OrderbookNotionalEvaluation(
        bid_notional=bid_notional,
        ask_notional=ask_notional,
        difference=difference,
        direction=direction,
        dominant_side=dominant_side,
        confidence=confidence,
        passed=difference > cfg.difference_threshold,
    )


def _prediction_payload(
    symbol: str,
    duration: str,
    open_time: int,
    entry_price: float,
    evaluation: OrderbookNotionalEvaluation,
    config: OrderbookNotionalConfig,
    entry_window_passed: bool,
    *,
    strategy: StrategyDefinition,
) -> dict[str, Any]:
    rule_name = strategy.rule_names[0] if strategy.rule_names else ORDERBOOK_NOTIONAL_RULE_NAME
    confidence = round(evaluation.confidence, PROBABILITY_DECIMALS)
    probability_up = confidence if evaluation.direction == "up" else 1.0 - confidence
    trade_quality_passed = evaluation.passed and entry_window_passed
    return {
        "symbol": symbol,
        "strategy_key": strategy.key,
        "duration": duration,
        "open_time": int(open_time),
        "entry_price": round(float(entry_price), PRICE_DECIMALS),
        "direction": evaluation.direction,
        "probability_up": round(probability_up, PROBABILITY_DECIMALS),
        "confidence": confidence,
        "certainty_label": _certainty_label(trade_quality_passed),
        "threshold": config.difference_threshold,
        "trade_quality_score": _threshold_score(evaluation, config),
        "trade_quality_passed": trade_quality_passed,
        "trade_quality_gate": rule_name,
        "high_winrate_gate": None,
        "high_winrate_rule": rule_name,
        "high_winrate_gate_passed": None,
        "high_winrate_gate_value": None,
        "high_winrate_gate_min": None,
        "signal_source": strategy.signal_source,
        "rule_score": round(evaluation.confidence, SCORE_DECIMALS),
        "rule_reasons": _rule_reasons(evaluation, config, entry_window_passed, rule_name=rule_name),
        "orderbook": _orderbook_payload(evaluation, config, entry_window_passed),
        "timeframe_votes": [],
    }


def _side_notional(levels: Sequence[Sequence[Any]], config: OrderbookNotionalConfig) -> float:
    return sum(
        price * qty
        for price, qty in (_price_qty(level) for level in levels[: config.levels_per_side])
        if qty > config.min_qty
    )


def _price_qty(level: Sequence[Any]) -> tuple[float, float]:
    if len(level) < 2:
        raise ValueError(f"invalid orderbook level: {level}")
    price = float(level[0])
    qty = float(level[1])
    if price <= 0 or qty <= 0:
        raise ValueError(f"invalid orderbook level: {level}")
    return price, qty


def _threshold_score(
    evaluation: OrderbookNotionalEvaluation,
    config: OrderbookNotionalConfig,
) -> float:
    score = min(evaluation.difference / config.difference_threshold, NOTIONAL_SCORE_LIMIT)
    return round(score, SCORE_DECIMALS)


def _orderbook_payload(
    evaluation: OrderbookNotionalEvaluation,
    config: OrderbookNotionalConfig,
    entry_window_passed: bool,
) -> dict[str, Any]:
    return {
        "levelsPerSide": config.levels_per_side,
        "minQty": config.min_qty,
        "differenceThreshold": config.difference_threshold,
        "entryGraceMs": config.entry_grace_ms,
        "entryWindowPassed": entry_window_passed,
        "bidNotional": round(evaluation.bid_notional, SCORE_DECIMALS),
        "askNotional": round(evaluation.ask_notional, SCORE_DECIMALS),
        "notionalDifference": round(evaluation.difference, SCORE_DECIMALS),
        "dominantSide": evaluation.dominant_side,
    }


def _rule_reasons(
    evaluation: OrderbookNotionalEvaluation,
    config: OrderbookNotionalConfig,
    entry_window_passed: bool,
    *,
    rule_name: str,
) -> list[str]:
    return [
        f"rule={rule_name}",
        f"levels_per_side={config.levels_per_side}",
        f"entry_grace_ms={config.entry_grace_ms}",
        f"entry_window_passed={entry_window_passed}",
        f"min_qty>{config.min_qty:g}",
        f"bid_notional={evaluation.bid_notional:.2f}",
        f"ask_notional={evaluation.ask_notional:.2f}",
        f"difference={evaluation.difference:.2f}",
        f"threshold={config.difference_threshold:.2f}",
        f"dominant_side={evaluation.dominant_side}",
    ]


def _entry_price(row: dict[str, Any]) -> float:
    price = float(row.get("indexPrice") or 0)
    if price <= 0:
        raise ValueError("latest index price unavailable")
    return price


def _open_time(entry_open_time: int | None, depth: dict[str, Any]) -> int:
    if entry_open_time is not None:
        return int(entry_open_time)
    timestamp = depth.get("timestamp")
    if timestamp is None:
        raise ValueError("orderbook depth timestamp missing")
    return int(timestamp)


def _validated_config(config: OrderbookNotionalConfig) -> OrderbookNotionalConfig:
    if config.levels_per_side <= 0:
        raise ValueError("levels_per_side must be > 0")
    if config.min_qty < 0:
        raise ValueError("min_qty must be >= 0")
    if config.difference_threshold <= 0:
        raise ValueError("difference_threshold must be > 0")
    if config.entry_grace_ms < 0:
        raise ValueError("entry_grace_ms must be >= 0")
    return config


def _certainty_label(passed: bool) -> str:
    return "ORDERBOOK_NOTIONAL_TRADE" if passed else "ORDERBOOK_NOTIONAL_WAIT"
