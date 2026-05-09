from __future__ import annotations

from typing import Any

import pandas as pd

from app.db.session import get_conn
from app.services.binance_service import fetch_orderbook, fetch_premium_index
from app.services.optimized_rule_engine import build_optimized_feature_frame, evaluate_optimized_rules
from app.services.blind_reverse_martingale_strategy import predict_blind_reverse_martingale_direction
from app.services.orderbook_notional_strategy import predict_orderbook_notional_direction
from app.services.orderbook_trade_flow_strategy import predict_orderbook_trade_flow_direction
from app.services.rule_observation_service import observation_signal
from app.services.rule_orderbook_service import orderbook_rule_score, persist_orderbook_features
from app.services.rule_config import (
    MAX_SPREAD_BPS,
    MAX_RULE_CONFIDENCE,
    NEUTRAL_PROBABILITY,
    ORDERBOOK_LIMIT,
    QUALITY_ALIGNMENT_WEIGHT,
    QUALITY_CONFIDENCE_WEIGHT,
    QUALITY_SPREAD_WEIGHT,
    RULE_DURATION,
    RULE_GATE_NAME,
    RULE_MIN_QUALITY_SCORE,
    VEGAS_MIN_DIRECTION_SCORE,
    VEGAS_MIN_RESONANCE_SCORE,
    VEGAS_SCORE_SCALE,
)
from app.services.strategy_registry import (
    BLIND_REVERSE_MARTINGALE_STRATEGY_KEY,
    DEFAULT_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_STRATEGY_KEY,
    ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY,
    ORDERBOOK_TRADE_FLOW_STRATEGY_KEY,
    StrategyDefinition,
    strategy_definition,
)

RECENT_1M_LIMIT = 12_000


def predict_rule_direction(
    symbol: str,
    duration: str = RULE_DURATION,
    *,
    entry_open_time: int | None = None,
    strategy_key: str | None = DEFAULT_STRATEGY_KEY,
) -> dict[str, Any]:
    if duration != RULE_DURATION:
        raise ValueError(f"rule engine supports only {RULE_DURATION}, got {duration}")
    symbol = symbol.upper()
    strategy = strategy_definition(strategy_key)
    if strategy.key == ORDERBOOK_NOTIONAL_STRATEGY_KEY:
        return predict_orderbook_notional_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
        )
    if strategy.key == ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY:
        return predict_orderbook_notional_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
            result_strategy_key=strategy.key,
        )
    if strategy.key == ORDERBOOK_TRADE_FLOW_STRATEGY_KEY:
        return predict_orderbook_trade_flow_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
        )
    if strategy.key == ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY:
        return predict_orderbook_trade_flow_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
            result_strategy_key=strategy.key,
        )
    if strategy.key == BLIND_REVERSE_MARTINGALE_STRATEGY_KEY:
        return predict_blind_reverse_martingale_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
        )
    feature_row = _latest_feature_row(symbol, entry_open_time=entry_open_time)
    matched_rule = evaluate_optimized_rules(feature_row, strategy_key=strategy.key)
    orderbook = orderbook_rule_score(fetch_orderbook(symbol, ORDERBOOK_LIMIT))
    premium = fetch_premium_index(symbol)
    open_time = int(entry_open_time if entry_open_time is not None else feature_row["open_time"])
    persist_orderbook_features(symbol, open_time, orderbook)
    return _prediction_payload(
        symbol=symbol,
        duration=duration,
        open_time=open_time,
        entry_price=float(premium["indexPrice"]),
        feature_row=feature_row,
        rule=matched_rule,
        orderbook=orderbook,
        strategy=strategy,
    )


def _prediction_payload(
    *,
    symbol: str,
    duration: str,
    open_time: int,
    entry_price: float,
    feature_row: dict[str, Any],
    rule: dict[str, Any] | None,
    orderbook: dict[str, Any],
    strategy: StrategyDefinition,
) -> dict[str, Any]:
    observation = observation_signal(feature_row, orderbook)
    direction = str((rule or {}).get("direction") or observation["direction"])
    confidence = _confidence(rule, orderbook, observation)
    passed = rule is not None and orderbook["spreadBps"] <= MAX_SPREAD_BPS
    probability_up = confidence if direction == "up" else 1.0 - confidence
    return {
        "symbol": symbol,
        "strategy_key": strategy.key,
        "duration": duration,
        "open_time": int(open_time),
        "entry_price": round(float(entry_price), 8),
        "direction": direction,
        "probability_up": round(probability_up, 4),
        "confidence": round(confidence, 4),
        "certainty_label": _certainty_label(confidence, passed),
        "threshold": RULE_MIN_QUALITY_SCORE,
        "trade_quality_score": _quality_score(
            rule,
            orderbook,
            feature_row=feature_row,
            direction=direction,
        ),
        "trade_quality_passed": passed,
        "trade_quality_gate": RULE_GATE_NAME,
        "high_winrate_gate": RULE_GATE_NAME,
        "high_winrate_rule": (rule or {}).get("name"),
        "high_winrate_gate_passed": passed,
        "high_winrate_gate_value": (rule or {}).get("min_daily_win_rate"),
        "high_winrate_gate_min": 0.70,
        "signal_source": strategy.signal_source,
        "rule_score": round(confidence, 6),
        "rule_reasons": _rule_reasons(
            rule,
            orderbook,
            observation,
            feature_row=feature_row,
            strategy=strategy,
        ),
        "orderbook": orderbook,
        "timeframe_votes": _timeframe_votes(rule, observation),
    }


def _latest_feature_row(symbol: str, *, entry_open_time: int | None = None) -> dict[str, Any]:
    rows = _load_recent_1m_klines(symbol)
    frame = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    features = build_optimized_feature_frame(frame)
    if entry_open_time is not None:
        features = features[features["open_time"] < int(entry_open_time)]
    if features.empty:
        raise ValueError(f"no optimized rule features for {symbol}")
    return features.iloc[-1].to_dict()


def _load_recent_1m_klines(symbol: str) -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT open_time, open, high, low, close, volume
        FROM klines
        WHERE symbol = ? AND interval = '1m'
        ORDER BY open_time DESC
        LIMIT ?
        """,
        (symbol.upper(), RECENT_1M_LIMIT),
    ).fetchall()
    conn.close()
    if not rows:
        raise ValueError(f"no 1m klines found for {symbol}")
    return [dict(row) for row in reversed(rows)]


def _confidence(
    rule: dict[str, Any] | None,
    orderbook: dict[str, Any],
    observation: dict[str, Any],
) -> float:
    if rule is None:
        return float(observation["confidence"])
    penalty = max(0.0, 1.0 - float(orderbook["spreadScore"])) * 0.05
    return _clamp(float(rule["win_rate"]) - penalty, NEUTRAL_PROBABILITY, MAX_RULE_CONFIDENCE)


def _quality_score(
    rule: dict[str, Any] | None,
    orderbook: dict[str, Any],
    *,
    feature_row: dict[str, Any],
    direction: str,
) -> float:
    resonance = _resonance_quality(feature_row, direction)
    if rule is None:
        return round((float(orderbook["spreadScore"]) + resonance) * 0.5, 6)
    rule_quality = float(rule["min_daily_win_rate"])
    quality = (
        rule_quality * QUALITY_CONFIDENCE_WEIGHT
        + float(orderbook["spreadScore"]) * QUALITY_SPREAD_WEIGHT
        + resonance * QUALITY_ALIGNMENT_WEIGHT
    )
    return round(_clamp(quality, 0.0, 1.0), 6)


def _resonance_quality(feature_row: dict[str, Any], direction: str) -> float:
    direction_score = float(feature_row.get("vegas_direction_score") or 0.0)
    resonance_score = float(feature_row.get("vegas_resonance_score") or 0.0) / VEGAS_SCORE_SCALE
    if direction == "up":
        return _clamp(resonance_score * max(direction_score, 0.0), 0.0, 1.0)
    return _clamp(resonance_score * max(-direction_score, 0.0), 0.0, 1.0)


def _rule_reasons(
    rule: dict[str, Any] | None,
    orderbook: dict[str, Any],
    observation: dict[str, Any],
    *,
    feature_row: dict[str, Any],
    strategy: StrategyDefinition,
) -> list[str]:
    if rule is None:
        return [
            f"strategy={strategy.key}",
            "no_optimized_rule_matched",
            f"observation_score={float(observation['score']):.4f}",
            f"spread_bps={orderbook['spreadBps']:.2f}",
        ]
    return [
        f"strategy={strategy.key}",
        f"rule={rule['name']}",
        f"historical_win_rate={float(rule['win_rate']):.3f}",
        f"min_daily_win_rate={float(rule['min_daily_win_rate']):.3f}",
        f"vegas_resonance={float(feature_row.get('vegas_resonance_score') or 0.0):.2f}",
        f"vegas_direction={float(feature_row.get('vegas_direction_score') or 0.0):.3f}",
        f"vegas_min={VEGAS_MIN_RESONANCE_SCORE:.1f}/{VEGAS_MIN_DIRECTION_SCORE:.2f}",
        f"spread_bps={orderbook['spreadBps']:.2f}",
    ]


def _timeframe_votes(rule: dict[str, Any] | None, observation: dict[str, Any]) -> list[dict[str, Any]]:
    if rule is None:
        return list(observation["votes"])
    return [
        {"feature": feature, "operator": operator, "value": value}
        for feature, operator, value in rule["conditions"]
    ]


def _certainty_label(confidence: float, passed: bool) -> str:
    if confidence >= 0.85 and passed:
        return "RULE_STRONG"
    if passed:
        return "RULE_TRADABLE"
    return "RULE_WAIT"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
