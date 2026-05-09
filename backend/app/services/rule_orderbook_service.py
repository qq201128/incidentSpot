from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from app.db.session import get_conn
from app.services.rule_config import (
    BPS_DIVISOR,
    MAX_SIGNAL_SCORE,
    MAX_SPREAD_BPS,
    MIN_SIGNAL_SCORE,
    ORDERBOOK_IMBALANCE_SCALE,
    ORDERBOOK_IMBALANCE_WEIGHT,
    ORDERBOOK_MICROPRICE_BPS_SCALE,
    ORDERBOOK_MICROPRICE_WEIGHT,
    ORDERBOOK_OFI_SCALE,
    ORDERBOOK_OFI_WEIGHT,
)


_UPSERT_ORDERBOOK_FEATURE_SQL = """
INSERT INTO orderbook_features(
  symbol, open_time, imbalance, spread_bps, bid_qty_sum, ask_qty_sum,
  best_bid, best_ask, best_bid_qty, best_ask_qty, microprice, microprice_bps,
  ofi, ofi_ratio, quote_time
)
VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(symbol, open_time) DO UPDATE SET
  imbalance=excluded.imbalance,
  spread_bps=excluded.spread_bps,
  bid_qty_sum=excluded.bid_qty_sum,
  ask_qty_sum=excluded.ask_qty_sum,
  best_bid=excluded.best_bid,
  best_ask=excluded.best_ask,
  best_bid_qty=excluded.best_bid_qty,
  best_ask_qty=excluded.best_ask_qty,
  microprice=excluded.microprice,
  microprice_bps=excluded.microprice_bps,
  ofi=excluded.ofi,
  ofi_ratio=excluded.ofi_ratio,
  quote_time=excluded.quote_time
"""


@dataclass(frozen=True)
class OrderbookScoreInput:
    orderbook: dict[str, Any]
    best_bid: float
    best_ask: float
    imbalance: float
    spread_bps: float
    ofi_ratio: float | None
    microprice_bps: float


def orderbook_rule_score(orderbook: dict[str, Any]) -> dict[str, Any]:
    best_bid = _positive_float(orderbook.get("best_bid"), "best_bid")
    best_ask = _positive_float(orderbook.get("best_ask"), "best_ask")
    mid = (best_bid + best_ask) / 2.0
    imbalance = _finite_float(orderbook.get("imbalance"), "imbalance")
    spread_bps = (best_ask - best_bid) / mid * BPS_DIVISOR
    ofi_ratio = _optional_float(orderbook.get("ofi_ratio"), "ofi_ratio")
    microprice_bps = _finite_float(orderbook.get("microprice_bps"), "microprice_bps")
    return _score_payload(
        OrderbookScoreInput(
            orderbook=orderbook,
            best_bid=best_bid,
            best_ask=best_ask,
            imbalance=imbalance,
            spread_bps=spread_bps,
            ofi_ratio=ofi_ratio,
            microprice_bps=microprice_bps,
        )
    )


def persist_orderbook_features(symbol: str, open_time: int, orderbook: dict[str, Any]) -> None:
    conn = get_conn()
    try:
        conn.execute(_UPSERT_ORDERBOOK_FEATURE_SQL, _feature_values(symbol, open_time, orderbook))
        conn.commit()
    finally:
        conn.close()


def _score_payload(data: OrderbookScoreInput) -> dict[str, Any]:
    spread_score = 1.0 - min(max(data.spread_bps / MAX_SPREAD_BPS, 0.0), 1.0)
    score = _orderbook_score(data.imbalance, data.ofi_ratio, data.microprice_bps)
    return {
        "score": round(score, 6),
        "imbalance": round(data.imbalance, 6),
        "spreadBps": round(data.spread_bps, 4),
        "spreadScore": round(spread_score, 6),
        "bidQtySum": _finite_float(data.orderbook.get("bid_qty_sum"), "bid_qty_sum"),
        "askQtySum": _finite_float(data.orderbook.get("ask_qty_sum"), "ask_qty_sum"),
        "bestBid": data.best_bid,
        "bestAsk": data.best_ask,
        "bestBidQty": _finite_float(data.orderbook.get("best_bid_qty"), "best_bid_qty"),
        "bestAskQty": _finite_float(data.orderbook.get("best_ask_qty"), "best_ask_qty"),
        "microprice": _finite_float(data.orderbook.get("microprice"), "microprice"),
        "micropriceBps": round(data.microprice_bps, 6),
        "ofi": _optional_float(data.orderbook.get("ofi"), "ofi"),
        "ofiRatio": data.ofi_ratio,
        "quoteTime": _quote_time(data.orderbook),
    }


def _feature_values(symbol: str, open_time: int, orderbook: dict[str, Any]) -> tuple[Any, ...]:
    return (
        symbol.upper(),
        int(open_time),
        orderbook["imbalance"],
        orderbook["spreadBps"],
        orderbook["bidQtySum"],
        orderbook["askQtySum"],
        orderbook["bestBid"],
        orderbook["bestAsk"],
        orderbook["bestBidQty"],
        orderbook["bestAskQty"],
        orderbook["microprice"],
        orderbook["micropriceBps"],
        orderbook["ofi"],
        orderbook["ofiRatio"],
        orderbook["quoteTime"],
    )


def _orderbook_score(imbalance: float, ofi_ratio: float | None, microprice_bps: float) -> float:
    components = [
        _score_component(imbalance, ORDERBOOK_IMBALANCE_SCALE, ORDERBOOK_IMBALANCE_WEIGHT),
        _score_component(microprice_bps, ORDERBOOK_MICROPRICE_BPS_SCALE, ORDERBOOK_MICROPRICE_WEIGHT),
    ]
    if ofi_ratio is not None:
        components.append(_score_component(ofi_ratio, ORDERBOOK_OFI_SCALE, ORDERBOOK_OFI_WEIGHT))
    weighted = sum(score * weight for score, weight in components)
    total_weight = sum(weight for _, weight in components)
    return _clamp(weighted / total_weight, MIN_SIGNAL_SCORE, MAX_SIGNAL_SCORE)


def _score_component(value: float, scale: float, weight: float) -> tuple[float, float]:
    return _clamp(value / scale, MIN_SIGNAL_SCORE, MAX_SIGNAL_SCORE), weight


def _optional_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, label)


def _positive_float(value: Any, label: str) -> float:
    number = _finite_float(value, label)
    if number <= 0:
        raise ValueError(f"orderbook {label} unavailable at {int(time.time() * 1000)}")
    return number


def _finite_float(value: Any, label: str) -> float:
    if value is None:
        raise ValueError(f"orderbook {label} missing")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"orderbook {label} is not finite: {value}")
    return number


def _quote_time(orderbook: dict[str, Any]) -> int:
    value = orderbook.get("timestamp")
    if value is None:
        raise ValueError("orderbook timestamp missing")
    return int(value)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
