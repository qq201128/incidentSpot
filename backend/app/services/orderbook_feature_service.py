from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.db.session import get_conn, run_db_write_with_retry
from app.services.rule_config import BPS_DIVISOR


@dataclass(frozen=True)
class OrderbookSnapshotRequest:
    symbol: str
    bids: list[list[str]]
    asks: list[list[str]]
    cache: dict[str, dict[str, Any]]
    quote_time: int


@dataclass(frozen=True)
class OrderbookTop:
    symbol: str
    quote_time: int
    best_bid: float
    best_bid_qty: float
    best_ask: float
    best_ask_qty: float
    bid_qty_sum: float
    ask_qty_sum: float


def build_orderbook_snapshot(request: OrderbookSnapshotRequest) -> dict[str, Any]:
    symbol = request.symbol.upper()
    if not request.bids or not request.asks:
        raise ValueError(f"orderbook depth missing bids or asks for {symbol}")
    top = _top_levels(symbol, request)
    snapshot = _snapshot(top)
    _attach_ofi(snapshot, _previous_snapshot(symbol, request.cache))
    persist_orderbook_tick(snapshot)
    request.cache[symbol] = snapshot
    return snapshot


def persist_orderbook_tick(snapshot: dict[str, Any]) -> None:
    def _persist() -> None:
        conn = get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO orderbook_ticks(
                  symbol, quote_time, best_bid, best_ask, best_bid_qty, best_ask_qty,
                  bid_qty_sum, ask_qty_sum, imbalance, microprice, microprice_bps, ofi, ofi_ratio
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _tick_values(snapshot),
            )
            conn.commit()
        finally:
            conn.close()

    run_db_write_with_retry(_persist)


def _top_levels(symbol: str, request: OrderbookSnapshotRequest) -> OrderbookTop:
    best_bid, best_bid_qty = _level(request.bids[0])
    best_ask, best_ask_qty = _level(request.asks[0])
    bid_qty_sum = sum(_level_qty(level) for level in request.bids)
    ask_qty_sum = sum(_level_qty(level) for level in request.asks)
    if bid_qty_sum + ask_qty_sum <= 0:
        raise ValueError(f"orderbook depth has no positive quantity for {symbol}")
    return OrderbookTop(
        symbol=symbol,
        quote_time=request.quote_time,
        best_bid=best_bid,
        best_bid_qty=best_bid_qty,
        best_ask=best_ask,
        best_ask_qty=best_ask_qty,
        bid_qty_sum=bid_qty_sum,
        ask_qty_sum=ask_qty_sum,
    )


def _snapshot(top: OrderbookTop) -> dict[str, Any]:
    mid = (top.best_bid + top.best_ask) / 2.0
    total_qty = top.bid_qty_sum + top.ask_qty_sum
    top_qty = top.best_bid_qty + top.best_ask_qty
    microprice = (top.best_ask * top.best_bid_qty + top.best_bid * top.best_ask_qty) / top_qty
    return {
        "symbol": top.symbol,
        "timestamp": top.quote_time,
        "best_bid": top.best_bid,
        "best_ask": top.best_ask,
        "best_bid_qty": top.best_bid_qty,
        "best_ask_qty": top.best_ask_qty,
        "bid_qty_sum": top.bid_qty_sum,
        "ask_qty_sum": top.ask_qty_sum,
        "imbalance": (top.bid_qty_sum - top.ask_qty_sum) / total_qty,
        "microprice": microprice,
        "microprice_bps": (microprice - mid) / mid * BPS_DIVISOR,
    }


def _attach_ofi(snapshot: dict[str, Any], previous: dict[str, Any] | None) -> None:
    if previous is None:
        snapshot["ofi"] = None
        snapshot["ofi_ratio"] = None
        return
    ofi = _bid_ofi(snapshot, previous) + _ask_ofi(snapshot, previous)
    denominator = float(snapshot["best_bid_qty"]) + float(snapshot["best_ask_qty"])
    snapshot["ofi"] = ofi
    snapshot["ofi_ratio"] = ofi / denominator if denominator > 0 else None


def _bid_ofi(current: dict[str, Any], previous: dict[str, Any]) -> float:
    if current["best_bid"] > previous["best_bid"]:
        return float(current["best_bid_qty"])
    if current["best_bid"] == previous["best_bid"]:
        return float(current["best_bid_qty"]) - float(previous["best_bid_qty"])
    return -float(previous["best_bid_qty"])


def _ask_ofi(current: dict[str, Any], previous: dict[str, Any]) -> float:
    if current["best_ask"] < previous["best_ask"]:
        return -float(current["best_ask_qty"])
    if current["best_ask"] == previous["best_ask"]:
        return float(previous["best_ask_qty"]) - float(current["best_ask_qty"])
    return float(previous["best_ask_qty"])


def _previous_snapshot(symbol: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    cached = cache.get(symbol.upper())
    if cached is not None:
        return cached
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM orderbook_ticks WHERE symbol = ? ORDER BY quote_time DESC LIMIT 1",
            (symbol.upper(),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _tick_values(snapshot: dict[str, Any]) -> tuple:
    return (
        snapshot["symbol"], snapshot["timestamp"], snapshot["best_bid"], snapshot["best_ask"],
        snapshot["best_bid_qty"], snapshot["best_ask_qty"], snapshot["bid_qty_sum"],
        snapshot["ask_qty_sum"], snapshot["imbalance"], snapshot["microprice"],
        snapshot["microprice_bps"], snapshot.get("ofi"), snapshot.get("ofi_ratio"),
    )


def _level(level: list[str]) -> tuple[float, float]:
    price = float(level[0])
    qty = float(level[1])
    if price <= 0 or qty <= 0:
        raise ValueError(f"invalid orderbook level: {level}")
    return price, qty


def _level_qty(level: list[str]) -> float:
    return _level(level)[1]
