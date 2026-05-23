from __future__ import annotations

from typing import Any


def normalize_agg_trade_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Map Binance aggTrade REST/WS fields to display payload."""
    try:
        price = float(row.get("p", 0) or 0)
        qty = float(row.get("q", 0) or 0)
    except (TypeError, ValueError):
        return None
    if price <= 0 or qty <= 0:
        return None
    buyer_maker = bool(row.get("m", False))
    return {
        "price": price,
        "qty": qty,
        "quoteQty": price * qty,
        "time": int(row.get("T", 0) or 0),
        "side": "sell" if buyer_maker else "buy",
    }
