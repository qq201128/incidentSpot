from __future__ import annotations

from typing import Any

from app.services.blind_reverse_martingale_strategy import (
    blind_rm_order_qty_usdt,
    load_blind_rm_settlement_state,
)
from app.services.strategy_registry import BLIND_REVERSE_MARTINGALE_STRATEGY_KEY

_QUICK_TRADE_MARTINGALE_QTY_KEYS: frozenset[str] = frozenset(
    {BLIND_REVERSE_MARTINGALE_STRATEGY_KEY}
)


def adjust_quick_trade_qty_for_martingale(payload: Any, strategy_key: str, symbol: str) -> Any:
    if strategy_key not in _QUICK_TRADE_MARTINGALE_QTY_KEYS:
        return payload
    base = float(payload.order.qty)
    state = load_blind_rm_settlement_state(strategy_key, symbol.upper())
    qty = blind_rm_order_qty_usdt(base, state)
    if abs(qty - base) < 1e-9:
        return payload
    order = payload.order
    new_order = _copy_model(order, {"qty": qty})
    return _copy_model(payload, {"order": new_order})


def _copy_model(model: Any, update: dict[str, Any]) -> Any:
    if hasattr(model, "model_copy"):
        return model.model_copy(update=update)
    return model.copy(update=update)
