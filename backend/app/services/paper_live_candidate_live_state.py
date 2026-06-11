from __future__ import annotations

from typing import Any


def live_state_by_strategy(conn: Any, symbol: str, duration: str) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT strategy_key, enabled, live_trading_enabled, qty, updated_at
        FROM auto_trade_strategies
        WHERE symbol = ? AND duration = ?
        """,
        (symbol.strip().upper(), duration),
    ).fetchall()
    return {str(row["strategy_key"]): _state_payload(row) for row in rows}


def _state_payload(row: Any) -> dict[str, Any]:
    return {
        "autoTradeEnabled": bool(row["enabled"]),
        "liveTradingEnabled": bool(row["live_trading_enabled"]),
        "qty": float(row["qty"]),
        "updatedAt": row["updated_at"],
    }
