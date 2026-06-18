from __future__ import annotations

from app.services.rule_config import DURATION_TO_MINUTES

UNKNOWN_DURATION = -1

MINUTES_TO_INTERVAL: dict[int, str] = {minutes: interval for interval, minutes in DURATION_TO_MINUTES.items()}
KNOWN_INTERVAL_SQL = ", ".join(f"'{key}'" for key in DURATION_TO_MINUTES)


def settled_expected_profit_usdt(*, status: str, order_side: str | None, order_qty, order_price, result) -> float | None:
    qty = _finite_float(order_qty)
    price = _finite_float(order_price)
    if qty is None or qty <= 0 or price is None or price < 0:
        return None
    if status != "SETTLED" or result is None or order_side is None:
        return None
    is_correct = (order_side == "BUY" and result == "YES") or (order_side == "SELL" and result == "NO")
    return qty * price if is_correct else -qty


def event_interval_where(duration_minutes: int | None, *, alias: str = "e") -> tuple[str, tuple]:
    """SQL fragment filtering events by settlement duration (minutes). None = no filter."""
    if duration_minutes is None:
        return "", ()
    column = f"{alias}.event_interval"
    if duration_minutes == UNKNOWN_DURATION:
        return f" AND {column} NOT IN ({KNOWN_INTERVAL_SQL})", ()
    interval = MINUTES_TO_INTERVAL.get(duration_minutes)
    if interval is None:
        return " AND 1 = 0", ()
    return f" AND {column} = ?", (interval,)


def _finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number
