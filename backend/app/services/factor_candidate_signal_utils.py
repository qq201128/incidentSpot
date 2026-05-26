from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any

import pandas as pd

from app.services.factor_registry import FactorDirection

NEUTRAL_WIN_RATE = 0.5


def usable_factor_row(row: Any) -> bool:
    if not isinstance(row, dict) or not row.get("factorName"):
        return False
    if row.get("backtestValid") is False:
        return False
    return finite_float(row.get("winRate")) is not None


def candidate_failure(row: dict[str, Any], exc: Exception) -> dict[str, str]:
    return {
        "factorName": str(row.get("factorName") or "unknown"),
        "errorType": type(exc).__name__,
        "error": str(exc),
    }


def candidate_failure_message(symbol: str, duration: str, failures: list[dict[str, str]]) -> str:
    names = ", ".join(item["factorName"] for item in failures[:5])
    return f"all factor candidate signals failed for {symbol.strip().upper()} {duration}: {names}"


def factor_orientation(row: dict[str, Any]) -> int:
    direction = str(row.get("direction") or FactorDirection.NEUTRAL)
    if direction == FactorDirection.LOWER_BETTER.value:
        return -1
    if direction == FactorDirection.HIGHER_BETTER.value:
        return 1
    win_rate = finite_float(row.get("winRate"))
    if win_rate is not None and win_rate < NEUTRAL_WIN_RATE:
        return -1
    return 1 if metric_sign(row) >= 0 else -1


def directional_win_rate(row: dict[str, Any], orientation: int) -> float:
    win_rate = finite_float(row.get("winRate"))
    if win_rate is None:
        raise ValueError(f"factor candidate row missing winRate: {row.get('factorName')}")
    value = win_rate if orientation == 1 else 1.0 - win_rate
    return max(0.0, min(value, 0.99))


def metric_sign(row: dict[str, Any]) -> float:
    for key in ("icMean", "longShortReturn", "ir"):
        value = finite_float(row.get(key))
        if value is not None and value != 0:
            return value
    return 1.0


def series_value_at(series: pd.Series, index: Any) -> Any:
    value = series.loc[index]
    return value.iloc[-1] if isinstance(value, pd.Series) else value


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
