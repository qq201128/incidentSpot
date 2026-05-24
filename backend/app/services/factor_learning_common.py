from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

SUCCESS_WIN_RATE_MIN = 0.62
SUCCESS_PROFIT_FACTOR_MIN = 1.05
SUCCESS_SHARPE_MIN = 0.50
SUCCESS_IR_MIN = 0.20
REDUNDANCY_CORR_MIN = 0.50
TOP_FACTOR_LIMIT = 40
TOP_PATTERN_LIMIT = 12
LOSS_PATTERN_LIMIT = 12
MIN_LOSS_ROWS = 4
MIN_WIN_ROWS = 4
MIN_PATTERN_SUPPORT = 4
MIN_LOSS_LIFT = 0.15
MIN_MEDIAN_GAP_SCALE = 0.20
DEFAULT_MIN_CONFIRMATIONS = 2
EPSILON = 1e-12


def edge_score(row: dict[str, Any]) -> float:
    win_edge = max(num(row.get("winRate")) - 0.5, 0.0) * 4.0
    profit_edge = max(num(row.get("profitFactor")) - 1.0, 0.0)
    sharpe_edge = max(num(row.get("sharpe")), 0.0) * 0.05
    ir_edge = abs(num(row.get("ir"))) * 0.10
    return win_edge + profit_edge + sharpe_edge + ir_edge


def formula_tokens(formula: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)
    ignored = {"close", "open", "high", "low", "volume"}
    return sorted({token for token in tokens if token.lower() not in ignored})[:6]


def finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def num(value: Any) -> float:
    number = finite(value)
    return number if number is not None else 0.0


def round_metric(value: float, digits: int) -> float:
    return round(float(value), digits)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
