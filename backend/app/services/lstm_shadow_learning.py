from __future__ import annotations

from typing import Any

from app.db.session import get_conn
from app.services.factor_combo_simulation_keys import (
    FACTOR_COMBO_TOP_SIMULATION_RANKS,
    factor_combo_simulation_strategy_key,
)
from app.services.lstm_config import lstm_shadow_strategy_key
from app.services.lstm_prediction_service import lstm_model_status
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

RECENT_SAMPLE_SIZE = 20
LOSS_CLUSTER_SIZE = 5


def lstm_shadow_learning_summary(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    key = lstm_shadow_strategy_key(duration)
    rows = _settled_rows(sym, [key], duration=None)
    current_rows = [row for row in rows if row["duration"] == duration]
    comparison = _comparison_rows(sym, duration)
    return {
        **lstm_model_status(sym, duration),
        "strategyKey": key,
        "sampleCount": len(current_rows),
        "winRate": _win_rate(current_rows),
        "recentWinRate": _recent_win_rate(current_rows),
        "avgReturn": _avg_return(current_rows),
        "byDuration": _by_duration(rows),
        "comparison": comparison,
        "lossPatterns": _loss_patterns(current_rows),
    }


def _comparison_rows(symbol: str, duration: str) -> list[dict[str, Any]]:
    strategy_keys = [factor_combo_simulation_strategy_key(rank) for rank in FACTOR_COMBO_TOP_SIMULATION_RANKS]
    strategy_keys.append(lstm_shadow_strategy_key(duration))
    rows = _settled_rows(symbol, strategy_keys, duration=duration)
    return [_strategy_stats(key, [row for row in rows if row["signal_key"] == key]) for key in strategy_keys]


def _settled_rows(symbol: str, strategy_keys: list[str], duration: str | None) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _key in strategy_keys)
    duration_sql = "AND duration = ?" if duration is not None else ""
    params: tuple[Any, ...] = (*strategy_keys, symbol.upper())
    if duration is not None:
        params = (*params, duration)
    conn = get_conn()
    try:
        rows = conn.execute(
            f"""
            SELECT signal_key, strategy_key, symbol, duration, open_time, direction,
                   actual_return, prediction_correct, model_version, feature_window
            FROM predictions
            WHERE signal_key IN ({placeholders}) AND symbol = ?
              AND settled_at IS NOT NULL {duration_sql}
            ORDER BY open_time
            """,
            params,
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _by_duration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for duration in sorted(SUPPORTED_RULE_DURATIONS):
        duration_rows = [row for row in rows if row["duration"] == duration]
        result.append({"duration": duration, **_stats(duration_rows)})
    return result


def _strategy_stats(strategy_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"strategyKey": strategy_key, **_stats(rows)}


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sampleCount": len(rows),
        "winRate": _win_rate(rows),
        "recentWinRate": _recent_win_rate(rows),
        "avgReturn": _avg_return(rows),
    }


def _win_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if bool(row["prediction_correct"])) / len(rows)


def _recent_win_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return _win_rate(rows[-RECENT_SAMPLE_SIZE:])


def _avg_return(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(float(row["actual_return"] or 0.0) for row in rows) / len(rows)


def _loss_patterns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patterns = []
    if _recent_loss_count(rows) >= LOSS_CLUSTER_SIZE:
        patterns.append({"type": "recent_loss_cluster", "support": _recent_loss_count(rows)})
    if rows and (_avg_return(rows) or 0.0) < 0.0:
        patterns.append({"type": "negative_average_return", "support": len(rows)})
    return patterns


def _recent_loss_count(rows: list[dict[str, Any]]) -> int:
    losses = 0
    for row in reversed(rows):
        if bool(row["prediction_correct"]):
            break
        losses += 1
    return losses
