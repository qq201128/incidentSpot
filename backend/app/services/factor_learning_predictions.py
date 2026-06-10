from __future__ import annotations

from typing import Any

from app.db.session import get_conn
from app.services.factor_combo_simulation_keys import (
    BATCH_COMBO_KEY_PREFIX,
    BATCH_HIGH_WINRATE_KEY_PREFIX,
    factor_combo_simulation_strategy_keys,
    high_winrate_factor_combo_simulation_strategy_keys,
)
from app.services.lstm_config import lstm_shadow_strategy_key


def settled_factor_combo_predictions(symbol: str, duration: str) -> list[dict[str, Any]]:
    conn = get_conn()
    try:
        rows = conn.execute(
            _SETTLED_PREDICTIONS_SQL.format(placeholders=_strategy_placeholders()),
            (
                *_fixed_simulation_strategy_keys(duration),
                f"{BATCH_COMBO_KEY_PREFIX}%",
                f"{BATCH_HIGH_WINRATE_KEY_PREFIX}%",
                symbol.upper(),
                duration,
            ),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _fixed_simulation_strategy_keys(duration: str) -> tuple[str, ...]:
    return (
        *factor_combo_simulation_strategy_keys(),
        *high_winrate_factor_combo_simulation_strategy_keys(),
        lstm_shadow_strategy_key(duration),
    )


def _strategy_placeholders() -> str:
    return ",".join("?" for _key in _fixed_simulation_strategy_keys_for_placeholders())


def _fixed_simulation_strategy_keys_for_placeholders() -> tuple[str, ...]:
    return (
        *factor_combo_simulation_strategy_keys(),
        *high_winrate_factor_combo_simulation_strategy_keys(),
        "factor_lstm_shadow_placeholder",
    )


_SETTLED_PREDICTIONS_SQL = """
SELECT open_time, direction, confidence, trade_quality_score,
       actual_return, prediction_correct, high_winrate_rule,
       signal_key, strategy_key
FROM predictions
WHERE (
    signal_key IN ({placeholders})
    OR signal_key LIKE ?
    OR signal_key LIKE ?
)
  AND symbol = ? AND duration = ?
  AND settled_at IS NOT NULL
ORDER BY open_time
"""
