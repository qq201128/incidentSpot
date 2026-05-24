from __future__ import annotations

from typing import Any

from app.services.event_pnl_rows import batch_combo_strategy_keys
from app.services.factor_combo_simulation_keys import is_batch_combo_simulation_strategy
from app.services.simulation_event_demotion import evaluate_simulation_event_demotion


def evaluate_batch_combo_event_demotion(symbol: str, duration: str) -> dict[str, Any]:
    return evaluate_simulation_event_demotion(
        symbol,
        duration,
        source="batch_combo",
        list_strategy_keys=batch_combo_strategy_keys,
        validate_strategy_key=is_batch_combo_simulation_strategy,
    )
