from __future__ import annotations

from typing import Any

from app.services.event_pnl_rows import factor_candidate_strategy_keys
from app.services.factor_candidate_signal_keys import is_factor_candidate_signal_strategy
from app.services.simulation_event_demotion import evaluate_simulation_event_demotion


def evaluate_factor_candidate_event_demotion(symbol: str, duration: str) -> dict[str, Any]:
    return evaluate_simulation_event_demotion(
        symbol,
        duration,
        source="factor_candidate",
        list_strategy_keys=factor_candidate_strategy_keys,
        validate_strategy_key=is_factor_candidate_signal_strategy,
    )
