from __future__ import annotations

from typing import Any

from app.services.auto_trade_types import AutoTradeSettings
from app.services.ensemble_judge_service import refresh_ensemble_judge
from app.services.forward_validation_service import settle_due_predictions
from app.services.kline_prediction_refresh import refresh_prediction_klines
from app.services.kline_timing import MS_PER_MINUTE, current_rule_entry_open_time_for_duration
from app.services.paper_live_candidate_service import refresh_paper_live_candidate_states
from app.services.rule_config import DURATION_TO_MINUTES


def runtime_deps() -> dict[str, Any]:
    return {
        "current_entry": current_rule_entry_open_time_for_duration,
        "refresh_1m": refresh_1m_prediction_input,
        "refresh_duration": refresh_duration_prediction_input,
        "db_side_effects": run_prediction_db_side_effects,
    }


def refresh_1m_prediction_input(symbol: str, entry_open_time: int) -> None:
    refresh_prediction_klines(symbol, "1m", entry_open_time - MS_PER_MINUTE)


def refresh_duration_prediction_input(symbol: str, duration: str, entry_open_time: int) -> None:
    refresh_prediction_klines(symbol, duration, entry_open_time - _duration_ms(duration))


def run_prediction_db_side_effects(settings_list: list[AutoTradeSettings]) -> None:
    for symbol, duration in _unique_symbol_durations(settings_list):
        settle_due_predictions(symbol, duration)
        refresh_paper_live_candidate_states(symbol, duration)
        refresh_ensemble_judge(symbol, duration)


def _unique_symbol_durations(settings_list: list[AutoTradeSettings]) -> list[tuple[str, str]]:
    return sorted({(settings.symbol.upper(), settings.duration) for settings in settings_list})


def _duration_ms(duration: str) -> int:
    return DURATION_TO_MINUTES[duration] * MS_PER_MINUTE
