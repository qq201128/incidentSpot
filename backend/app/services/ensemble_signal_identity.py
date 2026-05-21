from __future__ import annotations

from typing import Any

from app.services.ensemble_judge_constants import (
    SIGNAL_FACTOR_CANDIDATE,
    SIGNAL_FACTOR_COMBO,
    SIGNAL_HIGH_WINRATE_COMBO,
    SIGNAL_MODEL_FAMILY,
    SIGNAL_OTHER,
)
from app.services.factor_candidate_signal_keys import is_factor_candidate_signal_key
from app.services.factor_combo_display import short_factor_label
from app.services.factor_combo_simulation_keys import (
    BATCH_COMBO_KEY_PREFIX,
    BATCH_HIGH_WINRATE_KEY_PREFIX,
    factor_combo_simulation_strategy_keys,
    high_winrate_factor_combo_simulation_strategy_keys,
)
from app.services.model_family_config import is_model_family_shadow_strategy

TECHNICAL_LABEL_PREFIXES = (
    "rsi_",
    "macd",
    "ema_",
    "sma_",
    "stochastic_",
    "williams_",
    "cci_",
    "aroon_",
    "dmi_",
    "adx_",
    "atr_",
    "ppo_",
    "tsi_",
    "ultimate_",
    "vol_",
    "volume_",
    "ret_",
    "log_ret_",
)


def signal_type(strategy_key: str, rows: list[dict[str, Any]] | None = None) -> str:
    if strategy_key in high_winrate_factor_combo_simulation_strategy_keys():
        return SIGNAL_HIGH_WINRATE_COMBO
    if strategy_key.startswith(BATCH_HIGH_WINRATE_KEY_PREFIX):
        return SIGNAL_HIGH_WINRATE_COMBO
    if strategy_key in factor_combo_simulation_strategy_keys():
        return SIGNAL_FACTOR_COMBO
    if strategy_key.startswith(BATCH_COMBO_KEY_PREFIX):
        return SIGNAL_FACTOR_COMBO
    if is_factor_candidate_signal_key(strategy_key):
        return _candidate_signal_type(rows)
    if is_model_family_shadow_strategy(strategy_key):
        return SIGNAL_MODEL_FAMILY
    return SIGNAL_OTHER


def signal_label(strategy_key: str, rows: list[dict[str, Any]] | None = None) -> str:
    if is_factor_candidate_signal_key(strategy_key):
        return _candidate_signal_label(rows)
    if rows:
        label = _row_label(rows[-1])
        if label:
            return label
    return short_factor_label(strategy_key)


def _candidate_signal_type(rows: list[dict[str, Any]] | None) -> str:
    label = _row_label(rows[-1] if rows else None).lower()
    if label.startswith(TECHNICAL_LABEL_PREFIXES):
        return "indicator"
    return SIGNAL_FACTOR_CANDIDATE


def _candidate_signal_label(rows: list[dict[str, Any]] | None) -> str:
    return _row_label(rows[-1] if rows else None) or "因子候选信号"


def _row_label(row: dict[str, Any] | None) -> str:
    return str((row or {}).get("high_winrate_rule") or (row or {}).get("model_version") or "")
