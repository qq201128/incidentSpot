from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.agent_factor_formula import materialize_agent_formula
from app.services.factor_catalog import (
    agent_factor_row_for_backtest,
    is_agent_factor_definition,
    is_mined_factor_definition,
    mined_factor_row_for_backtest,
)
from app.services.factor_registry import FactorDefinition


def materialized_frame_for_factor(
    frame: pd.DataFrame,
    factor_def: FactorDefinition,
    symbol: str,
    duration: str,
) -> pd.DataFrame:
    if is_agent_factor_definition(factor_def):
        return _materialized_agent_frame(frame, factor_def, symbol, duration)
    if is_mined_factor_definition(factor_def):
        return _materialized_mined_frame(frame, factor_def, symbol, duration)
    return frame


def _materialized_agent_frame(
    frame: pd.DataFrame,
    factor_def: FactorDefinition,
    symbol: str,
    duration: str,
) -> pd.DataFrame:
    row = agent_factor_row_for_backtest(factor_def.name, symbol, duration)
    if row is None:
        raise ValueError(f"unknown agent factor for {symbol.upper()} {duration}: {factor_def.name}")
    if factor_def.name in frame.columns:
        return frame
    series = materialize_agent_formula(frame, str(row["formula"]))
    return pd.concat([frame, pd.DataFrame({factor_def.name: series}, index=frame.index)], axis=1, copy=False)


def _materialized_mined_frame(
    frame: pd.DataFrame,
    factor_def: FactorDefinition,
    symbol: str,
    duration: str,
) -> pd.DataFrame:
    row = mined_factor_row_for_backtest(factor_def.name, symbol, duration)
    if row is None:
        raise ValueError(f"unknown mined factor for {symbol.upper()} {duration}: {factor_def.name}")
    return _materialize_mined_target_frame(frame, row, symbol, duration, factor_def.name)


def _materialize_mined_target_frame(
    frame: pd.DataFrame,
    row: dict[str, Any],
    symbol: str,
    duration: str,
    factor_name: str,
) -> pd.DataFrame:
    from app.services.factor_mined_candidates import materialize_mined_factor_frame_for_targets

    materialized = materialize_mined_factor_frame_for_targets(
        frame,
        symbol=symbol.upper(),
        duration=duration,
        target_rows=[row],
    )
    if materialized.failures:
        raise ValueError(f"failed to materialize mined factor {factor_name}: {materialized.failures}")
    return materialized.frame
