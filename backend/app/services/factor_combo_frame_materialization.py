from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.factor_mined_candidates import materialize_mined_factor_frame_for_rows
from app.services.factor_mined_library import mined_factor_rows_for_duration


def materialize_factor_combo_frame_for_row(
    frame: pd.DataFrame,
    *,
    symbol: str,
    duration: str,
    row: dict[str, Any],
    source_rows: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows = source_rows if source_rows is not None else mined_factor_rows_for_duration(symbol, duration)
    mined = materialize_mined_factor_frame_for_rows(
        frame,
        symbol=symbol,
        duration=duration,
        target_rows=[row],
        source_rows=rows,
    )
    _raise_materialization_failures(mined.failures)
    return mined.frame


def _raise_materialization_failures(failures: tuple[dict[str, Any], ...]) -> None:
    if not failures:
        return
    details = "; ".join(_failure_detail(failure) for failure in failures)
    raise ValueError(f"factor combo materialization failed: {details}")


def _failure_detail(failure: dict[str, Any]) -> str:
    factor_name = str(failure.get("factorName") or "unknown_factor")
    stage = str(failure.get("stage") or "unknown_stage")
    error = str(failure.get("error") or "unknown error")
    return f"{factor_name} {stage}: {error}"
