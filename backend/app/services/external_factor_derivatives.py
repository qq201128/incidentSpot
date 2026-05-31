from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.external_factor_columns import (
    ONCHAIN_RAW_COLUMNS,
    POSITIONING_RAW_COLUMNS,
    SENTIMENT_RAW_COLUMNS,
)
from app.services.external_factor_sql import (
    UPDATE_ONCHAIN_DERIVED_SQL,
    UPDATE_POSITIONING_DERIVED_SQL,
    UPDATE_SENTIMENT_DERIVED_SQL,
)


def add_positioning_derivatives(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "open_interest" not in out:
        return out
    return positioning_derivative_frame(out)


def add_sentiment_derivatives(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "fear_greed_value" not in out:
        return out
    return sentiment_derivative_frame(out)


def add_onchain_derivatives(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "exchange_netflow" not in out:
        return out
    return onchain_derivative_frame(out)


def refresh_positioning_derivatives(conn, symbol: str) -> None:
    frame = _frame_from_query(
        conn,
        f"SELECT open_time, {', '.join(POSITIONING_RAW_COLUMNS)} "
        "FROM futures_positioning_features WHERE symbol = ? ORDER BY open_time ASC",
        (symbol.upper(),),
        ("open_time", *POSITIONING_RAW_COLUMNS),
    )
    for row in positioning_derivative_frame(frame).itertuples(index=False):
        conn.execute(UPDATE_POSITIONING_DERIVED_SQL, _positioning_values(symbol, row))


def refresh_sentiment_derivatives(conn, source: str) -> None:
    frame = _frame_from_query(
        conn,
        "SELECT open_time, fear_greed_value FROM market_sentiment_features "
        "WHERE source = ? ORDER BY open_time ASC",
        (source,),
        ("open_time", *SENTIMENT_RAW_COLUMNS),
    )
    for row in sentiment_derivative_frame(frame).itertuples(index=False):
        conn.execute(UPDATE_SENTIMENT_DERIVED_SQL, _sentiment_values(source, row))


def refresh_onchain_derivatives(conn, symbol: str) -> None:
    frame = _frame_from_query(
        conn,
        f"SELECT open_time, {', '.join(ONCHAIN_RAW_COLUMNS)} "
        "FROM onchain_features WHERE symbol = ? ORDER BY open_time ASC",
        (symbol.upper(),),
        ("open_time", *ONCHAIN_RAW_COLUMNS),
    )
    for row in onchain_derivative_frame(frame).itertuples(index=False):
        conn.execute(UPDATE_ONCHAIN_DERIVED_SQL, _onchain_values(symbol, row))


def positioning_derivative_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["open_interest_chg_1"] = out["open_interest"].pct_change(1, fill_method=None)
    out["open_interest_value_chg_1"] = out["open_interest_value"].pct_change(1, fill_method=None)
    out["open_interest_z_20"] = _zscore(out["open_interest"], 20)
    out["long_short_ratio_chg_1"] = out["long_short_ratio"].pct_change(1, fill_method=None)
    total_taker = (out["taker_buy_vol"] + out["taker_sell_vol"]).replace(0, np.nan)
    out["taker_buy_share"] = out["taker_buy_vol"] / total_taker
    return out


def sentiment_derivative_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["fear_greed_chg_1"] = out["fear_greed_value"].diff(1)
    out["fear_greed_z_30"] = _zscore(out["fear_greed_value"], 30)
    return out


def onchain_derivative_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "exchange_netflow_z_20" not in out.columns:
        out["exchange_netflow_z_20"] = _zscore(out["exchange_netflow"], 20)
    if "active_addresses_chg_1" not in out.columns and "active_addresses" in out.columns:
        out["active_addresses_chg_1"] = out["active_addresses"].pct_change(1, fill_method=None)
    if "transaction_count_chg_1" not in out.columns and "transaction_count" in out.columns:
        out["transaction_count_chg_1"] = out["transaction_count"].pct_change(1, fill_method=None)
    return out


def float_or_none(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _frame_from_query(conn, sql: str, params: tuple, columns: tuple[str, ...]) -> pd.DataFrame:
    rows = conn.execute(sql, params).fetchall()
    frame = pd.DataFrame(rows, columns=columns)
    for column in columns:
        if column != "open_time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=2).mean()
    std = series.rolling(window, min_periods=2).std().replace(0, np.nan)
    return (series - mean) / std


def _positioning_values(symbol: str, row) -> tuple:
    return (
        float_or_none(row.open_interest_chg_1),
        float_or_none(row.open_interest_value_chg_1),
        float_or_none(row.open_interest_z_20),
        float_or_none(row.long_short_ratio_chg_1),
        float_or_none(row.taker_buy_share),
        symbol.upper(),
        int(row.open_time),
    )


def _sentiment_values(source: str, row) -> tuple:
    return (float_or_none(row.fear_greed_chg_1), float_or_none(row.fear_greed_z_30), source, int(row.open_time))


def _onchain_values(symbol: str, row) -> tuple:
    return (
        float_or_none(row.exchange_netflow_z_20),
        float_or_none(row.active_addresses_chg_1),
        float_or_none(row.transaction_count_chg_1),
        symbol.upper(),
        int(row.open_time),
    )
