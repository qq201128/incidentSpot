from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.db.session import get_conn, run_db_write_with_retry
from app.services.external_factor_columns import (
    ONCHAIN_COLUMNS,
    ONCHAIN_RAW_COLUMNS,
    POSITIONING_COLUMNS,
    POSITIONING_RAW_COLUMNS,
    SENTIMENT_COLUMNS,
)
from app.services.external_factor_derivatives import (
    add_onchain_derivatives,
    add_positioning_derivatives,
    add_sentiment_derivatives,
    float_or_none,
    refresh_onchain_derivatives,
    refresh_positioning_derivatives,
    refresh_sentiment_derivatives,
)
from app.services.external_factor_sql import (
    UPSERT_FUNDING_SQL,
    UPSERT_ONCHAIN_SQL,
    UPSERT_POSITIONING_SQL,
    UPSERT_SENTIMENT_SQL,
)


@dataclass(frozen=True)
class ExternalFeatureFrames:
    positioning: pd.DataFrame
    sentiment: pd.DataFrame
    onchain: pd.DataFrame


def load_external_feature_frames(symbol: str) -> ExternalFeatureFrames:
    return ExternalFeatureFrames(
        positioning=load_positioning_features(symbol),
        sentiment=load_sentiment_features(),
        onchain=load_onchain_features(symbol),
    )


def add_external_factor_features(base_df: pd.DataFrame, frames: ExternalFeatureFrames) -> pd.DataFrame:
    out = base_df.copy()
    out = _merge_asof_if_present(out, frames.positioning, POSITIONING_COLUMNS)
    out = _merge_asof_if_present(out, frames.sentiment, SENTIMENT_COLUMNS)
    out = _merge_asof_if_present(out, frames.onchain, ONCHAIN_COLUMNS)
    out = add_positioning_derivatives(out)
    out = add_sentiment_derivatives(out)
    out = add_onchain_derivatives(out)
    return out


def load_positioning_features(symbol: str) -> pd.DataFrame:
    return _load_table(_POSITIONING_SELECT_SQL, (symbol.upper(),), ("open_time", *POSITIONING_COLUMNS))


def load_sentiment_features() -> pd.DataFrame:
    return _load_table(_SENTIMENT_SELECT_SQL, (), ("open_time", *SENTIMENT_COLUMNS))


def load_onchain_features(symbol: str) -> pd.DataFrame:
    return _load_table(_ONCHAIN_SELECT_SQL, (symbol.upper(),), ("open_time", *ONCHAIN_COLUMNS))


def upsert_positioning_rows(symbol: str, rows: list[dict]) -> None:
    if not rows:
        return

    def _upsert() -> None:
        conn = get_conn()
        try:
            for row in rows:
                conn.execute(UPSERT_POSITIONING_SQL, _positioning_values(symbol, row))
            refresh_positioning_derivatives(conn, symbol)
            conn.commit()
        finally:
            conn.close()

    run_db_write_with_retry(_upsert)


def upsert_sentiment_rows(rows: list[dict], source: str = "alternative_me_fng") -> None:
    if not rows:
        return

    def _upsert() -> None:
        conn = get_conn()
        try:
            for row in rows:
                conn.execute(UPSERT_SENTIMENT_SQL, _sentiment_values(source, row))
            refresh_sentiment_derivatives(conn, source)
            conn.commit()
        finally:
            conn.close()

    run_db_write_with_retry(_upsert)


def upsert_onchain_rows(symbol: str, rows: list[dict]) -> None:
    if not rows:
        return

    def _upsert() -> None:
        conn = get_conn()
        try:
            for row in rows:
                conn.execute(UPSERT_ONCHAIN_SQL, _onchain_values(symbol, row))
            refresh_onchain_derivatives(conn, symbol)
            conn.commit()
        finally:
            conn.close()

    run_db_write_with_retry(_upsert)


def upsert_funding_rows(symbol: str, rows: list[dict]) -> None:
    if not rows:
        return

    def _upsert() -> None:
        conn = get_conn()
        try:
            for row in rows:
                values = (symbol.upper(), int(row["open_time"]), float_or_none(row.get("funding_rate")))
                conn.execute(UPSERT_FUNDING_SQL, values)
            conn.commit()
        finally:
            conn.close()

    run_db_write_with_retry(_upsert)


def _load_table(sql: str, params: tuple, columns: tuple[str, ...]) -> pd.DataFrame:
    conn = get_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows, columns=columns)
    for column in columns:
        if column != "open_time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["open_time"]).reset_index(drop=True)


def _merge_asof_if_present(base_df: pd.DataFrame, feature_df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if feature_df.empty:
        return base_df
    left = base_df.copy()
    right = feature_df[["open_time", *columns]].copy()
    left["open_time"] = pd.to_numeric(left["open_time"], errors="raise").astype("int64")
    right["open_time"] = pd.to_numeric(right["open_time"], errors="raise").astype("int64")
    left = left.sort_values("open_time").reset_index(drop=True)
    right = right.sort_values("open_time").reset_index(drop=True)
    return pd.merge_asof(left, right, on="open_time", direction="backward")


def _positioning_values(symbol: str, row: dict) -> tuple:
    values = [symbol.upper(), int(row["open_time"])]
    values.extend(float_or_none(row.get(column)) for column in POSITIONING_COLUMNS)
    return tuple(values)


def _sentiment_values(source: str, row: dict) -> tuple:
    return (
        source,
        int(row["open_time"]),
        float(row["fear_greed_value"]),
        str(row.get("fear_greed_classification", "")),
    )


def _onchain_values(symbol: str, row: dict) -> tuple:
    values = [symbol.upper(), int(row["open_time"])]
    values.extend(float_or_none(row.get(column)) for column in (*ONCHAIN_RAW_COLUMNS, *ONCHAIN_COLUMNS[4:]))
    return tuple(values)


_POSITIONING_SELECT_SQL = """
SELECT open_time, open_interest, open_interest_value, long_short_ratio,
       long_account, short_account, taker_buy_sell_ratio, taker_buy_vol, taker_sell_vol,
       open_interest_chg_1, open_interest_value_chg_1, open_interest_z_20,
       long_short_ratio_chg_1, taker_buy_share
FROM futures_positioning_features
WHERE symbol = ?
ORDER BY open_time ASC
"""

_SENTIMENT_SELECT_SQL = """
SELECT open_time, fear_greed_value, fear_greed_chg_1, fear_greed_z_30
FROM market_sentiment_features
WHERE source = 'alternative_me_fng'
ORDER BY open_time ASC
"""

_ONCHAIN_SELECT_SQL = """
SELECT open_time, exchange_netflow, stablecoin_supply_ratio, active_addresses,
       transaction_count, exchange_netflow_z_20, active_addresses_chg_1,
       transaction_count_chg_1
FROM onchain_features
WHERE symbol = ?
ORDER BY open_time ASC
"""
