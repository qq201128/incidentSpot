from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.db.session import get_conn

POSITIONING_COLUMNS = (
    "open_interest",
    "open_interest_value",
    "long_short_ratio",
    "long_account",
    "short_account",
    "taker_buy_sell_ratio",
    "taker_buy_vol",
    "taker_sell_vol",
)
SENTIMENT_COLUMNS = ("fear_greed_value",)
ONCHAIN_COLUMNS = (
    "exchange_netflow",
    "stablecoin_supply_ratio",
    "active_addresses",
    "transaction_count",
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
    out = _add_positioning_derivatives(out)
    out = _add_sentiment_derivatives(out)
    out = _add_onchain_derivatives(out)
    return out


def load_positioning_features(symbol: str) -> pd.DataFrame:
    return _load_table(
        """
        SELECT open_time, open_interest, open_interest_value, long_short_ratio,
               long_account, short_account, taker_buy_sell_ratio, taker_buy_vol, taker_sell_vol
        FROM futures_positioning_features
        WHERE symbol = ?
        ORDER BY open_time ASC
        """,
        (symbol.upper(),),
        ("open_time", *POSITIONING_COLUMNS),
    )


def load_sentiment_features() -> pd.DataFrame:
    return _load_table(
        """
        SELECT open_time, fear_greed_value
        FROM market_sentiment_features
        WHERE source = 'alternative_me_fng'
        ORDER BY open_time ASC
        """,
        (),
        ("open_time", *SENTIMENT_COLUMNS),
    )


def load_onchain_features(symbol: str) -> pd.DataFrame:
    return _load_table(
        """
        SELECT open_time, exchange_netflow, stablecoin_supply_ratio, active_addresses, transaction_count
        FROM onchain_features
        WHERE symbol = ?
        ORDER BY open_time ASC
        """,
        (symbol.upper(),),
        ("open_time", *ONCHAIN_COLUMNS),
    )


def upsert_positioning_rows(symbol: str, rows: list[dict]) -> None:
    if not rows:
        return
    conn = get_conn()
    try:
        for row in rows:
            conn.execute(_UPSERT_POSITIONING_SQL, _positioning_values(symbol, row))
        conn.commit()
    finally:
        conn.close()


def upsert_sentiment_rows(rows: list[dict], source: str = "alternative_me_fng") -> None:
    if not rows:
        return
    conn = get_conn()
    try:
        for row in rows:
            conn.execute(
                _UPSERT_SENTIMENT_SQL,
                (
                    source,
                    int(row["open_time"]),
                    float(row["fear_greed_value"]),
                    str(row.get("fear_greed_classification", "")),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _load_table(sql: str, params: tuple, columns: tuple[str, ...]) -> pd.DataFrame:
    conn = get_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
    except Exception:
        return pd.DataFrame(columns=columns)
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


def _add_positioning_derivatives(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "open_interest" not in out:
        return out
    out["open_interest_chg_1"] = out["open_interest"].pct_change(1, fill_method=None)
    out["open_interest_value_chg_1"] = out["open_interest_value"].pct_change(1, fill_method=None)
    out["open_interest_z_20"] = _zscore(out["open_interest"], 20)
    out["long_short_ratio_chg_1"] = out["long_short_ratio"].pct_change(1, fill_method=None)
    total_taker = (out["taker_buy_vol"] + out["taker_sell_vol"]).replace(0, np.nan)
    out["taker_buy_share"] = out["taker_buy_vol"] / total_taker
    return out


def _add_sentiment_derivatives(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "fear_greed_value" not in out:
        return out
    out["fear_greed_chg_1"] = out["fear_greed_value"].diff(1)
    out["fear_greed_z_30"] = _zscore(out["fear_greed_value"], 30)
    return out


def _add_onchain_derivatives(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "exchange_netflow" not in out:
        return out
    out["exchange_netflow_z_20"] = _zscore(out["exchange_netflow"], 20)
    out["active_addresses_chg_1"] = out["active_addresses"].pct_change(1)
    out["transaction_count_chg_1"] = out["transaction_count"].pct_change(1)
    return out


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=2).mean()
    std = series.rolling(window, min_periods=2).std().replace(0, np.nan)
    return (series - mean) / std


def _positioning_values(symbol: str, row: dict) -> tuple:
    return (
        symbol.upper(),
        int(row["open_time"]),
        _float_or_none(row.get("open_interest")),
        _float_or_none(row.get("open_interest_value")),
        _float_or_none(row.get("long_short_ratio")),
        _float_or_none(row.get("long_account")),
        _float_or_none(row.get("short_account")),
        _float_or_none(row.get("taker_buy_sell_ratio")),
        _float_or_none(row.get("taker_buy_vol")),
        _float_or_none(row.get("taker_sell_vol")),
    )


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    return float(value)


_UPSERT_POSITIONING_SQL = """
INSERT INTO futures_positioning_features(
  symbol, open_time, open_interest, open_interest_value, long_short_ratio,
  long_account, short_account, taker_buy_sell_ratio, taker_buy_vol, taker_sell_vol
)
VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(symbol, open_time) DO UPDATE SET
  open_interest=COALESCE(excluded.open_interest, futures_positioning_features.open_interest),
  open_interest_value=COALESCE(excluded.open_interest_value, futures_positioning_features.open_interest_value),
  long_short_ratio=COALESCE(excluded.long_short_ratio, futures_positioning_features.long_short_ratio),
  long_account=COALESCE(excluded.long_account, futures_positioning_features.long_account),
  short_account=COALESCE(excluded.short_account, futures_positioning_features.short_account),
  taker_buy_sell_ratio=COALESCE(excluded.taker_buy_sell_ratio, futures_positioning_features.taker_buy_sell_ratio),
  taker_buy_vol=COALESCE(excluded.taker_buy_vol, futures_positioning_features.taker_buy_vol),
  taker_sell_vol=COALESCE(excluded.taker_sell_vol, futures_positioning_features.taker_sell_vol)
"""

_UPSERT_SENTIMENT_SQL = """
INSERT INTO market_sentiment_features(source, open_time, fear_greed_value, fear_greed_classification)
VALUES(?, ?, ?, ?)
ON CONFLICT(source, open_time) DO UPDATE SET
  fear_greed_value=excluded.fear_greed_value,
  fear_greed_classification=excluded.fear_greed_classification
"""
