from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.db.session import get_conn
from app.services.enhanced_timeframes import add_online_timeframe_features
from app.services.external_factor_data import ExternalFeatureFrames, add_external_factor_features
from app.services.kline_features import build_feature_frame as _base_build_features

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DURATION_TO_HORIZON = {"10m": 10, "30m": 30, "60m": 60, "1d": 1440}
ORDERBOOK_COLUMNS = (
    "imbalance",
    "spread_bps",
    "bid_qty_sum",
    "ask_qty_sum",
    "microprice_bps",
    "ofi_ratio",
)

# --------------------------------------------------------------------------- #
# Data load helpers
# --------------------------------------------------------------------------- #
def load_klines(symbol: str, interval: str = "1m") -> pd.DataFrame:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT open_time, open, high, low, close, volume
        FROM klines
        WHERE symbol = ? AND interval = ?
        ORDER BY open_time ASC
        """,
        (symbol.upper(), interval),
    ).fetchall()
    conn.close()
    if not rows:
        raise ValueError(f"no klines for {symbol} {interval}")
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)


def load_orderbook_features(symbol: str) -> pd.DataFrame:
    """Load cached orderbook snapshots that were persisted alongside klines."""
    # Backend stores OB snapshots with kline open_time when fetched proactively.
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT open_time, imbalance, spread_bps, bid_qty_sum, ask_qty_sum, microprice_bps, ofi_ratio
        FROM orderbook_features
        WHERE symbol = ?
        ORDER BY open_time ASC
        """,
        (symbol.upper(),),
    ).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame(columns=["open_time", *ORDERBOOK_COLUMNS])
    df = pd.DataFrame(rows, columns=["open_time", *ORDERBOOK_COLUMNS])
    for col in df.columns:
        if col != "open_time":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_funding_features(symbol: str) -> pd.DataFrame:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT open_time, funding_rate
        FROM funding_features
        WHERE symbol = ?
        ORDER BY open_time ASC
        """,
        (symbol.upper(),),
    ).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame(columns=["open_time", "funding_rate"])
    df = pd.DataFrame(rows, columns=["open_time", "funding_rate"])
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Order-book helpers (single-shot snapshot -> derived metrics)
# --------------------------------------------------------------------------- #

def compute_orderbook_metrics(bids: list[list[str]], asks: list[list[str]], close: float) -> dict[str, float]:
    """Compute imbalance, spreads, wall distances from orderbook snapshot."""
    if not bids or not asks or close <= 0:
        return {"imbalance": 0.0, "spread_bps": 0.0, "bid_qty_sum": 0.0, "ask_qty_sum": 0.0}

    bid_prices = np.array([float(b[0]) for b in bids])
    bid_qtys = np.array([float(b[1]) for b in bids])
    ask_prices = np.array([float(a[0]) for a in asks])
    ask_qtys = np.array([float(a[1]) for a in asks])

    best_bid = bid_prices[0]
    best_ask = ask_prices[0]
    spread_bps = (best_ask - best_bid) / close * 10_000.0

    bid_sum = float(bid_qtys.sum())
    ask_sum = float(ask_qtys.sum())
    total = bid_sum + ask_sum
    imbalance = (bid_sum - ask_sum) / total if total > 0 else 0.0

    return {
        "imbalance": imbalance,
        "spread_bps": spread_bps,
        "bid_qty_sum": bid_sum,
        "ask_qty_sum": ask_sum,
    }


# --------------------------------------------------------------------------- #
# Feature builder
# --------------------------------------------------------------------------- #

def build_enhanced_feature_frame(
    df_1m: pd.DataFrame,
    *,
    ob_df: pd.DataFrame | None = None,
    funding_df: pd.DataFrame | None = None,
    external_frames: ExternalFeatureFrames | None = None,
    min_history: int = 240,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build feature matrix from:
     - 1m OHLCV klines (required)
     - Optional orderbook features (imbalance, spread, bid/ask qty)
     - Optional funding-rate features
     - Multi-timeframe klines (5m, 15m, 1h, 4h, 1d) auto-derived
    """
    d = _prepare_ohlcv_frame(df_1m)

    # ---- 1. Base technical features (same as legacy kline_features) ----
    base_df, base_spec = _base_build_features(d, min_history=min_history)
    # base_df already contains OHLCV + engineered columns, shifted 1 step.

    # ---- 2. Online multi-timeframe features from 1m ----
    base_df = add_online_timeframe_features(base_df, d)
    base_df = _add_stub_vegas_columns(base_df)

    base_df = _add_orderbook_features(base_df, ob_df)
    base_df = _add_funding_features(base_df, funding_df)
    if external_frames is not None:
        base_df = add_external_factor_features(base_df, external_frames)

    feature_cols = [c for c in base_df.columns if c not in {"open_time", "y"}]
    required_cols = [c for c in base_spec.columns if c in base_df.columns]
    out = base_df.dropna(subset=required_cols).reset_index(drop=True)
    if len(out) < min_history:
        raise ValueError(f"insufficient rows after feature engineering: {len(out)} < {min_history}")

    return out, feature_cols


def _add_stub_vegas_columns(base_df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder columns so downstream observation helpers stay compatible."""
    out = base_df.copy()
    out["vegas_resonance_score"] = 0.0
    out["vegas_direction_score"] = 0.0
    out["vegas_bull_score"] = 0.0
    out["vegas_bear_score"] = 0.0
    return out


def _prepare_ohlcv_frame(df_1m: pd.DataFrame) -> pd.DataFrame:
    d = df_1m.copy()
    for col in ("open", "high", "low", "close", "volume"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    if d.empty:
        raise ValueError("no valid 1m OHLCV rows")
    return d


def _add_orderbook_features(base_df: pd.DataFrame, ob_df: pd.DataFrame | None) -> pd.DataFrame:
    if ob_df is not None and not ob_df.empty:
        out = base_df.merge(ob_df, on="open_time", how="left")
    else:
        out = base_df.copy()
        for col in ORDERBOOK_COLUMNS:
            out[col] = 0.0
    for col in ORDERBOOK_COLUMNS:
        out[col] = out[col].fillna(0.0)
        out[f"{col}_ma_5"] = out[col].rolling(5, min_periods=1).mean()
        out[f"{col}_chg_1"] = out[col].pct_change(1).fillna(0.0)
    return out


def _add_funding_features(base_df: pd.DataFrame, funding_df: pd.DataFrame | None) -> pd.DataFrame:
    if funding_df is not None and not funding_df.empty:
        out = base_df.merge(funding_df, on="open_time", how="left")
    else:
        out = base_df.copy()
        out["funding_rate"] = 0.0
    out["funding_rate"] = out["funding_rate"].fillna(0.0)
    out["funding_ma_8"] = out["funding_rate"].rolling(8, min_periods=1).mean()
    fr_std = out["funding_rate"].rolling(20, min_periods=1).std().replace(0, np.nan)
    fr_mean = out["funding_rate"].rolling(20, min_periods=1).mean()
    out["funding_z_20"] = ((out["funding_rate"] - fr_mean) / fr_std).fillna(0.0)
    return out


# --------------------------------------------------------------------------- #
# Label builder
# --------------------------------------------------------------------------- #

def build_labels(df: pd.DataFrame, horizon: int, min_move_bps: float) -> pd.Series:
    """Return a Series of {1, 0, np.nan} aligned to df rows."""
    move = float(min_move_bps) / 10_000.0
    fwd_ret = df["close"].shift(-horizon) / df["close"] - 1.0
    y = pd.Series(np.nan, index=df.index)
    y = y.mask(fwd_ret > move, 1)
    y = y.mask(fwd_ret < -move, 0)
    return y
