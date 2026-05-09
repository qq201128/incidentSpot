"""
End-to-end enhanced training pipeline:
  1) Ingest orderbook / funding / multi-timeframe klines alongside 1m klines.
  2) Build enhanced feature matrix.
  3) Train LightGBM with time-series CV.
  4) Save model + calibration + metadata.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, roc_auc_score

from app.db.session import get_conn
from app.services.binance_service import (
    fetch_funding_rate,
    fetch_klines,
    fetch_orderbook,
    fetch_24h_ticker,
)
from app.services.enhanced_features import (
    build_enhanced_feature_frame,
    build_labels,
    DURATION_TO_HORIZON,
)
from app.services.backtest_profiles import (
    BacktestProfileContext,
    build_confidence_profiles,
    build_quality_profiles,
    build_selected_confidence_profile,
)
from app.services.high_winrate_optimizer import (
    optimize_high_winrate as run_high_winrate_optimization,
)
from app.services.model_metrics import TARGET_TRADES_PER_DAY, TARGET_WIN_RATE

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MIN_MOVE_BPS = 3.0
DEFAULT_MIN_TRADE_GAP_MINUTES = 0


def _select_topdown_10m_features(feature_cols: list[str]) -> list[str]:
    """
    Select top-down features for 10m prediction:
    prioritize higher timeframe trend/structure over noisy 1m micro-moves.
    """
    preferred_prefixes = ("tf_1d_", "tf_4h_", "tf_1h_", "tf_15m_", "tf_5m_")
    preferred_exact = {
        "ma_ratio_60", "ma_ratio_120", "ma_ratio_240",
        "ret_60", "ret_120", "ret_240",
        "vol_std_60", "vol_std_120", "vol_std_240",
        "atr_ratio", "ema_cross",
        "rsi_14", "rsi_14_chg_3", "macd_hist", "macd_hist_chg_3",
        "bb_width_20", "bb_z_20", "donchian_pos_20", "donchian_pos_60",
        "vwap_dev_20", "vwap_dev_60", "efficiency_ratio_10", "efficiency_ratio_20",
        "adx_14", "chop_14", "wick_imbalance",
        "imbalance", "spread_bps", "bid_qty_sum", "ask_qty_sum",
        "imbalance_ma_5", "spread_bps_ma_5", "bid_qty_sum_ma_5", "ask_qty_sum_ma_5",
        "funding_rate", "funding_ma_8", "funding_z_20",
    }
    selected = [
        c for c in feature_cols
        if c.startswith(preferred_prefixes) or c in preferred_exact
    ]
    if not selected:
        raise ValueError("no top-down 10m features selected; check feature builder output")
    return selected


def _upsert_orderbook_rows(symbol: str, rows: list[dict]) -> None:
    conn = get_conn()
    for r in rows:
        conn.execute(
            """
            INSERT INTO orderbook_features(symbol, open_time, imbalance, spread_bps, bid_qty_sum, ask_qty_sum)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, open_time) DO UPDATE SET
              imbalance=excluded.imbalance,
              spread_bps=excluded.spread_bps,
              bid_qty_sum=excluded.bid_qty_sum,
              ask_qty_sum=excluded.ask_qty_sum
            """,
            (
                symbol.upper(),
                int(r["open_time"]),
                float(r.get("imbalance", 0.0)),
                float(r.get("spread_bps", 0.0)),
                float(r.get("bid_qty_sum", 0.0)),
                float(r.get("ask_qty_sum", 0.0)),
            ),
        )
    conn.commit()
    conn.close()


def _upsert_funding_rows(symbol: str, rows: list[dict]) -> None:
    conn = get_conn()
    for r in rows:
        conn.execute(
            """
            INSERT INTO funding_features(symbol, open_time, funding_rate)
            VALUES(?, ?, ?)
            ON CONFLICT(symbol, open_time) DO UPDATE SET
              funding_rate=excluded.funding_rate
            """,
            (symbol.upper(), int(r["open_time"]), float(r.get("funding_rate", 0.0))),
        )
    conn.commit()
    conn.close()


def ingest_enhanced_data(
    symbol: str,
    target_klines: int = 20_000,
    intervals: tuple[str, ...] = ("1m", "5m", "15m", "1h"),
) -> None:
    """
    Fetch raw data from Binance and upsert into SQLite.
    For orderbook/funding we align snapshots to the latest 1m open_time windows.
    """
    sym = symbol.upper()
    # 1) Klines for all requested intervals
    for iv in intervals:
        rows = fetch_klines(sym, iv, limit=min(1000, target_klines))
        if rows:
            _upsert_klines(sym, iv, rows)

    # 2) Orderbook snapshot (single current)
    try:
        ob = fetch_orderbook(sym, limit=500)
        # Align to latest 1m close_time for simplicity
        df_1m = pd.read_sql_query(
            "SELECT open_time, close_time FROM klines WHERE symbol=? AND interval='1m' ORDER BY open_time DESC LIMIT 1",
            get_conn(),
            params=(sym,),
        )
        if not df_1m.empty:
            ot = int(df_1m.iloc[0]["open_time"])
            _upsert_orderbook_rows(sym, [{"open_time": ot, **ob}])
    except Exception:
        pass

    # 3) Funding rate (latest)
    try:
        rate = fetch_funding_rate(sym)
        if rate is not None:
            df_1m_ot = pd.read_sql_query(
                "SELECT open_time FROM klines WHERE symbol=? AND interval='1m' ORDER BY open_time DESC LIMIT 1",
                get_conn(),
                params=(sym,),
            )
            if not df_1m_ot.empty:
                _upsert_funding_rows(sym, [{"open_time": int(df_1m_ot.iloc[0]["open_time"]), "funding_rate": rate}])
    except Exception:
        pass


def _upsert_klines(symbol: str, interval: str, rows: list[dict]) -> None:
    conn = get_conn()
    for item in rows:
        conn.execute(
            """
            INSERT INTO klines(symbol, interval, open_time, open, high, low, close, volume, close_time)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, interval, open_time) DO UPDATE SET
              open=excluded.open,
              high=excluded.high,
              low=excluded.low,
              close=excluded.close,
              volume=excluded.volume,
              close_time=excluded.close_time
            """,
            (
                symbol.upper(),
                interval,
                item["openTime"],
                item["open"],
                item["high"],
                item["low"],
                item["close"],
                item["volume"],
                item["closeTime"],
            ),
        )
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# Training helpers (similar to legacy train_10m but with enhanced features)
# --------------------------------------------------------------------------- #

def _prepare_enhanced_frame(
    symbol: str,
    horizon: int,
    min_move_bps: float,
    min_rows_required: int,
    train_window_days: int = 45,
) -> tuple[pd.DataFrame, list[str]]:
    from app.services.enhanced_features import load_klines, load_orderbook_features, load_funding_features

    df_1m = load_klines(symbol, "1m")
    if train_window_days > 0 and not df_1m.empty:
        latest_ts = int(df_1m["open_time"].max())
        cutoff = latest_ts - int(train_window_days) * 24 * 60 * 60 * 1000
        sliced = df_1m[df_1m["open_time"] >= cutoff].reset_index(drop=True)
        if not sliced.empty:
            df_1m = sliced
    ob_df = load_orderbook_features(symbol)
    funding_df = load_funding_features(symbol)

    feat_df, feature_cols = build_enhanced_feature_frame(df_1m, ob_df=ob_df, funding_df=funding_df)
    feature_cols = _select_topdown_10m_features(feature_cols)
    move = float(min_move_bps) / 10_000.0
    fwd_ret = df_1m["close"].shift(-horizon) / df_1m["close"] - 1.0
    y = build_labels(df_1m, horizon=horizon, min_move_bps=min_move_bps)
    label_df = pd.DataFrame(
        {
            "open_time": df_1m["open_time"],
            "y": y,
            "fwd_ret_10m": fwd_ret,
        }
    )
    aligned = feat_df.merge(label_df, on="open_time", how="left")
    aligned = aligned[(aligned["fwd_ret_10m"] > move) | (aligned["fwd_ret_10m"] < -move)]
    aligned = aligned.dropna(subset=["y", "fwd_ret_10m"]).reset_index(drop=True)

    if len(aligned) < min_rows_required:
        raise ValueError(
            f"not enough labeled rows after filtering; got {len(aligned)}, need {min_rows_required}"
        )

    return aligned, feature_cols


def _train_calibrate_eval(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    trade_confidence_threshold: float = 0.85,
    min_trade_gap_minutes: int = 0,
) -> dict:
    x_train, y_train = train[feature_cols], train["y"].astype(int)
    x_val, y_val = val[feature_cols], val["y"].astype(int)
    x_test, y_test = test[feature_cols], test["y"].astype(int)
    test_fwd_ret = test["fwd_ret_10m"].astype(float).to_numpy()

    model = lgb.LGBMClassifier(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        min_child_samples=50,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    # Calibration split
    split_idx = max(10, int(len(val_prob) * 0.7))
    split_idx = min(split_idx, len(val_prob) - 10) if len(val_prob) > 20 else len(val_prob) // 2
    if split_idx <= 0 or split_idx >= len(val_prob):
        raise ValueError("validation split too small for calibration")

    calib_prob = val_prob[:split_idx]
    calib_y = y_val.iloc[:split_idx]
    sel_prob = val_prob[split_idx:]
    sel_y = y_val.iloc[split_idx:]

    platt = LogisticRegression(solver="lbfgs")
    platt.fit(calib_prob.reshape(-1, 1), calib_y)
    platt_prob_sel = np.clip(platt.predict_proba(sel_prob.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
    platt_brier = float(brier_score_loss(sel_y, platt_prob_sel))

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(calib_prob, calib_y)
    iso_prob_sel = np.clip(isotonic.predict(sel_prob), 1e-6, 1 - 1e-6)
    iso_brier = float(brier_score_loss(sel_y, iso_prob_sel))

    if iso_brier <= platt_brier:
        calibrator = {"method": "isotonic", "model": isotonic}
        calibrator_name = "isotonic"
    else:
        calibrator = {"method": "platt", "model": platt}
        calibrator_name = "platt"

    if calibrator["method"] == "platt":
        val_prob_cal = np.clip(calibrator["model"].predict_proba(val_prob.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
        test_prob_cal = np.clip(
            calibrator["model"].predict_proba(test_prob.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6
        )
    else:
        val_prob_cal = np.clip(calibrator["model"].predict(val_prob), 1e-6, 1 - 1e-6)
        test_prob_cal = np.clip(calibrator["model"].predict(test_prob), 1e-6, 1 - 1e-6)

    # Threshold tuning
    best_th, best_f1 = 0.5, -1.0
    for th in np.arange(0.45, 0.66, 0.01):
        pred = (val_prob_cal >= th).astype(int)
        score = f1_score(y_val, pred, zero_division=0)
        if score > best_f1:
            best_th = float(th)
            best_f1 = float(score)

    test_pred = (test_prob_cal >= best_th).astype(int)

    # Confidence buckets
    bucket_stats = [
        _bucket_directional_hit_rate(test_prob_cal, y_test, 0.80, 0.85),
        _bucket_directional_hit_rate(test_prob_cal, y_test, 0.85, 0.90),
        _bucket_directional_hit_rate(test_prob_cal, y_test, 0.90, None),
    ]

    test_side = np.where(test_prob_cal >= best_th, 1.0, -1.0)
    test_conf = np.maximum(test_prob_cal, 1.0 - test_prob_cal)
    profile_context = BacktestProfileContext(
        test_frame=test.reset_index(drop=True),
        test_side=test_side,
        test_confidence=test_conf,
        test_fwd_ret=test_fwd_ret,
        min_trade_gap_minutes=min_trade_gap_minutes,
    )
    confidence_profiles = build_confidence_profiles(profile_context)
    quality_profiles = build_quality_profiles(profile_context, feature_cols, "10m")
    selected_profile = build_selected_confidence_profile(profile_context, float(trade_confidence_threshold))

    return {
        "model": model,
        "calibrator": calibrator,
        "calibrator_name": calibrator_name,
        "best_threshold": best_th,
        "val_auc": float(roc_auc_score(y_val, val_prob)),
        "test_auc": float(roc_auc_score(y_test, test_prob)),
        "test_brier_calibrated": float(brier_score_loss(y_test, test_prob_cal)),
        "test_logloss_calibrated": float(log_loss(y_test, test_prob_cal)),
        "val_f1": best_f1,
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "confidence_buckets_side_test": bucket_stats,
        "backtest": selected_profile,
        "confidence_profiles": confidence_profiles,
        "quality_profiles": quality_profiles,
    }


def _bucket_directional_hit_rate(
    prob_up: np.ndarray, y_true: pd.Series, low: float, high: float | None = None
) -> dict:
    p = np.asarray(prob_up, dtype=float)
    conf = np.maximum(p, 1.0 - p)
    pred_side = (p >= 0.5).astype(int)
    y_arr = y_true.astype(int).to_numpy()
    correct = (pred_side == y_arr).astype(int)

    if high is None:
        mask = conf >= low
        label = f"{int(low * 100)}_plus_side_conf"
    else:
        mask = (conf >= low) & (conf < high)
        label = f"{int(low * 100)}_{int(high * 100)}_side_conf"

    count = int(mask.sum())
    if count == 0:
        return {"bucket": label, "count": 0, "hit_rate": None}
    hit_rate = float(correct[np.where(mask)[0]].mean())
    return {"bucket": label, "count": count, "hit_rate": hit_rate}


def train_enhanced(
    symbol: str,
    duration: str = "10m",
    min_move_bps: float = 3.0,
    train_window_days: int = 45,
    trade_confidence_threshold: float = 0.85,
    min_trade_gap_minutes: int = 0,
) -> dict:
    """
    One-shot train using enhanced features.  70/15/15 static split.
    """
    if duration != "10m":
        raise ValueError("enhanced training now supports only 10m")
    horizon = DURATION_TO_HORIZON["10m"]

    aligned, feature_cols = _prepare_enhanced_frame(
        symbol,
        horizon=horizon,
        min_move_bps=min_move_bps,
        min_rows_required=800,
        train_window_days=train_window_days,
    )
    n = len(aligned)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train = aligned.iloc[:n_train]
    val = aligned.iloc[n_train : n_train + n_val]
    test = aligned.iloc[n_train + n_val :]

    out = _train_calibrate_eval(
        train,
        val,
        test,
        feature_cols,
        trade_confidence_threshold=trade_confidence_threshold,
        min_trade_gap_minutes=min_trade_gap_minutes,
    )
    model = out["model"]
    calibrator = out["calibrator"]

    # Save artifacts
    suffix = f"{duration}_enhanced"
    joblib.dump(model, MODEL_DIR / f"model_{suffix}.pkl")
    joblib.dump(calibrator, MODEL_DIR / f"model_{suffix}_calibrator.pkl")

    meta = {
        "symbol": symbol.upper(),
        "duration": duration,
        "horizon_minutes": horizon,
        "min_move_bps": float(min_move_bps),
        "val_auc": out["val_auc"],
        "test_auc": out["test_auc"],
        "best_threshold": out["best_threshold"],
        "calibrator": out["calibrator_name"],
        "test_brier_calibrated": out["test_brier_calibrated"],
        "test_logloss_calibrated": out["test_logloss_calibrated"],
        "val_f1": out["val_f1"],
        "test_f1": out["test_f1"],
        "features": feature_cols,
        "row_count": n,
        "train_window_days": train_window_days,
        "trade_confidence_threshold": float(trade_confidence_threshold),
        "min_trade_gap_minutes": int(min_trade_gap_minutes),
        "confidence_buckets_side_test": out["confidence_buckets_side_test"],
        "backtest_test_split": out["backtest"],
        "backtest_confidence_profiles": out.get("confidence_profiles", []),
        "backtest_quality_profiles": out.get("quality_profiles", []),
    }
    with open(MODEL_DIR / f"model_{suffix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def backtest_window_grid(
    symbol: str,
    windows: list[int],
    min_move_bps: float = 3.0,
    trade_confidence_threshold: float = 0.85,
    min_trade_gap_minutes: int = 0,
) -> dict:
    results: list[dict] = []
    for days in windows:
        meta = train_enhanced(
            symbol,
            "10m",
            min_move_bps=min_move_bps,
            train_window_days=int(days),
            trade_confidence_threshold=trade_confidence_threshold,
            min_trade_gap_minutes=min_trade_gap_minutes,
        )
        bt = meta.get("backtest_test_split", {})
        results.append(
            {
                "train_window_days": int(days),
                "test_auc": meta.get("test_auc"),
                "test_f1": meta.get("test_f1"),
                "strategy_return": bt.get("strategy_return"),
                "buy_hold_return": bt.get("buy_hold_return"),
                "win_rate": bt.get("win_rate"),
                "avg_trade_return": bt.get("avg_trade_return"),
                "test_trades": bt.get("test_trades"),
                "test_rows": bt.get("test_rows"),
            }
        )
    best = sorted(results, key=lambda x: x["strategy_return"], reverse=True)[0] if results else None
    report = {
        "symbol": symbol.upper(),
        "duration": "10m",
        "trade_confidence_threshold": float(trade_confidence_threshold),
        "min_trade_gap_minutes": int(min_trade_gap_minutes),
        "windows": results,
        "best_by_strategy_return": best,
    }
    with open(MODEL_DIR / "backtest_10m_windows.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def optimize_high_winrate(
    symbol: str,
    windows: list[int],
    *,
    target_win_rate: float = TARGET_WIN_RATE,
    min_trades_per_day: float = TARGET_TRADES_PER_DAY,
    min_move_bps: float = DEFAULT_MIN_MOVE_BPS,
    min_trade_gap_minutes: int = DEFAULT_MIN_TRADE_GAP_MINUTES,
) -> dict:
    def train_window(days: int) -> dict:
        return _train_enhanced_to_dir(
            _optimization_candidate_dir(
                days,
                min_move_bps=min_move_bps,
                min_trade_gap_minutes=min_trade_gap_minutes,
            ),
            symbol=symbol,
            duration="10m",
            min_move_bps=min_move_bps,
            train_window_days=days,
            trade_confidence_threshold=target_win_rate,
            min_trade_gap_minutes=min_trade_gap_minutes,
        )

    return run_high_winrate_optimization(
        train_window,
        MODEL_DIR,
        symbol,
        windows,
        target_win_rate=target_win_rate,
        min_trades_per_day=min_trades_per_day,
        min_move_bps=min_move_bps,
        min_trade_gap_minutes=min_trade_gap_minutes,
    )


def _optimization_candidate_dir(
    train_window_days: int,
    *,
    min_move_bps: float,
    min_trade_gap_minutes: int,
) -> Path:
    bps_label = f"{float(min_move_bps):g}".replace(".", "p")
    return MODEL_DIR / "optimization" / f"10m_{int(train_window_days)}d_{bps_label}bps_gap{int(min_trade_gap_minutes)}"


def _train_enhanced_to_dir(
    output_dir: Path,
    *,
    symbol: str,
    duration: str,
    min_move_bps: float,
    train_window_days: int,
    trade_confidence_threshold: float,
    min_trade_gap_minutes: int,
) -> dict:
    global MODEL_DIR
    original_dir = MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR = output_dir
    try:
        return train_enhanced(
            symbol,
            duration,
            min_move_bps=min_move_bps,
            train_window_days=train_window_days,
            trade_confidence_threshold=trade_confidence_threshold,
            min_trade_gap_minutes=min_trade_gap_minutes,
        )
    finally:
        MODEL_DIR = original_dir


# --------------------------------------------------------------------------- #
# CLI entry-point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if args and args[0] == "--backtest-windows":
        symbol = args[1] if len(args) > 1 else "BTCUSDT"
        min_move_bps = float(args[2]) if len(args) > 2 else 3.0
        windows = [int(x) for x in (args[3] if len(args) > 3 else "30,45,60").split(",") if x.strip()]
        conf_th = float(args[4]) if len(args) > 4 else 0.85
        gap_min = int(args[5]) if len(args) > 5 else 0
        print(
            backtest_window_grid(
                symbol,
                windows,
                min_move_bps=min_move_bps,
                trade_confidence_threshold=conf_th,
                min_trade_gap_minutes=gap_min,
            )
        )
    elif args and args[0] == "--optimize-winrate":
        symbol = args[1] if len(args) > 1 else "BTCUSDT"
        target_wr = float(args[2]) if len(args) > 2 else TARGET_WIN_RATE
        min_tpd = float(args[3]) if len(args) > 3 else TARGET_TRADES_PER_DAY
        windows = [int(x) for x in (args[4] if len(args) > 4 else "30,45,60").split(",") if x.strip()]
        min_move_bps = float(args[5]) if len(args) > 5 else DEFAULT_MIN_MOVE_BPS
        gap_min = int(args[6]) if len(args) > 6 else DEFAULT_MIN_TRADE_GAP_MINUTES
        print(
            optimize_high_winrate(
                symbol=symbol,
                windows=windows,
                target_win_rate=target_wr,
                min_trades_per_day=min_tpd,
                min_move_bps=min_move_bps,
                min_trade_gap_minutes=gap_min,
            )
        )
    else:
        symbol = args[0] if len(args) > 0 else "BTCUSDT"
        duration = args[1] if len(args) > 1 else "10m"
        min_move_bps = float(args[2]) if len(args) > 2 else 3.0
        window_days = int(args[3]) if len(args) > 3 else 45
        conf_th = float(args[4]) if len(args) > 4 else 0.85
        gap_min = int(args[5]) if len(args) > 5 else 0
        # Data is assumed already present in DB (backfill via existing kline_backfill if needed)
        print(
            train_enhanced(
                symbol,
                duration,
                min_move_bps,
                window_days,
                trade_confidence_threshold=conf_th,
                min_trade_gap_minutes=gap_min,
            )
        )
