from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Any
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, roc_auc_score

from app.services.backtest_profiles import (
    BacktestProfileContext,
    build_confidence_profiles,
    build_quality_profiles,
    build_selected_confidence_profile,
)
from app.services.enhanced_features import build_enhanced_feature_frame, build_labels


def select_topdown_10m_features(feature_cols: list[str]) -> list[str]:
    preferred_prefixes = ("tf_1d_", "tf_4h_", "tf_1h_", "tf_15m_", "tf_5m_")
    preferred_exact = {
        "ma_ratio_60", "ma_ratio_120", "ma_ratio_240", "ret_60", "ret_120", "ret_240",
        "vol_std_60", "vol_std_120", "vol_std_240", "atr_ratio", "ema_cross",
        "rsi_14", "rsi_14_chg_3", "macd_hist", "macd_hist_chg_3",
        "bb_width_20", "bb_z_20", "donchian_pos_20", "donchian_pos_60",
        "vwap_dev_20", "vwap_dev_60", "efficiency_ratio_10", "efficiency_ratio_20",
        "adx_14", "chop_14", "wick_imbalance", "imbalance", "spread_bps",
        "bid_qty_sum", "ask_qty_sum", "imbalance_ma_5", "spread_bps_ma_5",
        "bid_qty_sum_ma_5", "ask_qty_sum_ma_5", "funding_rate", "funding_ma_8", "funding_z_20",
    }
    selected = [column for column in feature_cols if column.startswith(preferred_prefixes) or column in preferred_exact]
    if not selected:
        raise ValueError("no top-down 10m features selected; check feature builder output")
    return selected


def prepare_enhanced_frame(
    symbol: str,
    horizon: int,
    min_move_bps: float,
    min_rows_required: int,
    train_window_days: int = 45,
) -> tuple[pd.DataFrame, list[str]]:
    from app.services.enhanced_features import load_funding_features, load_klines, load_orderbook_features

    df_1m = training_window_klines(load_klines(symbol, "1m"), train_window_days)
    feat_df, feature_cols = build_enhanced_feature_frame(
        df_1m,
        ob_df=load_orderbook_features(symbol),
        funding_df=load_funding_features(symbol),
    )
    feature_cols = select_topdown_10m_features(feature_cols)
    aligned = aligned_training_frame(df_1m, feat_df, horizon, min_move_bps)
    if len(aligned) < min_rows_required:
        raise ValueError(f"not enough labeled rows after filtering; got {len(aligned)}, need {min_rows_required}")
    return aligned, feature_cols


def training_window_klines(df_1m: pd.DataFrame, train_window_days: int) -> pd.DataFrame:
    if train_window_days <= 0 or df_1m.empty:
        return df_1m
    latest_ts = int(df_1m["open_time"].max())
    cutoff = latest_ts - int(train_window_days) * 24 * 60 * 60 * 1000
    sliced = df_1m[df_1m["open_time"] >= cutoff].reset_index(drop=True)
    return sliced if not sliced.empty else df_1m


def aligned_training_frame(df_1m: pd.DataFrame, feat_df: pd.DataFrame, horizon: int, min_move_bps: float) -> pd.DataFrame:
    move = float(min_move_bps) / 10_000.0
    fwd_ret = df_1m["close"].shift(-horizon) / df_1m["close"] - 1.0
    labels = build_labels(df_1m, horizon=horizon, min_move_bps=min_move_bps)
    label_df = pd.DataFrame({"open_time": df_1m["open_time"], "y": labels, "fwd_ret_10m": fwd_ret})
    aligned = feat_df.merge(label_df, on="open_time", how="left")
    aligned = aligned[(aligned["fwd_ret_10m"] > move) | (aligned["fwd_ret_10m"] < -move)]
    return aligned.dropna(subset=["y", "fwd_ret_10m"]).reset_index(drop=True)


def train_calibrate_eval(
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
    model = fitted_lgbm_model(x_train, y_train, x_val, y_val)
    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    calibrator, calibrator_name = select_calibrator(val_prob, y_val)
    val_prob_cal = calibrated_probabilities(calibrator, val_prob)
    test_prob_cal = calibrated_probabilities(calibrator, test_prob)
    best_th, best_f1 = tuned_threshold(val_prob_cal, y_val)
    profiles = backtest_profiles(test, test_prob_cal, best_th, feature_cols, trade_confidence_threshold, min_trade_gap_minutes)
    test_pred = (test_prob_cal >= best_th).astype(int)
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
        "confidence_buckets_side_test": confidence_bucket_stats(test_prob_cal, y_test),
        **profiles,
    }


def fitted_lgbm_model(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> Any:
    model = lgb.LGBMClassifier(
        n_estimators=2000, max_depth=6, learning_rate=0.03, subsample=0.7,
        colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0,
        min_child_samples=50, random_state=42, n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    return model


def select_calibrator(val_prob: np.ndarray, y_val: pd.Series) -> tuple[dict, str]:
    split_idx = max(10, int(len(val_prob) * 0.7))
    split_idx = min(split_idx, len(val_prob) - 10) if len(val_prob) > 20 else len(val_prob) // 2
    if split_idx <= 0 or split_idx >= len(val_prob):
        raise ValueError("validation split too small for calibration")
    calib_prob, calib_y = val_prob[:split_idx], y_val.iloc[:split_idx]
    sel_prob, sel_y = val_prob[split_idx:], y_val.iloc[split_idx:]
    platt = LogisticRegression(solver="lbfgs")
    platt.fit(calib_prob.reshape(-1, 1), calib_y)
    platt_brier = float(brier_score_loss(sel_y, np.clip(platt.predict_proba(sel_prob.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)))
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(calib_prob, calib_y)
    iso_brier = float(brier_score_loss(sel_y, np.clip(isotonic.predict(sel_prob), 1e-6, 1 - 1e-6)))
    return ({"method": "isotonic", "model": isotonic}, "isotonic") if iso_brier <= platt_brier else ({"method": "platt", "model": platt}, "platt")


def calibrated_probabilities(calibrator: dict, probabilities: np.ndarray) -> np.ndarray:
    if calibrator["method"] == "platt":
        values = calibrator["model"].predict_proba(probabilities.reshape(-1, 1))[:, 1]
    else:
        values = calibrator["model"].predict(probabilities)
    return np.clip(values, 1e-6, 1 - 1e-6)


def tuned_threshold(val_prob_cal: np.ndarray, y_val: pd.Series) -> tuple[float, float]:
    best_th, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.45, 0.66, 0.01):
        score = f1_score(y_val, (val_prob_cal >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_th = float(threshold)
            best_f1 = float(score)
    return best_th, best_f1


def backtest_profiles(
    test: pd.DataFrame,
    test_prob_cal: np.ndarray,
    threshold: float,
    feature_cols: list[str],
    trade_confidence_threshold: float,
    min_trade_gap_minutes: int,
) -> dict:
    test_side = np.where(test_prob_cal >= threshold, 1.0, -1.0)
    test_conf = np.maximum(test_prob_cal, 1.0 - test_prob_cal)
    context = BacktestProfileContext(
        test_frame=test.reset_index(drop=True),
        test_side=test_side,
        test_confidence=test_conf,
        test_fwd_ret=test["fwd_ret_10m"].astype(float).to_numpy(),
        min_trade_gap_minutes=min_trade_gap_minutes,
    )
    return {
        "backtest": build_selected_confidence_profile(context, float(trade_confidence_threshold)),
        "confidence_profiles": build_confidence_profiles(context),
        "quality_profiles": build_quality_profiles(context, feature_cols, "10m"),
    }


def confidence_bucket_stats(prob_up: np.ndarray, y_true: pd.Series) -> list[dict]:
    return [
        bucket_directional_hit_rate(prob_up, y_true, 0.80, 0.85),
        bucket_directional_hit_rate(prob_up, y_true, 0.85, 0.90),
        bucket_directional_hit_rate(prob_up, y_true, 0.90, None),
    ]


def bucket_directional_hit_rate(prob_up: np.ndarray, y_true: pd.Series, low: float, high: float | None = None) -> dict:
    p = np.asarray(prob_up, dtype=float)
    conf = np.maximum(p, 1.0 - p)
    pred_side = (p >= 0.5).astype(int)
    correct = (pred_side == y_true.astype(int).to_numpy()).astype(int)
    mask = conf >= low if high is None else (conf >= low) & (conf < high)
    label = f"{int(low * 100)}_plus_side_conf" if high is None else f"{int(low * 100)}_{int(high * 100)}_side_conf"
    count = int(mask.sum())
    return {"bucket": label, "count": count, "hit_rate": None if count == 0 else float(correct[np.where(mask)[0]].mean())}
