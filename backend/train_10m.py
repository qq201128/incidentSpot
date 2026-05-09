from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, roc_auc_score

from app.db.session import get_conn
from app.services.kline_backfill import backfill_1m_history
from app.services.kline_features import build_feature_frame

INTERVAL = "1m"
MIN_ROWS = 800
TARGET_DB_ROWS = 20_000
DURATION_TO_HORIZON = {"10m": 10, "30m": 30, "60m": 60, "1d": 1440}

MODEL_DIR = Path(__file__).resolve().parent / "models"


def _prepare_aligned_frame(
    df: pd.DataFrame,
    horizon: int,
    min_move_bps: float,
    min_rows_required: int,
) -> tuple[pd.DataFrame, list[str]]:
    move = float(min_move_bps) / 10_000.0
    fwd_ret = df["close"].shift(-horizon) / df["close"] - 1.0
    y = pd.Series(np.nan, index=df.index)
    y = y.mask(fwd_ret > move, 1)
    y = y.mask(fwd_ret < -move, 0)

    feat_df, _spec = build_feature_frame(df)
    aligned = feat_df.join(y.rename("y"), how="left")
    aligned = aligned.dropna(subset=["y"]).reset_index(drop=True)

    if len(aligned) < min_rows_required:
        raise ValueError(
            "not enough labeled rows after filtering small moves; "
            "try lowering min_move_bps or increasing history backfill"
        )

    feature_cols = [c for c in aligned.columns if c not in {"open_time", "y"}]
    return aligned, feature_cols


def _lgbm_classifier_for_duration(duration: str) -> LGBMClassifier:
    """
    LightGBM settings by horizon: 10m uses slightly deeper trees, bagging, and
    class_weight=balanced to offset skew after neutral rows are dropped.
    """
    common: dict = {"random_state": 42, "n_jobs": -1, "verbosity": -1}
    if duration == "10m":
        return LGBMClassifier(
            n_estimators=2000,
            max_depth=7,
            num_leaves=48,
            learning_rate=0.032,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.35,
            reg_lambda=0.9,
            min_child_samples=40,
            class_weight="balanced",
            **common,
        )
    return LGBMClassifier(
        n_estimators=1500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        min_child_samples=50,
        **common,
    )


def _early_stopping_rounds_for_duration(duration: str) -> int:
    return 120 if duration == "10m" else 100


def _train_calibrate_eval_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    duration: str = "10m",
) -> dict:
    x_train, y_train = train[feature_cols], train["y"].astype(int)
    x_val, y_val = val[feature_cols], val["y"].astype(int)
    x_test, y_test = test[feature_cols], test["y"].astype(int)

    model = _lgbm_classifier_for_duration(duration)
    est_rounds = _early_stopping_rounds_for_duration(duration)
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=est_rounds, verbose=False)],
    )

    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    split_idx = max(10, int(len(val_prob) * 0.7))
    split_idx = min(split_idx, len(val_prob) - 10) if len(val_prob) > 20 else len(val_prob) // 2
    if split_idx <= 0 or split_idx >= len(val_prob):
        raise ValueError("validation split is too small for probability calibration")

    calib_prob = val_prob[:split_idx]
    calib_y = y_val.iloc[:split_idx]
    sel_prob = val_prob[split_idx:]
    sel_y = y_val.iloc[split_idx:]

    platt = LogisticRegression(solver="lbfgs")
    platt.fit(calib_prob.reshape(-1, 1), calib_y)
    platt_prob_sel = np.clip(platt.predict_proba(sel_prob.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
    platt_brier_sel = float(brier_score_loss(sel_y, platt_prob_sel))

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(calib_prob, calib_y)
    isotonic_prob_sel = np.clip(isotonic.predict(sel_prob), 1e-6, 1 - 1e-6)
    isotonic_brier_sel = float(brier_score_loss(sel_y, isotonic_prob_sel))

    if isotonic_brier_sel <= platt_brier_sel:
        calibrator = {"method": "isotonic", "model": isotonic}
        calibrator_name = "isotonic"
    else:
        calibrator = {"method": "platt", "model": platt}
        calibrator_name = "platt"

    if calibrator["method"] == "platt":
        val_prob_calibrated = np.clip(calibrator["model"].predict_proba(val_prob.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
        test_prob_calibrated = np.clip(
            calibrator["model"].predict_proba(test_prob.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6
        )
    else:
        val_prob_calibrated = np.clip(calibrator["model"].predict(val_prob), 1e-6, 1 - 1e-6)
        test_prob_calibrated = np.clip(calibrator["model"].predict(test_prob), 1e-6, 1 - 1e-6)

    calibrator_brier = float(brier_score_loss(y_test, test_prob_calibrated))
    calibrator_logloss = float(log_loss(y_test, test_prob_calibrated))

    best_th, best_f1 = 0.5, -1.0
    th_start, th_end, th_step = (0.42, 0.68, 0.01) if duration == "10m" else (0.45, 0.66, 0.01)
    for th in np.arange(th_start, th_end + 1e-9, th_step):
        pred = (val_prob_calibrated >= th).astype(int)
        score = f1_score(y_val, pred, zero_division=0)
        if score > best_f1:
            best_th = float(th)
            best_f1 = float(score)

    test_pred = (test_prob_calibrated >= best_th).astype(int)
    # Align with frontend: confidence = max(P(up), P(down)); evaluate hit when predicting side = (P(up)>=0.5).
    bucket_stats = [
        _bucket_directional_hit_rate(test_prob_calibrated, y_test, 0.80, 0.85),
        _bucket_directional_hit_rate(test_prob_calibrated, y_test, 0.85, 0.90),
        _bucket_directional_hit_rate(test_prob_calibrated, y_test, 0.90, None),
    ]

    return {
        "model": model,
        "calibrator": calibrator,
        "calibrator_name": calibrator_name,
        "best_threshold": best_th,
        "val_auc": float(roc_auc_score(y_val, val_prob)),
        "test_auc": float(roc_auc_score(y_test, test_prob)),
        "test_brier_calibrated": calibrator_brier,
        "test_logloss_calibrated": calibrator_logloss,
        "val_f1": best_f1,
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "confidence_buckets_side_test": bucket_stats,
    }


def walk_forward_scores(
    symbol: str,
    duration: str,
    min_move_bps: float,
    train_window_days: int,
    n_folds: int = 4,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict:
    """Time-ordered walk-forward: each fold trains on prefix, validates/tests on later slices."""
    if duration not in DURATION_TO_HORIZON:
        raise ValueError(f"unsupported duration: {duration}")
    horizon = DURATION_TO_HORIZON[duration]

    df = _ensure_training_rows(symbol, horizon)
    df = _slice_recent_window(df, train_window_days)

    min_rows_required = max(MIN_ROWS, horizon + 2000)
    if len(df) < min_rows_required:
        raise ValueError(f"not enough rows for training; need >= {min_rows_required}, current={len(df)}")

    aligned, feature_cols = _prepare_aligned_frame(df, horizon, min_move_bps, min_rows_required)
    n = len(aligned)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    fold_metrics: list[dict] = []
    for k in range(n_folds):
        train_end = int(n * (k + 1) / (n_folds + 1))
        if train_end < min_rows_required:
            continue
        train = aligned.iloc[:train_end]
        rest = aligned.iloc[train_end:]
        n_rest = len(rest)
        if n_rest < max(50, int(0.05 * n)):
            continue
        n_val = max(50, int(n_rest * val_frac))
        n_test = max(50, int(n_rest * test_frac))
        if n_val + n_test >= n_rest:
            continue
        val = rest.iloc[:n_val]
        test = rest.iloc[n_val : n_val + n_test]

        out = _train_calibrate_eval_split(train, val, test, feature_cols, duration=duration)
        fold_metrics.append(
            {
                "fold": k,
                "train_rows": int(len(train)),
                "val_rows": int(len(val)),
                "test_rows": int(len(test)),
                "calibrator": out["calibrator_name"],
                "best_threshold": out["best_threshold"],
                "val_auc": out["val_auc"],
                "test_auc": out["test_auc"],
                "test_brier_calibrated": out["test_brier_calibrated"],
                "test_logloss_calibrated": out["test_logloss_calibrated"],
                "val_f1": out["val_f1"],
                "test_f1": out["test_f1"],
                "confidence_buckets_side_test": out["confidence_buckets_side_test"],
            }
        )

    if not fold_metrics:
        raise ValueError("walk-forward produced no valid folds; need more history or smaller n_folds")

    def mean_metric(key: str) -> float:
        vals = [float(f[key]) for f in fold_metrics if key in f and f[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "symbol": symbol.upper(),
        "duration": duration,
        "train_window_days": int(train_window_days),
        "n_folds_requested": int(n_folds),
        "folds_used": len(fold_metrics),
        "mean_test_auc": mean_metric("test_auc"),
        "mean_test_brier": mean_metric("test_brier_calibrated"),
        "mean_test_logloss": mean_metric("test_logloss_calibrated"),
        "folds": fold_metrics,
    }


def systematic_retrain(
    symbol: str,
    duration: str = "10m",
    min_move_bps: float = 3.0,
    window_days_candidates: list[int] | None = None,
    n_folds: int = 4,
    save: bool = True,
) -> dict:
    """Grid search recent training windows using walk-forward mean test metrics, then fit final model on best window."""
    candidates = window_days_candidates or [14, 21, 30, 45, 60]
    scored: list[dict] = []
    for days in candidates:
        try:
            wf = walk_forward_scores(symbol, duration, min_move_bps, days, n_folds=n_folds)
        except Exception as exc:
            scored.append({"train_window_days": days, "error": str(exc)})
            continue
        wf["train_window_days"] = days
        scored.append(wf)

    usable = [s for s in scored if "mean_test_brier" in s and not np.isnan(s.get("mean_test_brier", float("nan")))]
    if not usable:
        raise ValueError("no valid window candidates; check data length and parameters")

    # Primary: lower calibrated Brier; tie-break: higher mean test AUC
    best = sorted(
        usable,
        key=lambda s: (s["mean_test_brier"], -s.get("mean_test_auc", 0.0)),
    )[0]

    days_best = int(best["train_window_days"])
    final = train_for_symbol(symbol, duration, min_move_bps, train_window_days=days_best)

    report = {
        "selected_train_window_days": days_best,
        "selection_metric": "mean_test_brier_calibrated (walk-forward)",
        "candidates": scored,
        "final_metrics": final,
    }
    if save:
        report_path = MODEL_DIR / f"systematic_retrain_{duration}.json"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def _load_klines_ohlcv(symbol: str) -> pd.DataFrame:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT open_time, open, high, low, close, volume
        FROM klines
        WHERE symbol = ? AND interval = ?
        ORDER BY open_time ASC
        """,
        (symbol.upper(), INTERVAL),
    ).fetchall()
    conn.close()

    if not rows:
        raise ValueError(f"no 1m klines found for {symbol.upper()}")

    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return df


def _ensure_training_rows(symbol: str, horizon: int) -> pd.DataFrame:
    df = _load_klines_ohlcv(symbol)

    target_rows = max(TARGET_DB_ROWS, MIN_ROWS, horizon + 2000)
    if len(df) < target_rows:
        backfill_1m_history(symbol, target_rows=target_rows, chunk=1000)
        df = _load_klines_ohlcv(symbol)
    return df


def _slice_recent_window(df: pd.DataFrame, train_window_days: int) -> pd.DataFrame:
    if train_window_days <= 0:
        return df
    if df.empty:
        return df
    latest_ts = int(df["open_time"].max())
    window_ms = int(train_window_days) * 24 * 60 * 60 * 1000
    cutoff = latest_ts - window_ms
    recent = df[df["open_time"] >= cutoff].reset_index(drop=True)
    return recent if not recent.empty else df


def _bucket_hit_rate(prob: np.ndarray, y_true: pd.Series, low: float, high: float | None = None) -> dict:
    arr = np.asarray(prob, dtype=float)
    if high is None:
        mask = arr >= low
        label = f"{int(low * 100)}_plus"
    else:
        mask = (arr >= low) & (arr < high)
        label = f"{int(low * 100)}_{int(high * 100)}"
    count = int(mask.sum())
    if count == 0:
        return {"bucket": label, "count": 0, "hit_rate": None}
    y_slice = y_true.iloc[np.where(mask)[0]].astype(int).to_numpy()
    hit_rate = float(y_slice.mean())
    return {"bucket": label, "count": count, "hit_rate": hit_rate}


def _bucket_directional_hit_rate(
    prob_up: np.ndarray, y_true: pd.Series, low: float, high: float | None = None
) -> dict:
    """Hit rate for samples whose *side* confidence max(p,1-p) falls in [low, high)."""
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


def train_for_symbol(
    symbol: str,
    duration: str = "10m",
    min_move_bps: float = 3.0,
    train_window_days: int = 45,
) -> dict:
    if duration not in DURATION_TO_HORIZON:
        raise ValueError(f"unsupported duration: {duration}")
    horizon = DURATION_TO_HORIZON[duration]

    df = _ensure_training_rows(symbol, horizon)
    df = _slice_recent_window(df, train_window_days)

    min_rows_required = max(MIN_ROWS, horizon + 2000)
    if len(df) < min_rows_required:
        raise ValueError(f"not enough rows for training; need >= {min_rows_required}, current={len(df)}")

    aligned, feature_cols = _prepare_aligned_frame(df, horizon, min_move_bps, min_rows_required)
    n = len(aligned)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train = aligned.iloc[:n_train]
    val = aligned.iloc[n_train : n_train + n_val]
    test = aligned.iloc[n_train + n_val :]

    out = _train_calibrate_eval_split(train, val, test, feature_cols, duration=duration)
    model = out["model"]
    calibrator = out["calibrator"]
    calibrator_name = out["calibrator_name"]

    model_path = MODEL_DIR / f"model_{duration}.pkl"
    calibrator_path = MODEL_DIR / f"model_{duration}_calibrator.pkl"
    meta_path = MODEL_DIR / f"model_{duration}_meta.json"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(calibrator, calibrator_path)

    metrics = {
        "symbol": symbol.upper(),
        "duration": duration,
        "horizon_minutes": horizon,
        "interval": INTERVAL,
        "min_move_bps": float(min_move_bps),
        "val_auc": out["val_auc"],
        "test_auc": out["test_auc"],
        "best_threshold": out["best_threshold"],
        "calibrator": calibrator_name,
        "test_brier_calibrated": out["test_brier_calibrated"],
        "test_logloss_calibrated": out["test_logloss_calibrated"],
        "val_f1": out["val_f1"],
        "test_f1": out["test_f1"],
        "features": feature_cols,
        "row_count": n,
        "labeled_row_count": int(aligned["y"].notna().sum()),
        "train_window_days": int(train_window_days),
        "confidence_buckets_side_test": out["confidence_buckets_side_test"],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


if __name__ == "__main__":
    # Examples:
    #   python train_10m.py BTCUSDT 10m 5 45
    #   python train_10m.py --systematic BTCUSDT 10m 5
    import sys

    args = sys.argv[1:]
    if args and args[0] == "--systematic":
        args = args[1:]
        sym = args[0] if len(args) > 0 else "BTCUSDT"
        dur = args[1] if len(args) > 1 else "10m"
        bps = float(args[2]) if len(args) > 2 else 3.0
        folds = int(args[3]) if len(args) > 3 else 4
        windows = (
            [int(x) for x in args[4].split(",") if x.strip()]
            if len(args) > 4
            else [14, 21, 30, 45, 60]
        )
        print(systematic_retrain(sym, dur, bps, window_days_candidates=windows, n_folds=folds, save=True))
    else:
        target_symbol = args[0] if len(args) > 0 else "BTCUSDT"
        target_duration = args[1] if len(args) > 1 else "10m"
        target_min_move_bps = float(args[2]) if len(args) > 2 else 3.0
        target_train_window_days = int(args[3]) if len(args) > 3 else 45
        print(train_for_symbol(target_symbol, target_duration, target_min_move_bps, target_train_window_days))
