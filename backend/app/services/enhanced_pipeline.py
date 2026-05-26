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
import pandas as pd

from app.services.enhanced_features import (
    DURATION_TO_HORIZON,
)
from app.services.enhanced_pipeline_ingest import (
    ingest_enhanced_data,
    upsert_funding_rows as _upsert_funding_rows,
    upsert_klines as _upsert_klines,
    upsert_orderbook_rows as _upsert_orderbook_rows,
)
from app.services.enhanced_pipeline_training import (
    bucket_directional_hit_rate as _bucket_directional_hit_rate,
    prepare_enhanced_frame as _prepare_enhanced_frame,
    select_topdown_10m_features as _select_topdown_10m_features,
    train_calibrate_eval as _train_calibrate_eval,
)
from app.services.high_winrate_optimizer import (
    optimize_high_winrate as run_high_winrate_optimization,
)
from app.services.model_metrics import TARGET_TRADES_PER_DAY, TARGET_WIN_RATE

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MIN_MOVE_BPS = 3.0
DEFAULT_MIN_TRADE_GAP_MINUTES = 0


# --------------------------------------------------------------------------- #
# Training helpers (similar to legacy train_10m but with enhanced features)
# --------------------------------------------------------------------------- #
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
