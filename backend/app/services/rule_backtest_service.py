from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.db.session import get_conn
from app.services.enhanced_features import load_orderbook_features
from app.services.kline_timing import is_rule_entry_boundary
from app.services.live_order_settings import FIXED_PAYOUT_RATIO
from app.services.optimized_rule_engine import build_optimized_feature_frame, evaluate_optimized_rules
from app.services.rule_backtest_metrics import (
    daily_stats,
    passed,
    summary,
    walk_forward_validation,
)
from app.services.rule_config import (
    MS_PER_MINUTE,
    RULE_DURATION,
    RULE_HORIZON_MINUTES,
    RULE_TARGET_WIN_RATE,
)
from app.services.strategy_registry import DEFAULT_STRATEGY_KEY, StrategyDefinition, strategy_definition

RULE_ARTIFACT_DIR = Path(__file__).resolve().parent.parent.parent / "rules"
MINUTES_PER_DAY = 24 * 60
LAST_EVALUABLE_MINUTE = MINUTES_PER_DAY - RULE_HORIZON_MINUTES - 1


@dataclass(frozen=True)
class BacktestReportInput:
    symbol: str
    duration: str
    strategy: StrategyDefinition
    feature_frame: pd.DataFrame
    trades: list[dict[str, Any]]
    walk_forward: dict[str, Any]


def run_rule_backtest(
    symbol: str,
    duration: str = RULE_DURATION,
    save: bool = True,
    *,
    strategy_key: str | None = DEFAULT_STRATEGY_KEY,
) -> dict[str, Any]:
    if duration != RULE_DURATION:
        raise ValueError(f"rule backtest supports only {RULE_DURATION}, got {duration}")
    strategy = strategy_definition(strategy_key)
    feature_frame = _labeled_feature_frame(symbol)
    trades = _simulate_trades(feature_frame, strategy.key)
    walk_forward = walk_forward_validation(feature_frame, trades)
    report = _backtest_report(
        BacktestReportInput(
            symbol=symbol.upper(),
            duration=duration,
            strategy=strategy,
            feature_frame=feature_frame,
            trades=trades,
            walk_forward=walk_forward,
        )
    )
    if save:
        _write_report(duration, strategy.key, report)
    return report


def _labeled_feature_frame(symbol: str) -> pd.DataFrame:
    klines = _load_1m_klines(symbol)
    orderbook = load_orderbook_features(symbol)
    features = build_optimized_feature_frame(klines, ob_df=orderbook)
    labeled = _entry_labeled_frame(features, klines)
    return _drop_incomplete_latest_day(labeled)


def _entry_labeled_frame(features: pd.DataFrame, klines: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    frame["entry_open_time"] = frame["open_time"] + MS_PER_MINUTE
    frame = frame[frame["entry_open_time"].map(is_rule_entry_boundary)].copy()
    frame["exit_open_time"] = frame["entry_open_time"] + RULE_HORIZON_MINUTES * MS_PER_MINUTE
    prices = klines.set_index("open_time")["open"]
    frame["entry_price"] = frame["entry_open_time"].map(prices)
    frame["exit_price"] = frame["exit_open_time"].map(prices)
    frame = frame.dropna(subset=["entry_price", "exit_price"]).reset_index(drop=True)
    frame = _with_entry_time_columns(frame)
    frame["fwd_ret"] = frame["exit_price"] / frame["entry_price"] - 1.0
    return frame


def _with_entry_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["open_time"] = out["entry_open_time"].astype("int64")
    timestamps = pd.to_datetime(out["open_time"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    out["tod_bucket"] = ((timestamps.dt.hour * 60 + timestamps.dt.minute) // 30).astype(int)
    out["trade_day"] = timestamps.dt.date.astype(str)
    return out


def _drop_incomplete_latest_day(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    local_time = pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    latest_day = str(local_time.dt.date.max())
    latest_time = local_time[frame["trade_day"] == latest_day].max()
    latest_minute = int(latest_time.hour) * 60 + int(latest_time.minute)
    if latest_minute >= LAST_EVALUABLE_MINUTE:
        return frame
    return frame[frame["trade_day"] != latest_day].reset_index(drop=True)


def _load_1m_klines(symbol: str) -> pd.DataFrame:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT open_time, open, high, low, close, volume
        FROM klines
        WHERE symbol = ? AND interval = '1m'
        ORDER BY open_time ASC
        """,
        (symbol.upper(),),
    ).fetchall()
    conn.close()
    if not rows:
        raise ValueError(f"no 1m klines found for {symbol.upper()}")
    frame = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _simulate_trades(frame: pd.DataFrame, strategy_key: str) -> list[dict[str, Any]]:
    trades = []
    for _, row in frame.iterrows():
        rule = evaluate_optimized_rules(row.to_dict(), strategy_key=strategy_key)
        if rule is None:
            continue
        trades.append(_trade_result(row, rule))
    return trades


def _trade_result(row: pd.Series, rule: dict[str, Any]) -> dict[str, Any]:
    direction = str(rule["direction"])
    fwd_ret = float(row["fwd_ret"])
    win = (fwd_ret > 0) if direction == "up" else (fwd_ret < 0)
    return {
        "entryTime": int(row["open_time"]),
        "day": str(row["trade_day"]),
        "entryPrice": float(row["entry_price"]),
        "exitPrice": float(row["exit_price"]),
        "direction": direction,
        "rule": rule["name"],
        "win": bool(win),
        "pnlPerStake": FIXED_PAYOUT_RATIO if win else -1.0,
    }


def _backtest_report(data: BacktestReportInput) -> dict[str, Any]:
    daily = daily_stats(data.feature_frame, data.trades)
    overall = _overall_summary(data.trades, daily)
    failed_days = [row for row in daily if row["trades"] > 0 and row["winRate"] < RULE_TARGET_WIN_RATE]
    low_trade_days = _low_trade_days(daily, data.strategy)
    no_trade_days = [row["day"] for row in daily if row["trades"] == 0]
    return {
        "symbol": data.symbol,
        "strategyKey": data.strategy.key,
        "duration": data.duration,
        "source": data.strategy.signal_source,
        "liveExecutionUsesOrderbook": True,
        "historicalOrderbookSnapshots": _orderbook_snapshot_count(data.symbol),
        "target": {
            "dailyWinRateMin": RULE_TARGET_WIN_RATE,
            "dailyTradeLimit": None,
            "minDailyTrades": data.strategy.min_daily_trades,
        },
        "overall": overall,
        "daily": daily,
        "failedDays": failed_days,
        "lowTradeDays": low_trade_days,
        "noTradeDays": no_trade_days,
        "walkForward": data.walk_forward,
        "passed": passed(overall, failed_days, low_trade_days),
    }


def _overall_summary(trades: list[dict[str, Any]], daily: list[dict[str, Any]]) -> dict[str, Any]:
    overall = summary(trades)
    days = len(daily)
    return {**overall, "tradesPerDay": overall["trades"] / days if days else 0.0}


def _low_trade_days(
    daily: list[dict[str, Any]],
    strategy: StrategyDefinition,
) -> list[dict[str, Any]]:
    if strategy.min_daily_trades is None:
        return []
    return [row for row in daily if row["trades"] < strategy.min_daily_trades]


def _orderbook_snapshot_count(symbol: str) -> int:
    conn = get_conn()
    row = conn.execute(
        "SELECT COUNT(*) AS total FROM orderbook_features WHERE symbol = ?",
        (symbol.upper(),),
    ).fetchone()
    conn.close()
    return int(row["total"] if row else 0)


def _write_report(duration: str, strategy_key: str, report: dict[str, Any]) -> None:
    RULE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if strategy_key == DEFAULT_STRATEGY_KEY else f"_{strategy_key}"
    path = RULE_ARTIFACT_DIR / f"rule_backtest_{duration}{suffix}.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
