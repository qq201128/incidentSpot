from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.rule_config import (
    MS_PER_MINUTE,
    RULE_DURATION,
    RULE_TARGET_WIN_RATE,
    WALK_FORWARD_MIN_TRAIN_DAYS,
    WALK_FORWARD_PURGE_MINUTES,
    WALK_FORWARD_TEST_DAYS,
    walk_forward_purge_minutes_for_duration,
)


def walk_forward_validation(
    frame: pd.DataFrame,
    trades: list[dict[str, Any]],
    *,
    duration: str = RULE_DURATION,
) -> dict[str, Any]:
    purge_minutes = walk_forward_purge_minutes_for_duration(duration)
    days = sorted(str(day) for day in frame["trade_day"].unique())
    fold_results = _walk_forward_folds(frame, days, trades, purge_minutes=purge_minutes)
    test_trades = [trade for fold in fold_results for trade in fold["_testTrades"]]
    folds = [_public_walk_forward_fold(fold) for fold in fold_results]
    return {
        "method": "anchored_walk_forward_with_purge",
        "minTrainDays": WALK_FORWARD_MIN_TRAIN_DAYS,
        "testDays": WALK_FORWARD_TEST_DAYS,
        "purgeMinutes": purge_minutes,
        "folds": folds,
        "overall": summary(test_trades),
        "passed": _walk_forward_passed(folds, test_trades),
    }


def summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {"trades": 0, "wins": 0, "winRate": None, "avgPnlPerStake": None}
    wins = sum(1 for trade in trades if trade["win"])
    pnl = sum(float(trade["pnlPerStake"]) for trade in trades)
    return {
        "trades": len(trades),
        "wins": wins,
        "winRate": wins / len(trades),
        "avgPnlPerStake": pnl / len(trades),
    }


def daily_stats(feature_frame: pd.DataFrame, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    days = sorted(str(day) for day in feature_frame["trade_day"].unique())
    by_day = _trades_by_day(trades)
    return [_daily_row(day, by_day.get(day, [])) for day in days]


def passed(
    overall: dict[str, Any],
    failed_days: list[dict[str, Any]],
    low_trade_days: list[dict[str, Any]] | None = None,
) -> bool:
    if overall["winRate"] is None or overall["trades"] <= 0:
        return False
    if overall["winRate"] < RULE_TARGET_WIN_RATE:
        return False
    return not failed_days and not (low_trade_days or [])


def _walk_forward_folds(
    frame: pd.DataFrame,
    days: list[str],
    trades: list[dict[str, Any]],
    *,
    purge_minutes: int = WALK_FORWARD_PURGE_MINUTES,
) -> list[dict[str, Any]]:
    folds = []
    start = WALK_FORWARD_MIN_TRAIN_DAYS
    while start < len(days):
        train_days = days[:start]
        test_days = days[start:start + WALK_FORWARD_TEST_DAYS]
        if not test_days:
            break
        folds.append(_walk_forward_fold(
            frame, train_days, test_days, trades=trades, purge_minutes=purge_minutes
        ))
        start += WALK_FORWARD_TEST_DAYS
    return folds


def _walk_forward_fold(
    frame: pd.DataFrame,
    train_days: list[str],
    test_days: list[str],
    *,
    trades: list[dict[str, Any]],
    purge_minutes: int = WALK_FORWARD_PURGE_MINUTES,
) -> dict[str, Any]:
    test = frame[frame["trade_day"].isin(test_days)].copy()
    train_before = frame[frame["trade_day"].isin(train_days)].copy()
    train = _purged_train_frame(train_before, test, purge_minutes=purge_minutes)
    train_trades = _trades_in_frame(trades, train)
    test_trades = _trades_in_frame(trades, test)
    return {
        "trainStart": train_days[0],
        "trainEnd": train_days[-1],
        "testStart": test_days[0],
        "testEnd": test_days[-1],
        "trainRows": int(len(train)),
        "purgedRows": int(len(train_before) - len(train)),
        "testRows": int(len(test)),
        "trainOverall": summary(train_trades),
        "overall": summary(test_trades),
        "failedDays": _failed_walk_forward_days(test, test_trades),
        "_testTrades": test_trades,
    }


def _failed_walk_forward_days(test: pd.DataFrame, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in daily_stats(test, trades)
        if row["trades"] > 0 and row["winRate"] < RULE_TARGET_WIN_RATE
    ]


def _trades_in_frame(trades: list[dict[str, Any]], frame: pd.DataFrame) -> list[dict[str, Any]]:
    open_times = set(int(value) for value in frame["open_time"].tolist())
    return [trade for trade in trades if int(trade["entryTime"]) in open_times]


def _public_walk_forward_fold(fold: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fold.items() if not key.startswith("_")}


def _purged_train_frame(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    purge_minutes: int = WALK_FORWARD_PURGE_MINUTES,
) -> pd.DataFrame:
    if train.empty or test.empty:
        return train
    purge_ms = purge_minutes * MS_PER_MINUTE
    test_start = int(test["open_time"].min())
    return train[train["open_time"] < test_start - purge_ms]


def _walk_forward_passed(folds: list[dict[str, Any]], trades: list[dict[str, Any]]) -> bool:
    overall = summary(trades)
    if overall["winRate"] is None or overall["winRate"] < RULE_TARGET_WIN_RATE:
        return False
    return all(not fold["failedDays"] for fold in folds)


def _trades_by_day(trades: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        grouped.setdefault(str(trade["day"]), []).append(trade)
    return grouped


def _daily_row(day: str, trades: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(trades)
    wins = sum(1 for trade in trades if trade["win"])
    pnl = sum(float(trade["pnlPerStake"]) for trade in trades)
    return {
        "day": day,
        "trades": total,
        "wins": wins,
        "winRate": wins / total if total else None,
        "avgPnlPerStake": pnl / total if total else None,
    }
