from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricRule:
    path: str
    label: str
    direction: str
    tolerance: float


METRIC_RULES = (
    MetricRule("test_auc", "测试区分度", "higher", 0.02),
    MetricRule("test_f1", "测试综合分数", "higher", 0.03),
    MetricRule("test_brier_calibrated", "校准误差", "lower", 0.02),
    MetricRule("test_logloss_calibrated", "校准损失", "lower", 0.04),
    MetricRule("backtest_test_split.win_rate", "回测胜率", "higher", 0.03),
    MetricRule("backtest_test_split.direction_hit_rate", "方向命中率", "higher", 0.03),
    MetricRule("backtest_test_split.avg_trade_return", "平均交易收益", "higher", 0.0003),
)
TARGET_WIN_RATE = 0.90
TARGET_TRADES_PER_DAY = 5.0


def metric_summary(meta: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "symbol",
        "duration",
        "horizon_minutes",
        "min_move_bps",
        "train_window_days",
        "val_auc",
        "test_auc",
        "test_f1",
        "test_brier_calibrated",
        "test_logloss_calibrated",
        "best_threshold",
        "row_count",
        "labeled_row_count",
    )
    out = {key: meta[key] for key in keys if key in meta}
    if "backtest_test_split" in meta:
        out["backtest_test_split"] = meta["backtest_test_split"]
    if "backtest_confidence_profiles" in meta:
        out["backtest_confidence_profiles"] = meta["backtest_confidence_profiles"]
    if "backtest_quality_profiles" in meta:
        out["backtest_quality_profiles"] = meta["backtest_quality_profiles"]
    if "production_gate_backtest" in meta:
        out["production_gate_backtest"] = meta["production_gate_backtest"]
    out["production_target"] = production_target_summary(meta)
    return out


def production_target_summary(meta: dict[str, Any]) -> dict[str, Any]:
    source, backtest = _production_backtest(meta)
    win_rate = _dict_float(backtest, "win_rate")
    trades_per_day = _dict_float(backtest, "trades_per_day")
    passed = (
        win_rate is not None
        and trades_per_day is not None
        and win_rate > TARGET_WIN_RATE
        and trades_per_day >= TARGET_TRADES_PER_DAY
    )
    return {
        "targetWinRate": TARGET_WIN_RATE,
        "targetTradesPerDay": TARGET_TRADES_PER_DAY,
        "winRate": win_rate,
        "tradesPerDay": trades_per_day,
        "passed": passed,
        "source": source,
    }


def _production_backtest(meta: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    gate_backtest = meta.get("production_gate_backtest")
    if isinstance(gate_backtest, dict):
        return "production_gate_backtest", gate_backtest
    split_backtest = meta.get("backtest_test_split")
    if isinstance(split_backtest, dict):
        return "backtest_test_split", split_backtest
    return "missing", {}


def _dict_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    return float(value)


def compare_candidate_meta(
    current_meta: dict[str, Any] | None,
    candidate_meta: dict[str, Any],
) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    target_item = _production_target_decision(candidate_meta)
    checked.append(target_item)
    if target_item["failed"]:
        failed.append(target_item)
    if current_meta is None:
        return {
            "approved": not failed,
            "checked": checked,
            "failed": failed,
            "reason": "no active baseline; production target checked",
        }
    for rule in METRIC_RULES:
        current = _metric_value(current_meta, rule.path)
        candidate = _metric_value(candidate_meta, rule.path)
        if current is None or candidate is None:
            continue
        item = _metric_decision(rule, current, candidate)
        checked.append(item)
        if item["failed"]:
            failed.append(item)
    return {
        "approved": not failed,
        "checked": checked,
        "failed": failed,
        "reason": "within configured tolerances" if checked else "no comparable metrics",
    }


def _metric_value(meta: dict[str, Any], path: str) -> float | None:
    node: Any = meta
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    value = float(node)
    return value if math.isfinite(value) else None


def _metric_decision(rule: MetricRule, current: float, candidate: float) -> dict[str, Any]:
    worse_by = current - candidate if rule.direction == "higher" else candidate - current
    failed = worse_by > rule.tolerance
    return {
        "path": rule.path,
        "label": rule.label,
        "direction": rule.direction,
        "current": current,
        "candidate": candidate,
        "tolerance": rule.tolerance,
        "worseBy": worse_by,
        "failed": failed,
    }


def _production_target_decision(meta: dict[str, Any]) -> dict[str, Any]:
    summary = production_target_summary(meta)
    return {
        "path": "production_target",
        "label": f"生产目标：胜率>{TARGET_WIN_RATE:.0%} 且每日交易>={TARGET_TRADES_PER_DAY:g}",
        "direction": "target",
        "current": None,
        "candidate": {
            "winRate": summary["winRate"],
            "tradesPerDay": summary["tradesPerDay"],
        },
        "tolerance": 0,
        "worseBy": 0,
        "failed": not summary["passed"],
    }
