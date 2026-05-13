from __future__ import annotations

from typing import Any

from app.db.session import get_conn
from app.services.factor_learning_common import round_metric, utc_now
from app.services.factor_combo_simulation_keys import factor_combo_simulation_strategy_keys
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY

MONITOR_SAMPLE_LIMIT = 200
MIN_MONITOR_SAMPLE_COUNT = 10
LOW_SUCCESS_RATE = 0.45
LOW_PASSED_SUCCESS_RATE = 0.48
CONSECUTIVE_LOSS_ALERT_COUNT = 3


def factor_combo_monitor_report(symbol: str, duration: str) -> dict[str, Any]:
    rows = _settled_prediction_rows(symbol, duration)
    metrics = _monitor_metrics(rows)
    issues = _monitor_issues(metrics, rows)
    return {
        "status": _monitor_status(metrics, issues),
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "updatedAt": utc_now(),
        "strategyKey": FACTOR_COMBO_STRATEGY_KEY,
        "thresholds": _threshold_payload(),
        "metrics": metrics,
        "issues": issues,
        "solutions": _solutions(issues),
    }


def _settled_prediction_rows(symbol: str, duration: str) -> list[dict[str, Any]]:
    strategy_keys = factor_combo_simulation_strategy_keys()
    placeholders = ",".join("?" for _key in strategy_keys)
    conn = get_conn()
    try:
        rows = conn.execute(
            f"""
            SELECT open_time, direction, confidence, trade_quality_score,
                   trade_quality_passed, actual_return, prediction_correct,
                   high_winrate_rule, strategy_key
            FROM predictions
            WHERE strategy_key IN ({placeholders}) AND symbol = ? AND duration = ?
              AND settled_at IS NOT NULL
            ORDER BY open_time DESC
            LIMIT ?
            """,
            (*strategy_keys, symbol.strip().upper(), duration, MONITOR_SAMPLE_LIMIT),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _monitor_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [row for row in rows if bool(row.get("trade_quality_passed"))]
    return {
        "sampleCount": len(rows),
        "qualityPassedCount": len(passed),
        "predictionSuccessRate": _success_rate(rows),
        "qualityPassedSuccessRate": _success_rate(passed),
        "qualityPassRate": _ratio(len(passed), len(rows)),
        "latestConsecutiveLosses": _latest_consecutive_losses(rows),
    }


def _monitor_issues(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if metrics["sampleCount"] < MIN_MONITOR_SAMPLE_COUNT:
        return []
    issues = []
    if metrics["predictionSuccessRate"] < LOW_SUCCESS_RATE:
        issues.append(_issue("prediction_success_rate_low", metrics["predictionSuccessRate"]))
    if _passed_rate_low(metrics):
        issues.append(_issue("quality_passed_success_rate_low", metrics["qualityPassedSuccessRate"]))
    if metrics["latestConsecutiveLosses"] >= CONSECUTIVE_LOSS_ALERT_COUNT:
        issues.append(_loss_streak_issue(rows))
    return issues


def _issue(code: str, value: float | None) -> dict[str, Any]:
    return {
        "code": code,
        "severity": "high",
        "value": value,
        "message": _issue_message(code),
        "solutionKeys": _solution_keys(code),
    }


def _loss_streak_issue(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest_rules = [str(row.get("high_winrate_rule") or "") for row in rows[:CONSECUTIVE_LOSS_ALERT_COUNT]]
    payload = _issue("recent_loss_streak", float(CONSECUTIVE_LOSS_ALERT_COUNT))
    return {**payload, "latestRules": latest_rules}


def _monitor_status(metrics: dict[str, Any], issues: list[dict[str, Any]]) -> str:
    if metrics["sampleCount"] < MIN_MONITOR_SAMPLE_COUNT:
        return "insufficient_data"
    return "warning" if issues else "healthy"


def _passed_rate_low(metrics: dict[str, Any]) -> bool:
    return (
        metrics["qualityPassedCount"] >= MIN_MONITOR_SAMPLE_COUNT
        and metrics["qualityPassedSuccessRate"] < LOW_PASSED_SUCCESS_RATE
    )


def _success_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    wins = sum(1 for row in rows if bool(row.get("prediction_correct")))
    return _ratio(wins, len(rows))


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round_metric(numerator / denominator, 4)


def _latest_consecutive_losses(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if bool(row.get("prediction_correct")):
            break
        count += 1
    return count


def _solutions(issues: list[dict[str, Any]]) -> list[str]:
    keys = []
    for issue in issues:
        keys.extend(issue.get("solutionKeys") or [])
    return [_solution_text(key) for key in dict.fromkeys(keys)]


def _threshold_payload() -> dict[str, float | int]:
    return {
        "minSampleCount": MIN_MONITOR_SAMPLE_COUNT,
        "lowSuccessRate": LOW_SUCCESS_RATE,
        "lowPassedSuccessRate": LOW_PASSED_SUCCESS_RATE,
        "consecutiveLossAlertCount": CONSECUTIVE_LOSS_ALERT_COUNT,
    }


def _issue_message(code: str) -> str:
    messages = {
        "prediction_success_rate_low": "多因子组合整体预测成功率偏低",
        "quality_passed_success_rate_low": "已通过质量过滤的模拟单成功率偏低",
        "recent_loss_streak": "最近连续亏损，需要复核当前组合与亏损特征",
    }
    return messages[code]


def _solution_keys(code: str) -> list[str]:
    mapping = {
        "prediction_success_rate_low": ["refresh_learning", "tighten_promotions"],
        "quality_passed_success_rate_low": ["raise_confirmations", "downweight_loss_features"],
        "recent_loss_streak": ["block_recent_loss_patterns", "inspect_latest_combo"],
    }
    return mapping[code]


def _solution_text(key: str) -> str:
    texts = {
        "refresh_learning": "重新执行本地复盘，结算到期预测并刷新亏损模式记忆。",
        "tighten_promotions": "仅把同时满足胜率和盈亏比阈值的组合写入挖掘因子库。",
        "raise_confirmations": "提高成员方向确认数，让单一成员反向时不进入模拟候选。",
        "downweight_loss_features": "对命中亏损特征的成员降权，下一次组合打分时显式避开。",
        "block_recent_loss_patterns": "把最近连续亏损对应的特征阈值加入硬过滤观察区。",
        "inspect_latest_combo": "检查最近亏损组合的成员、方向和盈亏比，优先淘汰重复命中的组合。",
    }
    return texts[key]
