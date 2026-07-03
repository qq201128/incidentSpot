"""
组合因子周期评分服务
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.db.session import get_conn

logger = logging.getLogger(__name__)

# 支持的周期
SUPPORTED_PERIODS = ["1m", "5m", "10m", "30m", "1h", "4h", "1d"]


def calculate_period_scores(factor_id: int, symbol: str = "BTCUSDT") -> dict[str, Any]:
    """
    计算因子在各周期的评分

    Args:
        factor_id: 因子ID
        symbol: 交易对

    Returns:
        各周期的评分数据
    """
    conn = get_conn()
    scores = {}

    for period in SUPPORTED_PERIODS:
        try:
            # 获取该周期的回测指标
            metrics = get_period_metrics(conn, factor_id, symbol, period)

            if not metrics:
                scores[period] = None
                continue

            # 计算综合评分
            score = calculate_composite_score(metrics)

            scores[period] = {
                "score": score,
                "win_rate": metrics.get("win_rate"),
                "icir": metrics.get("icir"),
                "sharpe": metrics.get("sharpe"),
                "trades": metrics.get("trades"),
                "max_drawdown": metrics.get("max_drawdown"),
            }

        except Exception as e:
            logger.error(f"Failed to calculate score for {period}: {e}")
            scores[period] = None

    conn.close()

    return scores


def get_period_metrics(conn, factor_id: int, symbol: str, period: str) -> dict | None:
    """
    获取特定周期的回测指标

    可能的数据来源：
    1. factor_period_metrics 表（如果存在）
    2. 从 factor_trades 表聚合计算
    3. 从主表的 period_scores JSON字段读取
    """
    # 方案1：尝试从专门的周期指标表读取
    try:
        row = conn.execute("""
            SELECT win_rate, icir, sharpe, trades, max_drawdown
            FROM factor_period_metrics
            WHERE factor_id = ? AND symbol = ? AND period = ?
        """, (factor_id, symbol, period)).fetchone()

        if row:
            return dict(row)
    except:
        pass

    # 方案2：从主表的JSON字段读取
    try:
        import json
        row = conn.execute("""
            SELECT period_scores
            FROM factor_combinations
            WHERE id = ?
        """, (factor_id,)).fetchone()

        if row and row["period_scores"]:
            period_scores = json.loads(row["period_scores"])
            if period in period_scores:
                return period_scores[period]
    except:
        pass

    # 方案3：从交易记录计算（如果有period字段）
    try:
        trades = conn.execute("""
            SELECT pnl, position_value
            FROM factor_trades
            WHERE factor_id = ? AND symbol = ? AND period = ?
        """, (factor_id, symbol, period)).fetchall()

        if trades and len(trades) >= 5:
            return calculate_metrics_from_trades(trades)
    except:
        pass

    return None


def calculate_metrics_from_trades(trades: list) -> dict:
    """从交易记录计算指标"""
    returns = []

    for trade in trades:
        pnl = trade.get("pnl", 0)
        position_value = trade.get("position_value", 1000)

        if position_value > 0:
            returns.append(pnl / position_value)

    if not returns:
        return None

    returns_arr = np.array(returns)

    # 胜率
    win_rate = float(np.sum(returns_arr > 0) / len(returns_arr))

    # ICIR
    mean_return = np.mean(returns_arr)
    std_return = np.std(returns_arr)
    icir = float(mean_return / std_return if std_return > 0 else 0)

    # 夏普比率
    sharpe = float(mean_return / std_return * np.sqrt(252) if std_return > 0 else 0)

    # 最大回撤
    cumulative = np.cumsum(returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = float(np.min(drawdown))

    return {
        "win_rate": win_rate,
        "icir": icir,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "trades": len(trades),
    }


def calculate_composite_score(metrics: dict) -> float:
    """
    计算综合评分

    权重：
    - 胜率: 40%
    - ICIR: 30%
    - 夏普: 20%
    - 交易次数: 10%
    """
    win_rate = metrics.get("win_rate", 0)
    icir = metrics.get("icir", 0)
    sharpe = metrics.get("sharpe", 0)
    trades = metrics.get("trades", 0)

    # 归一化各指标到0-100
    wr_score = min(100, win_rate * 150)  # 60%胜率 = 90分
    icir_score = min(100, abs(icir) * 200)  # 0.5 ICIR = 100分
    sharpe_score = min(100, abs(sharpe) * 50)  # 2.0 夏普 = 100分

    # 交易次数得分
    if trades < 10:
        trades_score = trades * 5
    elif trades < 100:
        trades_score = 50 + (trades - 10) * 0.5
    else:
        trades_score = 100

    # 综合评分
    score = (
        0.4 * wr_score +
        0.3 * icir_score +
        0.2 * sharpe_score +
        0.1 * trades_score
    )

    return round(score, 1)


def batch_calculate_period_scores(
    factor_ids: list[int] | None = None,
    symbol: str = "BTCUSDT"
) -> dict[int, dict]:
    """
    批量计算周期评分

    Args:
        factor_ids: 因子ID列表（None表示全部）
        symbol: 交易对

    Returns:
        {factor_id: period_scores}
    """
    conn = get_conn()

    if factor_ids is None:
        # 获取所有因子
        rows = conn.execute("SELECT id FROM factor_combinations").fetchall()
        factor_ids = [row["id"] for row in rows]

    results = {}

    for factor_id in factor_ids:
        try:
            scores = calculate_period_scores(factor_id, symbol)
            results[factor_id] = scores

            # 保存到数据库
            save_period_scores(conn, factor_id, scores)

            logger.info(f"Calculated scores for factor {factor_id}")

        except Exception as e:
            logger.error(f"Failed to calculate scores for factor {factor_id}: {e}")
            results[factor_id] = None

    conn.commit()
    conn.close()

    return results


def save_period_scores(conn, factor_id: int, scores: dict):
    """保存周期评分到数据库"""
    import json

    try:
        conn.execute("""
            UPDATE factor_combinations
            SET period_scores = ?
            WHERE id = ?
        """, (json.dumps(scores), factor_id))
    except Exception as e:
        logger.error(f"Failed to save period scores for factor {factor_id}: {e}")


def get_best_period_for_factor(factor_id: int, symbol: str = "BTCUSDT") -> str | None:
    """
    获取因子的最佳周期

    Returns:
        最佳周期（如 "10m"）
    """
    scores = calculate_period_scores(factor_id, symbol)

    if not scores:
        return None

    # 找到评分最高的周期
    best_period = None
    best_score = -1

    for period, data in scores.items():
        if data and data.get("score", 0) > best_score:
            best_score = data["score"]
            best_period = period

    return best_period
