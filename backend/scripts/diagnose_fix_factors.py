"""
组合因子回测诊断和修复工具
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging
from datetime import datetime

import numpy as np

from app.db.session import get_conn

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def diagnose_factors():
    """诊断组合因子状态"""
    conn = get_conn()

    logger.info("=" * 80)
    logger.info("组合因子诊断报告")
    logger.info("=" * 80)

    # 1. 总体统计
    stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN icir IS NOT NULL THEN 1 END) as has_icir,
            COUNT(CASE WHEN win_rate IS NOT NULL THEN 1 END) as has_win_rate,
            COUNT(CASE WHEN backtest_completed = 1 THEN 1 END) as completed,
            COUNT(CASE WHEN backtest_completed = 0 THEN 1 END) as pending
        FROM factor_combinations
    """).fetchone()

    logger.info(f"\n📊 总体统计:")
    logger.info(f"  总数: {stats['total']}")
    logger.info(f"  已完成回测: {stats['completed']}")
    logger.info(f"  待回测: {stats['pending']}")
    logger.info(f"  有ICIR数据: {stats['has_icir']}")
    logger.info(f"  有胜率数据: {stats['has_win_rate']}")

    # 2. 不完整的回测
    incomplete = conn.execute("""
        SELECT id, name, created_at, backtest_completed, icir, win_rate, trades
        FROM factor_combinations
        WHERE backtest_completed = 1
        AND (icir IS NULL OR win_rate IS NULL)
        ORDER BY created_at DESC
        LIMIT 10
    """).fetchall()

    if incomplete:
        logger.warning(f"\n⚠️ 发现 {len(incomplete)} 个不完整的回测:")
        for row in incomplete:
            logger.warning(
                f"  - ID:{row['id']} {row['name'][:50]} "
                f"(ICIR={row['icir']}, 胜率={row['win_rate']}, 交易={row['trades']})"
            )

    # 3. 最近的组合因子
    recent = conn.execute("""
        SELECT id, name, created_at, icir, win_rate, trades
        FROM factor_combinations
        ORDER BY created_at DESC
        LIMIT 5
    """).fetchall()

    logger.info(f"\n📅 最近5个组合因子:")
    for row in recent:
        logger.info(
            f"  - ID:{row['id']} {row['name'][:50]} "
            f"(ICIR={row['icir']}, 胜率={row['win_rate']:.2%} if row['win_rate'] else 'N/A', "
            f"交易={row['trades']})"
        )

    # 4. 表现最好的因子
    top_performers = conn.execute("""
        SELECT id, name, icir, win_rate, sharpe, trades
        FROM factor_combinations
        WHERE icir IS NOT NULL
        ORDER BY icir DESC
        LIMIT 5
    """).fetchall()

    if top_performers:
        logger.info(f"\n🏆 表现最好的5个因子:")
        for row in top_performers:
            logger.info(
                f"  - ID:{row['id']} {row['name'][:50]} "
                f"(ICIR={row['icir']:.3f}, 胜率={row['win_rate']:.2%}, "
                f"夏普={row['sharpe']:.2f}, 交易={row['trades']})"
            )

    conn.close()

    logger.info("\n" + "=" * 80)

    return stats


def fix_incomplete_backtests(dry_run=True, limit=None):
    """
    修复不完整的回测

    Args:
        dry_run: 只检查不修复
        limit: 最多修复数量
    """
    conn = get_conn()

    # 查找需要修复的因子
    query = """
        SELECT fc.id, fc.name, fc.created_at
        FROM factor_combinations fc
        WHERE fc.backtest_completed = 1
        AND (fc.icir IS NULL OR fc.win_rate IS NULL)
        ORDER BY fc.created_at DESC
    """

    if limit:
        query += f" LIMIT {limit}"

    to_fix = conn.execute(query).fetchall()

    logger.info(f"\n🔧 需要修复的因子: {len(to_fix)}")

    if not to_fix:
        logger.info("✅ 没有需要修复的因子")
        conn.close()
        return {"fixed": 0, "failed": 0}

    if dry_run:
        for row in to_fix:
            logger.info(f"  - ID:{row['id']} {row['name'][:50]}")
        conn.close()
        return {"dry_run": True, "count": len(to_fix)}

    # 执行修复
    fixed_count = 0
    failed_count = 0

    for row in to_fix:
        factor_id = row['id']
        factor_name = row['name']

        try:
            logger.info(f"\n修复 ID:{factor_id} {factor_name[:50]}...")

            # 重新计算指标
            metrics = recalculate_metrics(conn, factor_id)

            if not metrics:
                logger.warning(f"  ⚠️ 无法计算指标（可能缺少交易记录）")
                failed_count += 1
                continue

            # 更新数据库
            conn.execute("""
                UPDATE factor_combinations
                SET icir = ?,
                    win_rate = ?,
                    sharpe = ?,
                    max_drawdown = ?,
                    trades = ?,
                    updated_at = ?
                WHERE id = ?
            """, (
                metrics['icir'],
                metrics['win_rate'],
                metrics['sharpe'],
                metrics['max_drawdown'],
                metrics['trades'],
                datetime.now().isoformat(),
                factor_id
            ))

            logger.info(
                f"  ✅ 修复成功: ICIR={metrics['icir']:.3f}, "
                f"胜率={metrics['win_rate']:.2%}, 交易={metrics['trades']}"
            )
            fixed_count += 1

        except Exception as e:
            logger.error(f"  ❌ 修复失败: {e}")
            failed_count += 1

    conn.commit()
    conn.close()

    logger.info(f"\n✅ 修复完成: 成功{fixed_count}, 失败{failed_count}")

    return {"fixed": fixed_count, "failed": failed_count}


def recalculate_metrics(conn, factor_id):
    """重新计算回测指标"""
    # 注意：这里需要根据实际的交易记录表名调整
    # 可能是 factor_trades, factor_combination_trades 等

    # 尝试多个可能的表名
    tables_to_try = [
        'factor_trades',
        'factor_combination_trades',
        'trades',
    ]

    trades = None
    for table in tables_to_try:
        try:
            trades = conn.execute(f"""
                SELECT * FROM {table}
                WHERE factor_id = ? OR factor_combination_id = ?
                ORDER BY entry_time
            """, (factor_id, factor_id)).fetchall()

            if trades:
                break
        except:
            continue

    if not trades or len(trades) < 5:
        return None

    # 计算收益率序列
    returns = []
    for trade in trades:
        pnl = trade.get('pnl', 0)
        position_value = trade.get('position_value') or trade.get('notional') or 1000

        if position_value > 0:
            returns.append(pnl / position_value)

    if not returns:
        return None

    # 计算指标
    returns_arr = np.array(returns)

    # 胜率
    win_rate = np.sum(returns_arr > 0) / len(returns_arr)

    # ICIR (Information Coefficient * √频率)
    mean_return = np.mean(returns_arr)
    std_return = np.std(returns_arr)
    icir = mean_return / std_return if std_return > 0 else 0

    # 夏普比率
    sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0

    # 最大回撤
    cumulative = np.cumsum(returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = np.min(drawdown)

    return {
        'icir': float(icir),
        'win_rate': float(win_rate),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'trades': len(trades),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="组合因子回测诊断和修复")
    parser.add_argument('--diagnose', action='store_true', help="诊断当前状态")
    parser.add_argument('--fix', action='store_true', help="修复不完整的回测")
    parser.add_argument('--dry-run', action='store_true', help="只检查不修复")
    parser.add_argument('--limit', type=int, help="最多修复数量")

    args = parser.parse_args()

    if args.diagnose or (not args.fix):
        # 默认执行诊断
        diagnose_factors()

    if args.fix:
        fix_incomplete_backtests(dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
