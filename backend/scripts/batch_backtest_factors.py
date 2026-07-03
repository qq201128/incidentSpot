"""
批量回测组合因子脚本
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from app.db.session import get_conn
from app.services.binance_service import fetch_klines

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


async def backtest_factor(factor_id: int, symbol: str = "BTCUSDT", period: str = "10m", lookback_days: int = 30):
    """
    回测单个因子

    Args:
        factor_id: 因子ID
        symbol: 交易对
        period: 周期
        lookback_days: 回测天数
    """
    conn = get_conn()

    try:
        # 1. 获取因子信息
        factor = conn.execute("""
            SELECT id, name, formula, members
            FROM factor_combinations
            WHERE id = ?
        """, (factor_id,)).fetchone()

        if not factor:
            logger.error(f"Factor {factor_id} not found")
            return None

        logger.info(f"回测因子 {factor_id}: {factor['name'][:50]}")

        # 2. 获取历史K线数据
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)

        logger.info(f"  获取K线数据: {start_time.date()} ~ {end_time.date()}")

        klines = await asyncio.to_thread(
            fetch_klines,
            symbol,
            period,
            start_time=int(start_time.timestamp() * 1000),
            end_time=int(end_time.timestamp() * 1000),
            limit=1500
        )

        if not klines or len(klines) < 50:
            logger.warning(f"  ⚠️ K线数据不足: {len(klines) if klines else 0} 条")
            return None

        logger.info(f"  获取到 {len(klines)} 条K线")

        # 3. 生成交易信号（简化版）
        signals = generate_signals(klines, factor)

        if not signals:
            logger.warning(f"  ⚠️ 没有生成交易信号")
            return None

        logger.info(f"  生成 {len(signals)} 个交易信号")

        # 4. 计算回测指标
        metrics = calculate_backtest_metrics(signals, klines)

        logger.info(
            f"  ✅ 回测完成: ICIR={metrics['icir']:.3f}, "
            f"胜率={metrics['win_rate']:.2%}, 交易={metrics['trades']}"
        )

        # 5. 保存结果
        conn.execute("""
            UPDATE factor_combinations
            SET backtest_completed = 1,
                icir = ?,
                win_rate = ?,
                sharpe = ?,
                max_drawdown = ?,
                trades = ?,
                last_backtest_at = ?
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

        conn.commit()

        return metrics

    except Exception as e:
        logger.error(f"  ❌ 回测失败: {e}", exc_info=True)
        return None

    finally:
        conn.close()


def generate_signals(klines: list, factor: dict) -> list:
    """
    生成交易信号（简化版）

    实际应用中应该：
    1. 解析factor['formula']
    2. 计算技术指标
    3. 根据公式生成信号

    这里使用简化逻辑演示
    """
    signals = []

    # 提取价格序列
    closes = np.array([float(k['close']) for k in klines])
    highs = np.array([float(k['high']) for k in klines])
    lows = np.array([float(k['low']) for k in klines])

    # 简单均线策略示例
    if len(closes) < 20:
        return []

    # 计算短期和长期均线
    ma_short = np.convolve(closes, np.ones(7) / 7, mode='valid')
    ma_long = np.convolve(closes, np.ones(20) / 20, mode='valid')

    # 对齐长度
    min_len = min(len(ma_short), len(ma_long))
    ma_short = ma_short[-min_len:]
    ma_long = ma_long[-min_len:]

    # 生成信号
    for i in range(1, min_len):
        # 金叉：买入信号
        if ma_short[i-1] <= ma_long[i-1] and ma_short[i] > ma_long[i]:
            entry_idx = len(closes) - min_len + i
            entry_price = closes[entry_idx]

            # 寻找出场点（简化：固定持仓5根K线）
            exit_idx = min(entry_idx + 5, len(closes) - 1)
            exit_price = closes[exit_idx]

            pnl = exit_price - entry_price

            signals.append({
                'entry_time': klines[entry_idx]['open_time'],
                'exit_time': klines[exit_idx]['open_time'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'side': 'LONG',
            })

        # 死叉：卖出信号（做空）
        elif ma_short[i-1] >= ma_long[i-1] and ma_short[i] < ma_long[i]:
            entry_idx = len(closes) - min_len + i
            entry_price = closes[entry_idx]

            exit_idx = min(entry_idx + 5, len(closes) - 1)
            exit_price = closes[exit_idx]

            pnl = entry_price - exit_price

            signals.append({
                'entry_time': klines[entry_idx]['open_time'],
                'exit_time': klines[exit_idx]['open_time'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'side': 'SHORT',
            })

    return signals


def calculate_backtest_metrics(signals: list, klines: list) -> dict:
    """计算回测指标"""
    if not signals:
        return {
            'icir': 0,
            'win_rate': 0,
            'sharpe': 0,
            'max_drawdown': 0,
            'trades': 0,
        }

    # 计算收益率
    returns = []
    avg_price = np.mean([float(k['close']) for k in klines])

    for signal in signals:
        position_value = avg_price  # 简化：假设固定仓位
        ret = signal['pnl'] / position_value if position_value > 0 else 0
        returns.append(ret)

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
        'icir': icir,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'trades': len(signals),
    }


async def batch_backtest_factors(
    factor_ids: list[int] | None = None,
    symbol: str = "BTCUSDT",
    period: str = "10m",
    lookback_days: int = 30,
    limit: int | None = None
):
    """
    批量回测因子

    Args:
        factor_ids: 因子ID列表（None表示全部未回测的）
        symbol: 交易对
        period: 周期
        lookback_days: 回测天数
        limit: 最多回测数量
    """
    conn = get_conn()

    if factor_ids is None:
        # 获取未回测或回测不完整的因子
        query = """
            SELECT id
            FROM factor_combinations
            WHERE backtest_completed = 0
            OR icir IS NULL
            OR win_rate IS NULL
            ORDER BY created_at DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        rows = conn.execute(query).fetchall()
        factor_ids = [row['id'] for row in rows]

    conn.close()

    logger.info(f"=" * 80)
    logger.info(f"批量回测因子")
    logger.info(f"  总数: {len(factor_ids)}")
    logger.info(f"  交易对: {symbol}")
    logger.info(f"  周期: {period}")
    logger.info(f"  回测天数: {lookback_days}")
    logger.info(f"=" * 80)

    success_count = 0
    failed_count = 0

    for i, factor_id in enumerate(factor_ids, 1):
        logger.info(f"\n[{i}/{len(factor_ids)}] 回测因子 {factor_id}")

        try:
            result = await backtest_factor(factor_id, symbol, period, lookback_days)

            if result:
                success_count += 1
            else:
                failed_count += 1

            # 避免请求过快
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"回测失败: {e}")
            failed_count += 1

    logger.info(f"\n" + "=" * 80)
    logger.info(f"批量回测完成")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {failed_count}")
    logger.info(f"=" * 80)


def main():
    parser = argparse.ArgumentParser(description="批量回测组合因子")
    parser.add_argument('--factor-ids', type=int, nargs='+', help="指定因子ID列表")
    parser.add_argument('--symbol', default='BTCUSDT', help="交易对")
    parser.add_argument('--period', default='10m', help="周期")
    parser.add_argument('--lookback-days', type=int, default=30, help="回测天数")
    parser.add_argument('--limit', type=int, help="最多回测数量")

    args = parser.parse_args()

    asyncio.run(batch_backtest_factors(
        factor_ids=args.factor_ids,
        symbol=args.symbol,
        period=args.period,
        lookback_days=args.lookback_days,
        limit=args.limit
    ))


if __name__ == "__main__":
    main()
