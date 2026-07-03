/**
 * 持仓盈亏实时计算Hook
 */
import { useEffect, useState } from 'react';

export function usePositionPnL(position, currentPrice) {
  const [pnl, setPnl] = useState({
    unrealized: 0,
    unrealizedPercent: 0,
    realizedToday: 0,
    total: 0,
    totalPercent: 0,
  });

  useEffect(() => {
    if (!position || !currentPrice || currentPrice <= 0) {
      setPnl({
        unrealized: 0,
        unrealizedPercent: 0,
        realizedToday: 0,
        total: 0,
        totalPercent: 0,
      });
      return;
    }

    const { avgPrice, quantity, side, realizedPnL = 0 } = position;

    if (!avgPrice || avgPrice <= 0 || !quantity || quantity === 0) {
      setPnl({
        unrealized: 0,
        unrealizedPercent: 0,
        realizedToday: realizedPnL,
        total: realizedPnL,
        totalPercent: 0,
      });
      return;
    }

    // 计算未实现盈亏
    let unrealized = 0;
    if (side === 'LONG' || side === 'BUY') {
      unrealized = (currentPrice - avgPrice) * Math.abs(quantity);
    } else if (side === 'SHORT' || side === 'SELL') {
      unrealized = (avgPrice - currentPrice) * Math.abs(quantity);
    }

    // 持仓成本
    const positionValue = avgPrice * Math.abs(quantity);

    // 未实现盈亏百分比
    const unrealizedPercent = positionValue > 0 ? (unrealized / positionValue) * 100 : 0;

    // 总盈亏
    const total = unrealized + realizedPnL;
    const totalPercent = positionValue > 0 ? (total / positionValue) * 100 : 0;

    setPnl({
      unrealized,
      unrealizedPercent,
      realizedToday: realizedPnL,
      total,
      totalPercent,
    });
  }, [position, currentPrice]);

  return pnl;
}

/**
 * 格式化盈亏显示
 */
export function formatPnL(value, showSign = true) {
  if (!value || !Number.isFinite(value)) {
    return '0.00';
  }

  const formatted = Math.abs(value).toFixed(2);

  if (showSign) {
    return value >= 0 ? `+${formatted}` : `-${formatted}`;
  }

  return formatted;
}

/**
 * 获取盈亏颜色类名
 */
export function getPnLColorClass(value) {
  if (!value || !Number.isFinite(value)) {
    return 'pnl-neutral';
  }

  return value > 0 ? 'pnl-profit' : value < 0 ? 'pnl-loss' : 'pnl-neutral';
}
