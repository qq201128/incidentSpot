/**
 * 订单簿热力图组件
 */
import { memo, useMemo } from 'react';
import './OrderbookHeatmap.css';

export default function OrderbookHeatmap({ bids = [], asks = [], compact = false }) {
  const maxVolume = useMemo(() => {
    const allLevels = [...bids, ...asks];
    if (allLevels.length === 0) return 1;
    return Math.max(...allLevels.map(([, qty]) => Number(qty)));
  }, [bids, asks]);

  const spread = useMemo(() => {
    if (asks.length === 0 || bids.length === 0) return null;
    const bestAsk = Number(asks[0][0]);
    const bestBid = Number(bids[0][0]);
    const spreadValue = bestAsk - bestBid;
    const mid = (bestAsk + bestBid) / 2;
    const spreadBps = mid > 0 ? (spreadValue / mid) * 10000 : 0;
    return { value: spreadValue, bps: spreadBps };
  }, [asks, bids]);

  return (
    <div className={`orderbook-heatmap ${compact ? 'orderbook-heatmap--compact' : ''}`}>
      <div className="orderbook-heatmap-header">
        <span>价格</span>
        <span>数量</span>
        <span>累计</span>
      </div>

      {/* 卖单（从上到下，价格从高到低）*/}
      <div className="orderbook-heatmap-asks">
        {[...asks].reverse().map(([price, qty], idx) => (
          <HeatmapRow
            key={`ask-${idx}`}
            price={Number(price)}
            qty={Number(qty)}
            maxVolume={maxVolume}
            side="ask"
            cumulative={calculateCumulative(asks, idx, 'ask')}
          />
        ))}
      </div>

      {/* 价差 */}
      {spread && (
        <div className="orderbook-heatmap-spread">
          <span className="spread-value">{spread.value.toFixed(2)}</span>
          <span className="spread-bps">({spread.bps.toFixed(2)} bps)</span>
        </div>
      )}

      {/* 买单 */}
      <div className="orderbook-heatmap-bids">
        {bids.map(([price, qty], idx) => (
          <HeatmapRow
            key={`bid-${idx}`}
            price={Number(price)}
            qty={Number(qty)}
            maxVolume={maxVolume}
            side="bid"
            cumulative={calculateCumulative(bids, idx, 'bid')}
          />
        ))}
      </div>
    </div>
  );
}

const HeatmapRow = memo(function HeatmapRow({ price, qty, maxVolume, side, cumulative }) {
  const intensity = maxVolume > 0 ? (qty / maxVolume) * 100 : 0;

  return (
    <div className={`heatmap-row heatmap-row--${side}`}>
      <span className="heatmap-price">{price.toFixed(2)}</span>
      <span className="heatmap-qty">{qty.toFixed(4)}</span>
      <span className="heatmap-cumulative">{cumulative.toFixed(2)}</span>
      <div
        className="heatmap-bar"
        style={{
          width: `${intensity}%`,
          background:
            side === 'bid'
              ? `rgba(34, 197, 94, ${0.2 + intensity / 200})`
              : `rgba(239, 68, 68, ${0.2 + intensity / 200})`,
        }}
      />
    </div>
  );
});

function calculateCumulative(levels, currentIdx, side) {
  // 计算累计金额（价格 × 数量）
  let cumulative = 0;

  if (side === 'ask') {
    // 卖单：从最低价累计到当前价
    for (let i = 0; i <= currentIdx; i++) {
      const [price, qty] = levels[i];
      cumulative += Number(price) * Number(qty);
    }
  } else {
    // 买单：从最高价累计到当前价
    for (let i = 0; i <= currentIdx; i++) {
      const [price, qty] = levels[i];
      cumulative += Number(price) * Number(qty);
    }
  }

  return cumulative;
}
