/**
 * 虚拟滚动交易列表组件
 */
import { useVirtualizer } from '@tanstack/react-virtual';
import { memo, useRef } from 'react';
import './TradesList.css';

const TradeRow = memo(function TradeRow({ trade }) {
  const isBuy = trade.isBuyerMaker === false;
  const priceClass = isBuy ? 'trade-price--buy' : 'trade-price--sell';

  return (
    <div className="trade-row">
      <span className={`trade-price ${priceClass}`}>{Number(trade.price).toFixed(2)}</span>
      <span className="trade-qty">{Number(trade.qty).toFixed(4)}</span>
      <span className="trade-time">{formatTime(trade.time)}</span>
    </div>
  );
});

export default function TradesList({ trades = [], height = 400 }) {
  const parentRef = useRef();

  const virtualizer = useVirtualizer({
    count: trades.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 32,
    overscan: 5,
  });

  const items = virtualizer.getVirtualItems();

  return (
    <div className="trades-list">
      <div className="trades-list-header">
        <span>价格</span>
        <span>数量</span>
        <span>时间</span>
      </div>

      <div
        ref={parentRef}
        className="trades-list-container"
        style={{ height: `${height}px`, overflow: 'auto' }}
      >
        <div
          style={{
            height: `${virtualizer.getTotalSize()}px`,
            width: '100%',
            position: 'relative',
          }}
        >
          {items.map((virtualRow) => {
            const trade = trades[virtualRow.index];

            return (
              <div
                key={virtualRow.key}
                data-index={virtualRow.index}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: `${virtualRow.size}px`,
                  transform: `translateY(${virtualRow.start}px)`,
                }}
              >
                <TradeRow trade={trade} />
              </div>
            );
          })}
        </div>
      </div>

      {trades.length === 0 && (
        <div className="trades-list-empty">暂无成交记录</div>
      )}
    </div>
  );
}

function formatTime(timestamp) {
  const date = new Date(timestamp);
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  const seconds = String(date.getSeconds()).padStart(2, '0');
  return `${hours}:${minutes}:${seconds}`;
}
