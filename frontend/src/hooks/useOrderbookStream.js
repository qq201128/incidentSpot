/**
 * 订单簿WebSocket流Hook
 */
import { useEffect, useState } from 'react';

const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

export function useOrderbookStream(symbol, depth = 5) {
  const [orderbook, setOrderbook] = useState(null);
  const [error, setError] = useState(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let ws = null;
    let reconnectTimer = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;

    function connect() {
      try {
        ws = new WebSocket(
          `${WS_BASE_URL}/ws/orderbook?symbol=${symbol.toLowerCase()}&depth=${depth}`
        );

        ws.onopen = () => {
          console.log(`Orderbook stream connected: ${symbol}`);
          setConnected(true);
          setError(null);
          reconnectAttempts = 0;
        };

        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);

            if (message.type === 'snapshot') {
              // 完整快照
              setOrderbook(message.data);
            } else if (message.type === 'update') {
              // 增量更新
              setOrderbook((prev) => {
                if (!prev) return message.data;
                return applyUpdate(prev, message.data);
              });
            }
          } catch (err) {
            console.error('Failed to parse orderbook message:', err);
          }
        };

        ws.onerror = (error) => {
          console.error('Orderbook stream error:', error);
          setError('连接失败');
        };

        ws.onclose = () => {
          console.log('Orderbook stream closed');
          setConnected(false);

          // 自动重连
          if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
            console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);

            reconnectTimer = setTimeout(connect, delay);
          } else {
            setError('连接失败，请刷新页面');
          }
        };
      } catch (err) {
        console.error('Failed to create WebSocket:', err);
        setError('无法建立连接');
      }
    }

    connect();

    return () => {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
      if (ws) {
        ws.close();
      }
    };
  }, [symbol, depth]);

  return { orderbook, error, connected };
}

function applyUpdate(current, update) {
  // 应用增量更新到订单簿
  const newOrderbook = { ...current };

  // 更新买单
  if (update.bids && update.bids.length > 0) {
    newOrderbook.bids = mergeDepthUpdate(current.bids, update.bids);
  }

  // 更新卖单
  if (update.asks && update.asks.length > 0) {
    newOrderbook.asks = mergeDepthUpdate(current.asks, update.asks);
  }

  // 更新时间戳
  if (update.timestamp) {
    newOrderbook.timestamp = update.timestamp;
  }

  // 重新计算最优价和价差
  if (newOrderbook.bids.length > 0) {
    newOrderbook.bestBid = newOrderbook.bids[0][0];
  }
  if (newOrderbook.asks.length > 0) {
    newOrderbook.bestAsk = newOrderbook.asks[0][0];
  }
  if (newOrderbook.bestBid && newOrderbook.bestAsk) {
    newOrderbook.spread = newOrderbook.bestAsk - newOrderbook.bestBid;
    const mid = (newOrderbook.bestBid + newOrderbook.bestAsk) / 2;
    newOrderbook.spreadBps = mid > 0 ? (newOrderbook.spread / mid) * 10000 : 0;
  }

  return newOrderbook;
}

function mergeDepthUpdate(current, updates) {
  // 将更新合并到当前深度
  const priceMap = new Map();

  // 添加当前档位
  for (const [price, qty] of current) {
    priceMap.set(price, qty);
  }

  // 应用更新
  for (const [price, qty] of updates) {
    if (qty === 0) {
      // 数量为0表示移除该档位
      priceMap.delete(price);
    } else {
      priceMap.set(price, qty);
    }
  }

  // 转换回数组并排序
  const result = Array.from(priceMap.entries());

  // 买单降序，卖单升序（由上游决定）
  // 这里假设已经是正确顺序，只需限制数量
  return result.slice(0, 20);
}
