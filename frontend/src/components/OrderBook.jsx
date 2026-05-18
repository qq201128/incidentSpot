import { useEffect, useMemo, useRef, useState } from "react";
import { fetchOrderbookDepth } from "../api/client";

const POLL_MS = 1500;
/** 侧边栏列出数量严格大于该值的档位，便于快速识别相对大单 */
const HEAVY_QTY_THRESHOLD = 1;
const STORAGE_VIEW = "incidentSpot:orderbookView";
/** 与后端 / Binance depth 一致，每侧拉取档位数（界面不再提供修改） */
const DEPTH_LEVELS_PER_SIDE = 1000;
/** 列表仅展示距价差最近的若干档（压低高度，便于首屏露出最新成交） */
const DISPLAY_ROWS_PER_SIDE = 5;
/** 前端合并精度（无下拉，固定 0.1） */
const MERGE_TICK = 0.1;

/** @returns {'both' | 'bids' | 'asks'} */
function readStoredView() {
  try {
    const raw = localStorage.getItem(STORAGE_VIEW);
    if (raw === "bids" || raw === "asks" || raw === "both") return raw;
  } catch (err) {
    console.error("订单簿视图读取失败", err);
  }
  return "both";
}

/**
 * 按精度合并档位（仅展示层；asks 向上取整档、bids 向下取整档）。
 * @param {'ask' | 'bid'} side
 * @param {[number, number][]} rows
 * @param {number} tick
 */
function aggregateByTick(side, rows, tick) {
  if (!tick || tick <= 0 || !rows.length) return rows.map((r) => [Number(r[0]), Number(r[1])]);
  const m = new Map();
  for (const [p, q] of rows) {
    const price = Number(p);
    const qty = Number(q);
    if (!Number.isFinite(price) || !Number.isFinite(qty) || qty <= 0) continue;
    const bucket =
      side === "ask"
        ? Math.ceil((price / tick) - 1e-12) * tick
        : Math.floor((price / tick) + 1e-12) * tick;
    const k = Number(bucket.toPrecision(12));
    m.set(k, (m.get(k) || 0) + qty);
  }
  const sorted = [...m.entries()].sort((a, b) => (side === "ask" ? a[0] - b[0] : b[0] - a[0]));
  return sorted.map(([p, q]) => [p, q]);
}

/**
 * @typedef {{ bids: [number, number][], asks: [number, number][], bestBid: number, bestAsk: number, spread: number, spreadBps: number, timestamp?: number }} DepthPayload
 */

export default function OrderBook({ symbol, lastTrade = null }) {
  const [viewMode, setViewMode] = useState(readStoredView);
  const [depth, setDepth] = useState(/** @type {DepthPayload | null} */ (null));
  const [error, setError] = useState("");

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_VIEW, viewMode);
    } catch (err) {
      console.error("订单簿视图保存失败", err);
    }
  }, [viewMode]);

  useEffect(() => {
    let cancelled = false;

    let timer;

    async function tick() {
      try {
        const data = await fetchOrderbookDepth(symbol, DEPTH_LEVELS_PER_SIDE);
        if (!cancelled) {
          setDepth(data);
          setError("");
        }
      } catch (e) {
        if (!cancelled) {
          setError(String(e?.response?.data?.detail || e?.message || "加载失败"));
        }
      } finally {
        if (!cancelled) {
          timer = window.setTimeout(tick, POLL_MS);
        }
      }
    }

    void tick();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [symbol]);

  const { dispAsks, dispBids, cumAsks, cumBids, maxAskCum, maxBidCum } = useMemo(() => {
    const rawAsks = depth?.asks || [];
    const rawBids = depth?.bids || [];
    const asksFull = aggregateByTick("ask", rawAsks, MERGE_TICK);
    const bidsFull = aggregateByTick("bid", rawBids, MERGE_TICK);
    /** asks 升序、 bids 降序：前若干档即离价差最近的一侧档位 */
    const asks = asksFull.slice(0, DISPLAY_ROWS_PER_SIDE);
    const bids = bidsFull.slice(0, DISPLAY_ROWS_PER_SIDE);
    const cum = (/** @type {[number, number][]} */ rows) => {
      let s = 0;
      return rows.map(([p, q]) => {
        s += Number(p) * Number(q);
        return s;
      });
    };
    const ca = cum(asks);
    const cb = cum(bids);
    return {
      dispAsks: asks,
      dispBids: bids,
      cumAsks: ca,
      cumBids: cb,
      maxAskCum: ca.length ? Math.max(...ca) : 0,
      maxBidCum: cb.length ? Math.max(...cb) : 0,
    };
  }, [depth]);

  const {
    heavyAsks,
    heavyBids,
    heavyAskQtySum,
    heavyBidQtySum,
    heavyAskNotionalSum,
    heavyBidNotionalSum,
  } = useMemo(() => {
    const t = HEAVY_QTY_THRESHOLD;
    const asks = depth?.asks || [];
    const bids = depth?.bids || [];
    const heavyAsks = asks.filter(([, q]) => Number(q) > t);
    const heavyBids = bids.filter(([, q]) => Number(q) > t);
    const sumQty = (/** @type {[number, number][]} */ rows) =>
      rows.reduce((s, [, q]) => s + Number(q), 0);
    const sumNotional = (/** @type {[number, number][]} */ rows) =>
      rows.reduce((s, [p, q]) => s + Number(p) * Number(q), 0);
    return {
      heavyAsks,
      heavyBids,
      heavyAskQtySum: sumQty(heavyAsks),
      heavyBidQtySum: sumQty(heavyBids),
      heavyAskNotionalSum: sumNotional(heavyAsks),
      heavyBidNotionalSum: sumNotional(heavyBids),
    };
  }, [depth]);

  /** 当前拉取的完整深度两侧名义 USDT 合计（便于对比挂单多空力量） */
  const fullDepthTotals = useMemo(() => {
    const asks = depth?.asks || [];
    const bids = depth?.bids || [];
    const sumQuote = (/** @type {[number, number][]} */ rows) =>
      rows.reduce((s, [p, q]) => {
        const pq = Number(p) * Number(q);
        return s + (Number.isFinite(pq) ? pq : 0);
      }, 0);
    const askQuote = sumQuote(asks);
    const bidQuote = sumQuote(bids);
    const sum = askQuote + bidQuote;
    const bidShare = sum > 0 ? (bidQuote / sum) * 100 : 50;
    return { bidQuote, askQuote, bidShare, askShare: 100 - bidShare };
  }, [depth]);

  if (error && !depth) {
    return (
      <div className="orderbook-panel orderbook-panel--error" aria-live="polite">
        <div className="orderbook-head">
          <span className="orderbook-title">订单簿</span>
        </div>
        <p className="orderbook-error">{error}</p>
      </div>
    );
  }

  if (!depth) {
    return (
      <div className="orderbook-panel">
        <div className="orderbook-head">
          <span className="orderbook-title">订单簿</span>
        </div>
        <p className="orderbook-muted">加载中…</p>
      </div>
    );
  }

  const asksDisplay = [...dispAsks].reverse();
  const nAsk = dispAsks.length;

  return (
    <div className="orderbook-panel orderbook-panel--compact">
      <div className="orderbook-head-shot orderbook-head-shot--single" role="toolbar" aria-label="订单簿">
        <span className="orderbook-title">订单簿</span>
        <div className="orderbook-view-toggles">
          <button
            type="button"
            className={`orderbook-toggle${viewMode === "both" ? " orderbook-toggle--active" : ""}`}
            onClick={() => setViewMode("both")}
            aria-pressed={viewMode === "both"}
            title="买卖同列"
          >
            <span className="orderbook-toggle-i orderbook-toggle-i--both" aria-hidden />
            买卖
          </button>
          <button
            type="button"
            className={`orderbook-toggle${viewMode === "bids" ? " orderbook-toggle--active" : ""}`}
            onClick={() => setViewMode("bids")}
            aria-pressed={viewMode === "bids"}
            title="仅买盘"
          >
            <span className="orderbook-toggle-i orderbook-toggle-i--bid" aria-hidden />
            买
          </button>
          <button
            type="button"
            className={`orderbook-toggle${viewMode === "asks" ? " orderbook-toggle--active" : ""}`}
            onClick={() => setViewMode("asks")}
            aria-pressed={viewMode === "asks"}
            title="仅卖盘"
          >
            <span className="orderbook-toggle-i orderbook-toggle-i--ask" aria-hidden />
            卖
          </button>
        </div>
      </div>

      <p className="orderbook-spread orderbook-spread--compact">
        价差 {fmtPrice(depth.spread, depth.bestAsk)} · {Number(depth.spreadBps || 0).toFixed(2)} bps
      </p>

      <div
        className="orderbook-full-totals"
        aria-label="订单簿全档名义合计"
        title={`基于当前 depth 拉取的买盘 / 卖盘档位汇总（名义 USDT，各至多 ${DEPTH_LEVELS_PER_SIDE} 档）`}
      >
        <div className="orderbook-full-totals-row">
          <span className="orderbook-full-totals-side orderbook-full-totals-side--bid">
            买 Σ <strong>{fmtTotalCell(fullDepthTotals.bidQuote)}</strong>
          </span>
          <span className="orderbook-full-totals-vs" aria-hidden>
            /
          </span>
          <span className="orderbook-full-totals-side orderbook-full-totals-side--ask">
            卖 Σ <strong>{fmtTotalCell(fullDepthTotals.askQuote)}</strong>
          </span>
        </div>
        <div
          className="orderbook-full-totals-bar"
          role="img"
          aria-label={`买盘名义占比 ${fullDepthTotals.bidShare.toFixed(1)}%`}
        >
          <span
            className="orderbook-full-totals-bar-bid"
            style={{ width: `${fullDepthTotals.bidShare}%` }}
          />
          <span
            className="orderbook-full-totals-bar-ask"
            style={{ width: `${fullDepthTotals.askShare}%` }}
          />
        </div>
        <p className="orderbook-full-totals-hint">
          买 {fullDepthTotals.bidShare.toFixed(1)}% · 卖 {fullDepthTotals.askShare.toFixed(1)}%
        </p>
      </div>

      <div className="orderbook-table orderbook-ladder" role="grid" aria-label={`${symbol} 订单簿`}>
        <div className="orderbook-row orderbook-row--header">
          <span className="orderbook-col-price">价格 (USDT)</span>
          <span className="orderbook-col-qty">数量 (USDT)</span>
          <span className="orderbook-col-total">合计 (USDT)</span>
        </div>

        {viewMode !== "bids" ? (
          <div className="orderbook-side orderbook-side--asks" aria-label="卖盘">
            {asksDisplay.map((row, i) => (
              <OrderRow
                key={`ask-${i}-${row[0]}`}
                side="ask"
                price={row[0]}
                qty={row[1]}
                rowTotal={nAsk ? cumAsks[nAsk - 1 - i] : 0}
                maxTotal={maxAskCum}
                refPrice={depth.bestAsk}
              />
            ))}
          </div>
        ) : null}

        <MidStrip bestBid={depth.bestBid} bestAsk={depth.bestAsk} lastTrade={lastTrade} />

        {viewMode !== "asks" ? (
          <div className="orderbook-side orderbook-side--bids" aria-label="买盘">
            {dispBids.map((row, i) => (
              <OrderRow
                key={`bid-${i}-${row[0]}`}
                side="bid"
                price={row[0]}
                qty={row[1]}
                rowTotal={cumBids[i] ?? 0}
                maxTotal={maxBidCum}
                refPrice={depth.bestAsk}
              />
            ))}
          </div>
        ) : null}

        <aside
          className="orderbook-heavy orderbook-heavy--footer orderbook-heavy--screenshot-hide"
          aria-hidden="true"
          aria-label={`数量大于 ${HEAVY_QTY_THRESHOLD} 的档位`}
        >
          <div className="orderbook-heavy-title">量 &gt; {HEAVY_QTY_THRESHOLD}</div>
          <div className="orderbook-heavy-columns">
            <div className="orderbook-heavy-block" aria-label="卖盘大单">
              <div className="orderbook-heavy-side orderbook-heavy-side--ask">卖</div>
              {heavyAsks.length === 0 ? (
                <p className="orderbook-heavy-empty">—</p>
              ) : (
                heavyAsks.map((row, i) => (
                  <div key={`h-ask-${i}`} className="orderbook-heavy-row">
                    <span className="orderbook-heavy-price orderbook-price--ask">
                      {fmtPriceDisplay(row[0], depth.bestAsk)}
                    </span>
                    <span className="orderbook-heavy-qty">{fmtQty(row[1])}</span>
                  </div>
                ))
              )}
              <div className="orderbook-heavy-totals" aria-label="卖盘合计">
                <div className="orderbook-heavy-total-line">
                  <span className="orderbook-heavy-total-label">合计数量</span>
                  <span className="orderbook-heavy-total-value">{fmtQty(heavyAskQtySum)}</span>
                </div>
                <div className="orderbook-heavy-total-line">
                  <span className="orderbook-heavy-total-label">合计名义</span>
                  <span className="orderbook-heavy-total-value">{fmtNotionalUsdt(heavyAskNotionalSum)}</span>
                </div>
              </div>
            </div>
            <div className="orderbook-heavy-block" aria-label="买盘大单">
              <div className="orderbook-heavy-side orderbook-heavy-side--bid">买</div>
              {heavyBids.length === 0 ? (
                <p className="orderbook-heavy-empty">—</p>
              ) : (
                heavyBids.map((row, i) => (
                  <div key={`h-bid-${i}`} className="orderbook-heavy-row">
                    <span className="orderbook-heavy-price orderbook-price--bid">
                      {fmtPriceDisplay(row[0], depth.bestAsk)}
                    </span>
                    <span className="orderbook-heavy-qty">{fmtQty(row[1])}</span>
                  </div>
                ))
              )}
              <div className="orderbook-heavy-totals" aria-label="买盘合计">
                <div className="orderbook-heavy-total-line">
                  <span className="orderbook-heavy-total-label">合计数量</span>
                  <span className="orderbook-heavy-total-value">{fmtQty(heavyBidQtySum)}</span>
                </div>
                <div className="orderbook-heavy-total-line">
                  <span className="orderbook-heavy-total-label">合计名义</span>
                  <span className="orderbook-heavy-total-value">{fmtNotionalUsdt(heavyBidNotionalSum)}</span>
                </div>
              </div>
            </div>
          </div>
        </aside>
      </div>
      {error ? <p className="orderbook-warn">{error}</p> : null}
    </div>
  );
}

function MidStrip({ bestBid, bestAsk, lastTrade }) {
  const prevMid = useRef(null);
  const bb = Number(bestBid);
  const ba = Number(bestAsk);
  const lastPx =
    lastTrade && Number(lastTrade.price) > 0 && Number.isFinite(Number(lastTrade.price))
      ? Number(lastTrade.price)
      : null;
  const mid =
    lastPx != null
      ? lastPx
      : Number.isFinite(bb) && Number.isFinite(ba)
        ? (bb + ba) / 2
        : NaN;
  const trendUp =
    prevMid.current !== null && Number.isFinite(mid) ? mid >= prevMid.current : true;
  useEffect(() => {
    if (Number.isFinite(mid)) prevMid.current = mid;
  }, [mid]);

  const primaryNum = lastPx != null ? lastPx : bb;
  const secondaryNum = ba;
  const primarySell = lastTrade?.side === "sell";

  return (
    <div className="orderbook-mid" aria-label="最新成交与最优卖价">
      <span
        className={`orderbook-mid-arrow ${trendUp ? "orderbook-mid-arrow--up" : "orderbook-mid-arrow--down"}`}
        aria-hidden
      >
        {trendUp ? "▲" : "▼"}
      </span>
      <span
        className={`orderbook-mid-primary${primarySell ? " orderbook-mid-primary--sell" : ""}`}
      >
        {fmtPriceDisplay(primaryNum, ba)}
      </span>
      <span className="orderbook-mid-secondary">{fmtPriceDisplay(secondaryNum, ba)}</span>
    </div>
  );
}

function OrderRow({ side, price, qty, rowTotal, maxTotal, refPrice }) {
  const pct = maxTotal > 0 ? Math.min(100, (rowTotal / maxTotal) * 100) : 0;
  const ref = Number(refPrice);
  const r = Number.isFinite(ref) ? ref : price;
  const rowUsdt = Number(price) * Number(qty);
  return (
    <div
      className={`orderbook-row orderbook-row--${side}`}
      style={{ ["--depth-pct"]: `${pct}%` }}
    >
      <span className="orderbook-depth" aria-hidden />
      <span className={`orderbook-price orderbook-price--${side}`}>{fmtPriceDisplay(price, r)}</span>
      <span className="orderbook-qty">{fmtTotalCell(rowUsdt)}</span>
      <span className="orderbook-total">{fmtTotalCell(rowTotal)}</span>
    </div>
  );
}

function fmtPriceDisplay(value, ref) {
  const s = fmtPrice(value, ref);
  if (s === "—") return s;
  const parts = s.split(".");
  const intPart = parts[0] ?? "";
  const dec = parts[1];
  const withSep = intPart.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return dec !== undefined ? `${withSep}.${dec}` : withSep;
}

function fmtPrice(value, ref) {
  const v = Number(value);
  const r = Number(ref);
  const decimals =
    !Number.isFinite(v) || v <= 0
      ? 4
      : v >= 1000 || (Number.isFinite(r) && r >= 1000)
        ? 2
        : v >= 1
          ? 4
          : 6;
  return Number.isFinite(v) ? v.toFixed(decimals) : "—";
}

function fmtQty(q) {
  const v = Number(q);
  if (!Number.isFinite(v)) return "—";
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(2)}K`;
  if (v >= 100) return v.toFixed(1);
  return v.toFixed(v >= 1 ? 3 : 4);
}

/** 累计名义（USDT），表格内简短显示 */
function fmtTotalCell(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  if (v >= 1_000_000_000) return `${(v / 1_000_000_000).toFixed(2)}B`;
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(2)}K`;
  return v.toFixed(2);
}

/** 名义总额（USDT），Σ(价格×数量) */
function fmtNotionalUsdt(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  if (v >= 1_000_000_000) return `${(v / 1_000_000_000).toFixed(2)}B USDT`;
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M USDT`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(2)}K USDT`;
  return `${v.toFixed(2)} USDT`;
}
