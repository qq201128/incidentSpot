import { useCallback, useEffect, useState } from "react";
import { fetchEventsPage } from "../api/eventsClient";
import { eventDurationMinutesFromWindow } from "../utils/eventDuration";
import { formatPnlU, settledExpectedProfitUsdt } from "../utils/eventSettlement";
import "./StrategyRecentEventsPanel.css";

const PAGE_SIZE = 12;

export default function StrategyRecentEventsPanel({
  symbol,
  strategyKey,
  title,
  englishName,
  durationMinutes,
  durationHeading,
  onClose,
}) {
  const [page, setPage] = useState(1);
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [pageCount, setPageCount] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadRecords = useCallback(async () => {
    if (!symbol || !strategyKey) return;
    setLoading(true);
    setError("");
    try {
      const data = await fetchEventsPage({
        symbol,
        strategyKey,
        durationMinutes,
        page,
        pageSize: PAGE_SIZE,
      });
      setItems(Array.isArray(data.items) ? data.items : []);
      setTotal(Number(data.total) || 0);
      setPageCount(Math.max(1, Number(data.pageCount) || 1));
    } catch (err) {
      console.error("策略事件记录加载失败", err);
      setItems([]);
      setTotal(0);
      setPageCount(1);
      setError(err.message || "加载失败");
    } finally {
      setLoading(false);
    }
  }, [durationMinutes, page, strategyKey, symbol]);

  useEffect(() => {
    setPage(1);
  }, [durationMinutes, strategyKey, symbol]);

  useEffect(() => {
    void loadRecords();
  }, [loadRecords]);

  if (!strategyKey) return null;

  const periodLabel = durationHeading || "—";
  const countLabel = loading ? "同步中…" : `共 ${total} 条`;
  const scopeKicker = `${periodLabel} · ${countLabel}`;

  return (
    <aside className="strategy-recent-events" role="dialog" aria-label={`${title} 最近合约记录`}>
      <header className="strategy-recent-events-head">
        <div>
          <span className="strategy-recent-events-kicker">{scopeKicker}</span>
          <h3>{title}</h3>
          {englishName ? (
            <code className="strategy-recent-events-code" title="英文/字段名">
              {englishName}
            </code>
          ) : null}
        </div>
        <button type="button" className="strategy-recent-events-close" onClick={onClose} aria-label="关闭">
          ×
        </button>
      </header>
      {error ? <p className="strategy-recent-events-error">{error}</p> : null}
      <div className="strategy-recent-events-scroll">
        {items.map((event) => (
          <RecordCard key={event.id} event={event} />
        ))}
        {!loading && !items.length ? (
          <p className="strategy-recent-events-empty">暂无合约记录</p>
        ) : null}
        {loading && !items.length ? (
          <p className="strategy-recent-events-empty">加载中…</p>
        ) : null}
      </div>
      <footer className="strategy-recent-events-foot">
        <span className="strategy-recent-events-foot-scope">{scopeKicker}</span>
        <div className="strategy-recent-events-pager">
          <button type="button" disabled={page <= 1 || loading} onClick={() => setPage((p) => p - 1)}>
            上一页
          </button>
          <strong>
            {page}/{pageCount}
          </strong>
          <button type="button" disabled={page >= pageCount || loading} onClick={() => setPage((p) => p + 1)}>
            下一页
          </button>
        </div>
      </footer>
    </aside>
  );
}

function RecordCard({ event }) {
  const pnl = settledExpectedProfitUsdt(event);
  const settled = event.status === "SETTLED";
  const win = settled && Number(event.aiPredictionCorrect) === 1;
  const lose = settled && event.aiPredictionCorrect != null && !win;

  return (
    <article className="sre-record">
      <div className="sre-record-top">
        <span className={`sre-side ${sideToneClass(event.orderSide)}`}>{sideLabel(event.orderSide)}</span>
        <span className={`sre-hit${win ? " is-win" : lose ? " is-lose" : ""}`}>
          {!settled ? statusLabel(event) : win ? "命中" : lose ? "未中" : "—"}
        </span>
      </div>
      <div className="sre-record-grid">
        <div className="sre-cell">
          <small>开仓价格</small>
          <strong>{formatPrice(event.strikeValue)}</strong>
        </div>
        <div className="sre-cell">
          <small>结束价格</small>
          <strong>{formatSettlementPrice(event)}</strong>
        </div>
        <div className="sre-cell">
          <small>PnL</small>
          <strong className={pnl == null ? "" : pnl >= 0 ? "value-up" : "value-down"}>
            {pnl == null ? "—" : formatPnlU(pnl)}
          </strong>
        </div>
        <div className="sre-cell">
          <small>周期</small>
          <strong>{durationLabel(event)}</strong>
        </div>
        <div className="sre-cell sre-cell--wide">
          <small>开仓时间</small>
          <strong>{formatTime(event.startTime)}</strong>
        </div>
      </div>
    </article>
  );
}

function sideToneClass(side) {
  if (side === "BUY") return "is-up";
  if (side === "SELL") return "is-down";
  return "is-flat";
}

function sideLabel(side) {
  if (side === "BUY") return "看涨";
  if (side === "SELL") return "看跌";
  return "未下单";
}

function statusLabel(event) {
  if (event.status === "SETTLED") return "已结算";
  if (event.externalStatus === "SIMULATED") return "模拟中";
  return event.status || "—";
}

function durationLabel(event) {
  const minutes = eventDurationMinutesFromWindow(event);
  if (minutes == null) return "—";
  return minutes === 1440 ? "1天" : `${minutes}分钟`;
}

function formatPrice(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—";
}

function formatSettlementPrice(event) {
  if (event.status !== "SETTLED" || event.settlementPrice == null) return "—";
  return formatPrice(event.settlementPrice);
}

function formatTime(value) {
  if (!value) return "—";
  const dt = new Date(value);
  if (!Number.isFinite(dt.getTime())) return "—";
  return dt.toLocaleString("zh-CN", {
    hour12: false,
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}
