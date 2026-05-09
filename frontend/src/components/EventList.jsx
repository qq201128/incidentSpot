import { useEffect, useMemo, useState } from "react";
import { settledExpectedProfitUsdt } from "../utils/eventSettlement";
import { strategyLabel } from "../utils/strategyLabels";

const PAGE_SIZE = 6;
const FILTER_ALL = "";

export default function EventList({ events, onSettle }) {
  const [eventPage, setEventPage] = useState(1);
  const [strategyFilter, setStrategyFilter] = useState(FILTER_ALL);

  const strategyOptions = useMemo(() => _distinctStrategyKeys(events), [events]);
  const filteredEvents = useMemo(() => {
    if (!strategyFilter) return events;
    return events.filter((e) => (e.strategyKey || "manual") === strategyFilter);
  }, [events, strategyFilter]);

  const totalPages = Math.max(1, Math.ceil(filteredEvents.length / PAGE_SIZE));
  const pageStart = (eventPage - 1) * PAGE_SIZE;
  const pagedEvents = filteredEvents.slice(pageStart, pageStart + PAGE_SIZE);

  useEffect(() => {
    setEventPage((prev) => Math.min(prev, totalPages));
  }, [totalPages]);

  useEffect(() => {
    setEventPage(1);
  }, [strategyFilter]);

  return (
    <div className="card">
      <h3>持仓事件</h3>
      <div className="event-strategy-filter">
        <label htmlFor="event-strategy-select">策略</label>
        <select
          id="event-strategy-select"
          value={strategyFilter}
          onChange={(e) => setStrategyFilter(e.target.value)}
        >
          <option value={FILTER_ALL}>全部策略</option>
          {strategyOptions.map((key) => (
            <option key={key} value={key}>
              {strategyLabel(key)}
            </option>
          ))}
        </select>
      </div>
      <EventPager eventPage={eventPage} totalPages={totalPages} setEventPage={setEventPage} />
      <ul className="event-list">
        {pagedEvents.map((item) => (
          <EventItem key={item.id} item={item} onSettle={onSettle} />
        ))}
        {!filteredEvents.length && <li>{events.length ? "该策略下暂无事件" : "暂无事件"}</li>}
      </ul>
    </div>
  );
}

function EventPager({ eventPage, totalPages, setEventPage }) {
  return (
    <div className="event-pagination">
      <span>
        第 {eventPage}/{totalPages} 页
      </span>
      <div className="event-pagination-actions">
        <button
          type="button"
          className="pager-btn"
          onClick={() => setEventPage((page) => Math.max(1, page - 1))}
          disabled={eventPage <= 1}
        >
          上一页
        </button>
        <button
          type="button"
          className="pager-btn"
          onClick={() => setEventPage((page) => Math.min(totalPages, page + 1))}
          disabled={eventPage >= totalPages}
        >
          下一页
        </button>
      </div>
    </div>
  );
}

function EventItem({ item, onSettle }) {
  return (
    <li>
      <div>
        <EventHeader item={item} />
        <AiEventMeta item={item} />
        <OrderGrid item={item} />
        <SettlementNote item={item} />
      </div>
      <button
        className="settle-btn"
        onClick={() => onSettle(item.id)}
        disabled={!item.orderSide || item.status === "SETTLED"}
        title={!item.orderSide ? "该事件还没有下单，无法结算" : undefined}
      >
        结算
      </button>
    </li>
  );
}

function EventHeader({ item }) {
  return (
    <div className="order-card-head">
      <div className="order-pair-row">
        <strong>{item.symbol.replace("USDT", "/USDT")}</strong>
        <span className="strategy-tag">{strategyLabel(item.strategyKey)}</span>
        <span className={sideTagClass(item.orderSide)}>{sideLabel(item.orderSide)}</span>
        <span className={statusTagClass(item.status)}>{statusLabel(item.status)}</span>
      </div>
      <span className="order-time">
        开始 {formatEventTime(item.startTime)} / 到期 {formatEventTime(item.endTime)}
      </span>
    </div>
  );
}

function AiEventMeta({ item }) {
  const aiPct = aiConfidencePercent(item);
  if (!item.aiPredictedDirection) return null;
  return (
    <div className="ai-event-meta">
      <span>规则方向：{item.aiPredictedDirection === "up" ? "涨" : "跌"}</span>
      <span>置信度：{aiPct != null ? `${aiPct.toFixed(1).replace(/\.0$/, "")}%` : "--"}</span>
      {item.status === "SETTLED" && item.aiPredictionCorrect != null && (
        <span className={Number(item.aiPredictionCorrect) === 1 ? "value-up" : "value-down"}>
          规则{Number(item.aiPredictionCorrect) === 1 ? "命中" : "未中"}
        </span>
      )}
    </div>
  );
}

function OrderGrid({ item }) {
  const view = settlementView(item);
  const payout = view ? view.expectedReturn : 0;
  return (
    <div className="order-grid">
      <GridCell label="支付金额 ($)" value={signedMoney(payout)} className={payout >= 0 ? "value-up" : "value-down"} />
      <GridCell label="金额" value={Number(item.orderQty || 0).toFixed(2)} />
      <GridCell label="时长" value={eventDurationLabel(item)} />
      <GridCell label="回报率" value={`${Math.round(Number(item.orderPrice || 0) * 100)}%`} />
      <GridCell label="入场价 ($)" value={Number(item.strikeValue || 0).toFixed(2)} />
      <GridCell label="结算价 ($)" value={settlementPriceLabel(item)} />
    </div>
  );
}

function GridCell({ label, value, className }) {
  return (
    <div>
      <span>{label}</span>
      <strong className={className}>{value}</strong>
    </div>
  );
}

function SettlementNote({ item }) {
  const view = settlementView(item);
  if (!view) return null;
  return (
    <div className="settlement-note">
      <span>结算：{view.isCorrect ? "猜对" : "猜错"}</span>
      <span>盈亏：{signedMoney(view.expectedProfit)} USDT</span>
      <span>系统PNL：{signedMoney(view.totalPnl)} USDT</span>
    </div>
  );
}

function settlementView(item) {
  const qty = Number(item?.orderQty);
  const price = Number(item?.orderPrice);
  const totalPnl = Number(item?.totalPnl ?? 0);
  if (!Number.isFinite(qty) || qty <= 0 || !Number.isFinite(price) || price < 0) return null;
  if (item?.status !== "SETTLED") return null;
  const expectedProfit = settledExpectedProfitUsdt(item);
  if (expectedProfit == null) return null;
  const isCorrect = _isCorrectSettlement(item?.orderSide, item?.result);
  return {
    isCorrect,
    expectedReturn: isCorrect ? qty * (1 + price) : 0,
    expectedProfit,
    totalPnl,
  };
}

function _isCorrectSettlement(side, result) {
  return (side === "BUY" && result === "YES") || (side === "SELL" && result === "NO");
}

function aiConfidencePercent(item) {
  const p = Number(item?.aiProbabilityUp);
  if (!Number.isFinite(p)) return null;
  if (item?.orderSide === "BUY") return Math.round(p * 1000) / 10;
  if (item?.orderSide === "SELL") return Math.round((1 - p) * 1000) / 10;
  return Math.round(Math.max(p, 1 - p) * 1000) / 10;
}

function eventDurationLabel(item) {
  if (!item?.startTime || !item?.endTime) return "--";
  const start = new Date(item.startTime).getTime();
  const end = new Date(item.endTime).getTime();
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return "--";
  return durationLabel(Math.round((end - start) / 60000));
}

function formatEventTime(value) {
  if (!value) return "--";
  const dt = new Date(value);
  if (!Number.isFinite(dt.getTime())) return value;
  return dt.toLocaleString("zh-CN", { hour12: false });
}

function settlementPriceLabel(item) {
  if (item.settlementPrice == null) return "--";
  return Number(item.settlementPrice).toFixed(2);
}

function signedMoney(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function durationLabel(minutes) {
  return minutes === 1440 ? "1天" : `${minutes}分钟`;
}

function statusLabel(status) {
  return { OPEN: "进行中", SETTLED: "已结算" }[status] || status;
}

function _distinctStrategyKeys(events) {
  const set = new Set();
  for (const e of events) {
    set.add(e.strategyKey || "manual");
  }
  return [...set].sort();
}

function sideLabel(side) {
  if (side === "BUY") return "看涨";
  if (side === "SELL") return "看跌";
  return "未下单";
}

function sideTagClass(side) {
  if (side === "BUY") return "side-tag up";
  if (side === "SELL") return "side-tag down";
  return "side-tag neutral";
}

function statusTagClass(status) {
  if (status === "SETTLED") return "status-tag settled";
  return "status-tag open";
}
