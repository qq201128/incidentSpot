import { useEffect, useMemo, useState } from "react";
import { eventDurationMinutesFromWindow } from "../utils/eventDuration";
import { settledExpectedProfitUsdt } from "../utils/eventSettlement";
import { strategyLabel } from "../utils/strategyLabels";
import EnsembleRankingTable from "./EnsembleRankingTable";

const PAGE_SIZE = 8;
const EVENT_TAB = "events";
const ENSEMBLE_TAB = "ensemble-ranking";
const TABS = Object.freeze([
  { key: EVENT_TAB, label: "事件合约记录" },
  { key: "orders", label: "订单记录" },
  { key: "settlements", label: "结算验证" },
  { key: "failures", label: "失败原因" },
  { key: ENSEMBLE_TAB, label: "候选信号排名" },
]);

export default function EventRecordsTable({
  events,
  compact = false,
  symbol = "",
  ensembleDuration = "10m",
  ensembleReloadKey = 0,
}) {
  const [activeTab, setActiveTab] = useState(EVENT_TAB);
  const [page, setPage] = useState(1);
  const showEnsemble = !compact && activeTab === ENSEMBLE_TAB;
  const view = useMemo(
    () => (showEnsemble ? null : buildView(events, compact ? EVENT_TAB : activeTab, compact)),
    [activeTab, compact, events, showEnsemble],
  );

  const total = view?.rows.length ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));

  useEffect(() => {
    setPage(1);
  }, [activeTab, compact, events]);

  useEffect(() => {
    if (page > pageCount) setPage(pageCount);
  }, [page, pageCount]);

  const pageRows = useMemo(() => {
    if (!view) return [];
    const start = (page - 1) * PAGE_SIZE;
    return view.rows.slice(start, start + PAGE_SIZE);
  }, [page, view]);

  const sectionClass = compact
    ? "event-records event-records--compact"
    : showEnsemble
      ? "event-records event-records--ensemble"
      : "event-records";

  return (
    <section className={sectionClass}>
      <RecordTabs
        activeTab={activeTab}
        compact={compact}
        ensembleTab={showEnsemble}
        onChange={setActiveTab}
        page={page}
        pageCount={pageCount}
        total={total}
      />
      {showEnsemble ? (
        <EnsembleRankingTable
          symbol={symbol}
          duration={ensembleDuration}
          reloadKey={ensembleReloadKey}
        />
      ) : (
        <>
          <div className="event-records-table" role="table" aria-label={view.title}>
            <div className="event-records-rows">
              <RecordHeader labels={view.labels} viewKey={view.key} />
              {pageRows.map((row) => (
                <RecordRow key={row.key} cells={row.cells} viewKey={view.key} />
              ))}
            </div>
            {!total ? <p className="event-records-empty">{view.emptyText}</p> : null}
          </div>
          {total ? (
            <RecordsPagination page={page} pageCount={pageCount} total={total} onPageChange={setPage} />
          ) : null}
        </>
      )}
    </section>
  );
}

function RecordTabs({ activeTab, compact, ensembleTab, onChange, page, pageCount, total }) {
  const meta = ensembleTab
    ? "综合裁判候选列表"
    : total
      ? `共 ${total} 条 · ${page}/${pageCount} 页`
      : "暂无记录";
  if (compact) {
    return (
      <div className="event-records-head">
        <h2>事件列表</h2>
        <span>{meta}</span>
      </div>
    );
  }
  return (
    <div className="event-records-head" role="tablist" aria-label="事件记录视图">
      <div className="event-records-tabs">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.key}
            className={`event-records-tab${activeTab === tab.key ? " is-active" : ""}`}
            onClick={() => onChange(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <span className="event-records-meta">{meta}</span>
    </div>
  );
}

function RecordsPagination({ page, pageCount, total, onPageChange }) {
  return (
    <nav className="event-records-pagination" aria-label="事件记录分页">
      <span className="event-records-page-total">共 {total} 条</span>
      <div className="event-records-page-actions">
        <button type="button" disabled={page <= 1} onClick={() => onPageChange(1)} aria-label="首页">
          «
        </button>
        <button type="button" disabled={page <= 1} onClick={() => onPageChange(page - 1)} aria-label="上一页">
          ‹
        </button>
        <strong>
          {page} / {pageCount}
        </strong>
        <button
          type="button"
          disabled={page >= pageCount}
          onClick={() => onPageChange(page + 1)}
          aria-label="下一页"
        >
          ›
        </button>
        <button
          type="button"
          disabled={page >= pageCount}
          onClick={() => onPageChange(pageCount)}
          aria-label="末页"
        >
          »
        </button>
      </div>
    </nav>
  );
}

function RecordHeader({ labels, viewKey }) {
  return (
    <div className={`event-records-row event-records-row--header event-records-row--${viewKey}`} role="row">
      {labels.map((label) => (
        <span key={label} role="columnheader" title={label}>
          {label}
        </span>
      ))}
    </div>
  );
}

function RecordRow({ cells, viewKey }) {
  return (
    <div className={`event-records-row event-records-row--${viewKey}`} role="row">
      {cells.map((cell) => (
        <span key={cell.key} className={cell.className || ""} role="cell" title={cell.title || String(cell.value)}>
          {cell.value}
        </span>
      ))}
    </div>
  );
}

function buildView(events, activeTab, compact) {
  const source = Array.isArray(events) ? events : [];
  if (compact) {
    return view("compact", compactLabels(), source.map(compactRow), "暂无事件", "事件列表");
  }
  if (activeTab === "orders") return ordersView(source);
  if (activeTab === "settlements") return settlementsView(source);
  if (activeTab === "failures") return failuresView(source);
  return view("events", eventLabels(), source.map(eventRow), "暂无事件", "事件合约记录");
}

function ordersView(events) {
  const rows = events.filter((event) => event.orderSide).map(orderRow);
  return view("orders", orderLabels(), rows, "暂无订单记录", "订单记录");
}

function settlementsView(events) {
  const rows = events
    .filter((event) => event.status === "SETTLED" || event.settlementPrice != null)
    .map(settlementRow);
  return view("settlements", settlementLabels(), rows, "暂无结算记录", "结算验证");
}

function failuresView(events) {
  const rows = events.filter(hasFailure).map(failureRow);
  return view("failures", failureLabels(), rows, "暂无失败记录", "失败原因");
}

function view(key, labels, rows, emptyText, title) {
  return { key, labels, rows, emptyText, title };
}

function eventRow(event) {
  return row(event.id, [
    text("id", eventCode(event.id)),
    text("symbol", event.symbol),
    sideCell(event.orderSide),
    text("start", timeHm(event.startTime)),
    text("strike", price(event.strikeValue)),
    text("duration", duration(event)),
    text("end", timeHm(event.endTime)),
    text("settle", nullablePrice(event.settlementPrice)),
    statusCell(event.status),
    modeCell(event.externalStatus),
    resultCell(event),
    pnlCell(event),
    confidenceCell(event),
    text("rule", ruleName(event)),
  ]);
}

function orderRow(event) {
  return row(event.id, [
    text("id", eventCode(event.id)),
    text("symbol", event.symbol),
    sideCell(event.orderSide),
    text("qty", amount(event.orderQty)),
    text("price", payout(event.orderPrice)),
    text("created", timeHm(event.orderCreatedAt)),
    text("orderStatus", event.orderStatus || "—"),
    modeCell(event.externalStatus),
    text("externalId", event.externalOrderId || "—"),
  ]);
}

function settlementRow(event) {
  return row(event.id, [
    text("id", eventCode(event.id)),
    sideCell(event.orderSide),
    text("strike", price(event.strikeValue)),
    text("settlement", nullablePrice(event.settlementPrice)),
    resultCell(event),
    pnlCell(event),
    text("quoteTime", quoteTime(event.settlementQuoteTime)),
    text("source", event.settlementSource || "—"),
  ]);
}

function failureRow(event) {
  const reason = failureReason(event);
  return row(event.id, [
    text("id", eventCode(event.id)),
    text("symbol", event.symbol),
    statusCell(event.status),
    text("orderStatus", event.orderStatus || "—"),
    modeCell(event.externalStatus),
    text("reason", reason, "", reason),
  ]);
}

function compactRow(event) {
  return row(event.id, [
    text("id", eventCode(event.id)),
    sideCell(event.orderSide),
    text("strike", price(event.strikeValue)),
    text("duration", duration(event)),
    statusCell(event.status),
    modeCell(event.externalStatus),
    resultCell(event),
  ]);
}

function row(key, cells) {
  return { key, cells };
}

function text(key, value, className = "", title = "") {
  return { key, value, className, title };
}

function sideCell(side) {
  if (side === "BUY") return text("side", "BUY", "value-up");
  if (side === "SELL") return text("side", "SELL", "value-down");
  return text("side", "—");
}

function statusCell(status) {
  const ok = status === "OPEN" || status === "SETTLED";
  return text("status", status || "—", ok ? "tag tag--green" : "tag tag--red");
}

function modeCell(externalStatus) {
  const live = externalStatus && externalStatus !== "SIMULATED";
  return text("mode", live ? "LIVE" : "SIM", live ? "tag tag--orange" : "tag tag--blue");
}

function resultCell(event) {
  if (event.status !== "SETTLED") return text("result", "—");
  const win = Number(event.aiPredictionCorrect) === 1;
  return text("result", win ? "WIN" : "LOSE", win ? "tag tag--green" : "tag tag--red");
}

function pnlCell(event) {
  const pnl = settledExpectedProfitUsdt(event);
  if (pnl == null) return text("pnl", "—");
  return text("pnl", signed(pnl), pnl >= 0 ? "value-up" : "value-down");
}

function confidenceCell(event) {
  const p = Number(event.aiProbabilityUp);
  if (!Number.isFinite(p)) return text("confidence", "—");
  return text("confidence", `${Math.round(Math.max(p, 1 - p) * 1000) / 10}%`);
}

function hasFailure(event) {
  const status = String(event.status || "").toUpperCase();
  const orderStatus = String(event.orderStatus || "").toUpperCase();
  const externalStatus = String(event.externalStatus || "").toUpperCase();
  return status === "FAILED" || orderStatus === "FAILED" || /FAIL|ERROR|REJECT/.test(externalStatus);
}

function failureReason(event) {
  return event.externalResponse || event.settlementSource || event.title || "—";
}

function eventCode(id) {
  return `EVT-${String(id).padStart(6, "0")}`;
}

function duration(event) {
  const minutes = eventDurationMinutesFromWindow(event);
  return minutes == null ? "—" : minutes === 1440 ? "1d" : `${minutes}m`;
}

function price(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—";
}

function nullablePrice(value) {
  return value == null ? "—" : price(value);
}

function amount(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(3).replace(/\.?0+$/, "") : "—";
}

function payout(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `${Math.round(n * 100)}%` : "—";
}

function timeHm(value) {
  if (!value) return "—";
  const dt = new Date(value);
  if (!Number.isFinite(dt.getTime())) return "—";
  return dt.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false });
}

function quoteTime(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return "—";
  return timeHm(n);
}

function signed(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function ruleName(event) {
  if (event.strategyKey) return strategyLabel(event.strategyKey);
  return event.title || "手动";
}

const eventLabels = () => [
  "事件ID",
  "交易对",
  "方向",
  "入场",
  "入场价",
  "周期",
  "结算",
  "结算价",
  "状态",
  "模式",
  "结果",
  "PnL",
  "置信",
  "规则",
];
const orderLabels = () => ["事件ID", "交易对", "方向", "数量", "支付率", "下单", "订单", "模式", "外部ID"];
const settlementLabels = () => ["事件ID", "方向", "入场价", "结算价", "结果", "PnL", "报价", "来源"];
const failureLabels = () => ["事件ID", "交易对", "事件", "订单", "模式", "原因"];
const compactLabels = () => ["事件ID", "方向", "入场价", "周期", "状态", "模式", "结果"];
