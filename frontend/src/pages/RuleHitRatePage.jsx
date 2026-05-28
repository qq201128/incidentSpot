import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchAiHistoryMeta, fetchAiHistorySuccess } from "../api/workbenchClient";
import StrategyRecentEventsPanel from "../components/StrategyRecentEventsPanel";
import { contractDurationLabel } from "../utils/eventDuration";
import { formatPnlU } from "../utils/eventSettlement";
import { aiHistoryRowEnglishName, aiHistoryRowLabel } from "../utils/aiHistoryLabels";
import "./RuleHitRatePage.css";

const SYMBOLS = [
  { value: "BTCUSDT", label: "BTC / USDT" },
  { value: "ETHUSDT", label: "ETH / USDT" },
];
const POLL_MS = 5000;
const PAGE_SIZE = 10;
const UNKNOWN_DURATION = -1;
const DEFAULT_DURATIONS = [10, 30, 60, 1440];
const SKELETON_ROWS = 5;

const EMPTY_PERIOD = Object.freeze({
  total: 0,
  hits: 0,
  rate: null,
  pnlU: 0,
  factorCount: 0,
});

const EMPTY_PAGINATION = Object.freeze({
  page: 1,
  pageSize: PAGE_SIZE,
  total: 0,
  pageCount: 1,
});

function durationKeyFromMinutes(dm) {
  return dm === UNKNOWN_DURATION ? "unknown" : String(dm);
}

function durationHeading(dm) {
  return dm === UNKNOWN_DURATION ? contractDurationLabel(null) : contractDurationLabel(dm);
}

function buildDefaultTabs() {
  return DEFAULT_DURATIONS.map((durationMinutes) => ({
    durationMinutes,
    durationKey: durationKeyFromMinutes(durationMinutes),
    heading: durationHeading(durationMinutes),
    factorCount: null,
  }));
}

export default function RuleHitRatePage() {
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [durationSummaries, setDurationSummaries] = useState([]);
  const [periodStats, setPeriodStats] = useState(EMPTY_PERIOD);
  const [byStrategy, setByStrategy] = useState([]);
  const [pagination, setPagination] = useState(EMPTY_PAGINATION);
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [status, setStatus] = useState("");
  const [selectedRow, setSelectedRow] = useState(null);
  const [selectedDurationMinutes, setSelectedDurationMinutes] = useState(10);
  const [page, setPage] = useState(1);
  const abortRef = useRef(null);
  const hasLoadedRef = useRef(false);

  const durationTabs = useMemo(() => {
    if (!durationSummaries.length) return buildDefaultTabs();
    const byMinutes = new Map(buildDefaultTabs().map((tab) => [tab.durationMinutes, { ...tab }]));
    for (const item of durationSummaries) {
      const dm = item.durationMinutes;
      byMinutes.set(dm, {
        durationMinutes: dm,
        durationKey: durationKeyFromMinutes(dm),
        heading: durationHeading(dm),
        factorCount: item.factorCount,
      });
    }
    return [...byMinutes.values()].sort((a, b) => {
      const ak = a.durationMinutes === UNKNOWN_DURATION ? 1_000_000 : a.durationMinutes;
      const bk = b.durationMinutes === UNKNOWN_DURATION ? 1_000_000 : b.durationMinutes;
      return ak - bk;
    });
  }, [durationSummaries]);

  const queryDurationMinutes = selectedDurationMinutes ?? 10;

  const activeTab = useMemo(
    () => durationTabs.find((t) => t.durationMinutes === queryDurationMinutes) ?? durationTabs[0],
    [durationTabs, queryDurationMinutes],
  );

  const loadMeta = useCallback(
    async (signal) => {
      const { data } = await fetchAiHistoryMeta(symbol, { signal });
      if (signal.aborted) return;
      setDurationSummaries(data?.durationSummaries || []);
    },
    [symbol],
  );

  const loadPage = useCallback(
    async (signal) => {
      const { data } = await fetchAiHistorySuccess(symbol, {
        durationMinutes: queryDurationMinutes,
        page,
        pageSize: PAGE_SIZE,
        signal,
      });
      if (signal.aborted) return;
      setPeriodStats(data?.period || EMPTY_PERIOD);
      setByStrategy(data?.byStrategy || []);
      setPagination(data?.pagination || EMPTY_PAGINATION);
    },
    [symbol, queryDurationMinutes, page],
  );

  const reload = useCallback(
    async ({ background = false } = {}) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      if (!background) setRefreshing(true);
      try {
        await Promise.all([loadMeta(controller.signal), loadPage(controller.signal)]);
        if (controller.signal.aborted) return;
        setStatus("");
        hasLoadedRef.current = true;
      } catch (err) {
        if (controller.signal.aborted || err?.code === "ERR_CANCELED") return;
        console.error("规则命中率加载失败", err);
        setStatus(`加载失败：${err.message}`);
      } finally {
        if (!controller.signal.aborted) {
          setInitialLoading(false);
          setRefreshing(false);
        }
      }
    },
    [loadMeta, loadPage],
  );

  useEffect(() => {
    let timer;
    let stopped = false;
    const tick = async () => {
      if (stopped) return;
      await reload({ background: hasLoadedRef.current });
      if (!stopped) timer = window.setTimeout(tick, POLL_MS);
    };
    void tick();
    return () => {
      stopped = true;
      abortRef.current?.abort();
      if (timer) clearTimeout(timer);
    };
  }, [reload]);

  useEffect(() => {
    hasLoadedRef.current = false;
    setInitialLoading(true);
    setDurationSummaries([]);
    setPeriodStats(EMPTY_PERIOD);
    setByStrategy([]);
    setPagination(EMPTY_PAGINATION);
    setSelectedRow(null);
    setSelectedDurationMinutes(10);
    setPage(1);
  }, [symbol]);

  useEffect(() => {
    if (!selectedRow) return;
    if (selectedRow.durationMinutes !== queryDurationMinutes) setSelectedRow(null);
  }, [queryDurationMinutes, selectedRow]);

  useEffect(() => {
    if (!hasLoadedRef.current) return;
    setByStrategy([]);
  }, [queryDurationMinutes, page, symbol]);

  useEffect(() => {
    if (pagination.pageCount > 0 && page > pagination.pageCount) {
      setPage(pagination.pageCount);
    }
  }, [page, pagination.pageCount]);

  const { page: currentPage, pageCount, total: totalFactors } = pagination;
  const periodHeading = activeTab?.heading ?? "—";
  const factorCount = periodStats.factorCount ?? totalFactors;
  const showGlobalEmpty =
    !initialLoading && !refreshing && durationSummaries.length === 0 && byStrategy.length === 0;
  const showListSkeleton = (initialLoading || refreshing) && byStrategy.length === 0 && !showGlobalEmpty;

  const listRangeLabel =
    totalFactors === 0
      ? "0 个因子"
      : totalFactors <= PAGE_SIZE
        ? `${totalFactors} 个因子`
        : `${(currentPage - 1) * PAGE_SIZE + 1}–${Math.min(currentPage * PAGE_SIZE, totalFactors)} / ${totalFactors}`;

  const selectedTitle = selectedRow ? aiHistoryRowLabel(selectedRow) : "";
  const selectedEnglishName = selectedRow ? aiHistoryRowEnglishName(selectedRow) : "";

  return (
    <main className="rule-hit-rate-page layout">
      <header className="rhr-topbar card-surface">
        <div className="rhr-topbar-main">
          <span className="rhr-eyebrow">Performance</span>
          <h1>规则命中率</h1>
          <p>按结算周期切换查看因子命中率 · 点击因子查看最近合约记录</p>
        </div>
        <div className="rhr-topbar-actions">
          <div className="rhr-symbol-tabs" role="tablist" aria-label="交易对">
            {SYMBOLS.map((item) => (
              <button
                key={item.value}
                type="button"
                role="tab"
                aria-selected={symbol === item.value}
                className={`rhr-symbol-tab${symbol === item.value ? " is-active" : ""}`}
                onClick={() => setSymbol(item.value)}
              >
                {item.label}
              </button>
            ))}
          </div>
          <button type="button" className="rhr-refresh-btn" onClick={() => void reload()} disabled={refreshing}>
            {refreshing ? "同步中…" : "刷新"}
          </button>
        </div>
      </header>

      {status ? <div className="rhr-banner">{status}</div> : null}

      <div className="rhr-duration-tabs-wrap">
        <span className="rhr-duration-tabs-label">结算周期</span>
        <div className="rhr-duration-tabs" role="tablist" aria-label="结算周期">
          {durationTabs.map((tab) => (
            <button
              key={tab.durationKey}
              type="button"
              role="tab"
              aria-selected={queryDurationMinutes === tab.durationMinutes}
              className={`rhr-duration-tab${queryDurationMinutes === tab.durationMinutes ? " is-active" : ""}`}
              onClick={() => {
                setSelectedDurationMinutes(tab.durationMinutes);
                setSelectedRow(null);
                setPage(1);
              }}
            >
              <span className="rhr-duration-tab-label">{tab.heading}</span>
              <span className="rhr-duration-tab-count">
                {tab.factorCount == null ? "…" : tab.factorCount}
              </span>
            </button>
          ))}
        </div>
      </div>

      <section className="rhr-kpis" aria-label="当前周期汇总">
        <KpiCard
          label="命中率"
          loading={showListSkeleton}
          value={periodStats.total === 0 ? "—" : `${Math.round(periodStats.rate * 100)}%`}
          hint={periodStats.total === 0 ? "暂无样本" : `${periodStats.hits} / ${periodStats.total} 笔命中`}
          tone={hitRateTone(periodStats.rate)}
        />
        <KpiCard
          label="样本笔数"
          loading={showListSkeleton}
          value={periodStats.total || "—"}
          hint="已结算且可判定"
        />
        <KpiCard
          label="累计 PnL"
          loading={showListSkeleton}
          value={periodStats.total === 0 ? "—" : formatPnlU(periodStats.pnlU)}
          hint={`${symbol} · ${periodHeading}`}
          tone={periodStats.pnlU < 0 ? "down" : periodStats.pnlU > 0 ? "up" : "neutral"}
        />
        <KpiCard
          label="因子条目"
          loading={showListSkeleton}
          value={factorCount || "—"}
          hint={periodHeading}
        />
      </section>

      <div className={`rhr-body${selectedRow ? " has-detail" : ""}`}>
        <section className={`rhr-list card-surface${refreshing ? " is-refreshing" : ""}`} aria-busy={refreshing}>
          {showGlobalEmpty ? (
            <div className="rhr-empty">
              <div className="rhr-empty-icon" aria-hidden />
              <p>暂无已结算的规则样本</p>
              <span>模拟或规则下单并结算后，数据会出现在此处</span>
            </div>
          ) : (
            <div className="rhr-duration-block">
              <header className="rhr-duration-head">
                <div className="rhr-duration-title">
                  <span className="rhr-duration-badge">{periodHeading}</span>
                  <h2>因子列表</h2>
                </div>
                <span className="rhr-duration-count">{showListSkeleton ? "加载中…" : listRangeLabel}</span>
              </header>
              {showListSkeleton ? (
                <FactorListSkeleton />
              ) : byStrategy.length === 0 ? (
                <div className="rhr-empty rhr-empty-inline">
                  <p>该周期暂无因子样本</p>
                </div>
              ) : (
                <>
                  <ul className="rhr-factor-list">
                    {byStrategy.map((row) => {
                      const label = aiHistoryRowLabel(row);
                      const englishName = aiHistoryRowEnglishName(row);
                      const active =
                        selectedRow?.strategyKey === row.strategyKey &&
                        selectedRow?.durationMinutes === row.durationMinutes;
                      const ratePct = row.total ? Math.round(row.rate * 100) : null;
                      return (
                        <li key={`${row.strategyKey}-${row.durationMinutes}`}>
                          <button
                            type="button"
                            className={`rhr-factor-card${active ? " is-active" : ""}`}
                            onClick={() => setSelectedRow(row)}
                            title="查看最近合约记录"
                          >
                            <HitRateRing percent={ratePct} />
                            <div className="rhr-factor-main">
                              <span className="rhr-factor-name">{label}</span>
                              {englishName ? (
                                <code className="rhr-factor-code">{englishName}</code>
                              ) : null}
                              <span className="rhr-factor-meta">
                                {row.total === 0 ? "暂无样本" : `${row.hits} 命中 · ${row.total} 笔`}
                              </span>
                            </div>
                            <div className="rhr-factor-side">
                              {ratePct != null ? (
                                <span className={`rhr-rate-pill tone-${hitRateTone(row.rate)}`}>{ratePct}%</span>
                              ) : null}
                              <span
                                className={`rhr-pnl-pill${row.pnlU < 0 ? " is-loss" : row.pnlU > 0 ? " is-profit" : ""}`}
                              >
                                {row.total === 0 ? "—" : formatPnlU(row.pnlU)}
                              </span>
                            </div>
                            <span className="rhr-factor-chevron" aria-hidden />
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                  <FactorListPagination
                    page={currentPage}
                    pageCount={pageCount}
                    total={totalFactors}
                    onPageChange={setPage}
                  />
                </>
              )}
            </div>
          )}
        </section>

        {selectedRow ? (
          <StrategyRecentEventsPanel
            symbol={symbol}
            strategyKey={selectedRow.strategyKey}
            title={selectedTitle}
            englishName={selectedEnglishName}
            onClose={() => setSelectedRow(null)}
          />
        ) : (
          <aside className="rhr-detail-placeholder card-surface" aria-hidden={false}>
            <div className="rhr-detail-placeholder-inner">
              <span className="rhr-detail-placeholder-icon" aria-hidden />
              <h3>选择因子</h3>
              <p>点击左侧任意因子，在此查看最近合约记录、命中与 PnL 明细</p>
            </div>
          </aside>
        )}
      </div>
    </main>
  );
}

function FactorListSkeleton() {
  return (
    <ul className="rhr-factor-list rhr-factor-list-skeleton" aria-hidden>
      {Array.from({ length: SKELETON_ROWS }, (_, index) => (
        <li key={index}>
          <div className="rhr-factor-card rhr-skeleton-card">
            <div className="rhr-skeleton-ring" />
            <div className="rhr-skeleton-lines">
              <span className="rhr-skeleton-line wide" />
              <span className="rhr-skeleton-line" />
            </div>
          </div>
        </li>
      ))}
    </ul>
  );
}

function FactorListPagination({ page, pageCount, total, onPageChange }) {
  if (total <= PAGE_SIZE) {
    return total > 0 ? <p className="rhr-page-total">共 {total} 条</p> : null;
  }
  return (
    <nav className="rhr-pagination" aria-label="因子列表分页">
      <span className="rhr-page-total">共 {total} 条</span>
      <div className="rhr-page-actions">
        <button type="button" disabled={page <= 1} onClick={() => onPageChange(1)} aria-label="首页">
          «
        </button>
        <button type="button" disabled={page <= 1} onClick={() => onPageChange(page - 1)} aria-label="上一页">
          ‹
        </button>
        <strong className="rhr-page-indicator">
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

function KpiCard({ label, value, hint, tone = "neutral", loading = false }) {
  return (
    <article className={`rhr-kpi tone-${tone}${loading ? " is-skeleton" : ""}`}>
      <span>{label}</span>
      {loading ? <strong className="rhr-skeleton-block" aria-hidden /> : <strong>{value}</strong>}
      {hint && !loading ? <small>{hint}</small> : null}
    </article>
  );
}

function HitRateRing({ percent }) {
  const safe = percent == null ? 0 : Math.max(0, Math.min(100, percent));
  const tone = hitRateTone(percent == null ? null : percent / 100);
  return (
    <div className={`rhr-ring tone-${tone}`} style={{ "--pct": safe }} aria-hidden>
      <span>{percent == null ? "—" : `${percent}%`}</span>
    </div>
  );
}

function hitRateTone(rate) {
  if (rate == null || Number.isNaN(rate)) return "neutral";
  if (rate >= 0.6) return "up";
  if (rate >= 0.45) return "mid";
  return "down";
}
