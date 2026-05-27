import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchWorkbenchSummary } from "../api/workbenchClient";
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
const UNKNOWN_DURATION = -1;

const EMPTY_AI_HISTORY = Object.freeze({
  overall: { total: 0, hits: 0, rate: null, pnlU: 0 },
  byStrategy: [],
});

export default function RuleHitRatePage() {
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [aiHistorySuccess, setAiHistorySuccess] = useState(EMPTY_AI_HISTORY);
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState("");
  const [selectedRow, setSelectedRow] = useState(null);

  const reload = useCallback(async () => {
    try {
      const { data } = await fetchWorkbenchSummary(symbol, "10m");
      setAiHistorySuccess(data?.aiHistorySuccess || EMPTY_AI_HISTORY);
      setStatus("");
    } catch (err) {
      console.error("规则命中率加载失败", err);
      setStatus(`加载失败：${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  useEffect(() => {
    setLoading(true);
    let timer;
    let stopped = false;
    const tick = async () => {
      if (stopped) return;
      await reload();
      if (!stopped) timer = window.setTimeout(tick, POLL_MS);
    };
    void tick();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [reload]);

  useEffect(() => {
    setSelectedRow(null);
  }, [symbol]);

  const durationGroups = useMemo(
    () => groupAiSuccessByTradeDuration(aiHistorySuccess.byStrategy || []),
    [aiHistorySuccess.byStrategy],
  );
  const { overall } = aiHistorySuccess;
  const selectedTitle = selectedRow ? aiHistoryRowLabel(selectedRow) : "";
  const selectedEnglishName = selectedRow ? aiHistoryRowEnglishName(selectedRow) : "";
  const factorCount = aiHistorySuccess.byStrategy?.length ?? 0;

  return (
    <main className="rule-hit-rate-page layout">
      <header className="rhr-topbar card-surface">
        <div className="rhr-topbar-main">
          <span className="rhr-eyebrow">Performance</span>
          <h1>规则命中率</h1>
          <p>已结算样本按因子聚合 · 点击因子名称查看最近合约记录</p>
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
          <button type="button" className="rhr-refresh-btn" onClick={() => void reload()} disabled={loading}>
            {loading ? "同步中…" : "刷新"}
          </button>
        </div>
      </header>

      {status ? <div className="rhr-banner">{status}</div> : null}

      <section className="rhr-kpis" aria-label="汇总指标">
        <KpiCard
          label="命中率"
          value={overall.total === 0 ? "—" : `${Math.round(overall.rate * 100)}%`}
          hint={overall.total === 0 ? "暂无样本" : `${overall.hits} / ${overall.total} 笔命中`}
          tone={hitRateTone(overall.rate)}
        />
        <KpiCard label="样本笔数" value={overall.total || "—"} hint="已结算且可判定" />
        <KpiCard
          label="累计 PnL"
          value={overall.total === 0 ? "—" : formatPnlU(overall.pnlU)}
          hint={symbol}
          tone={overall.pnlU < 0 ? "down" : overall.pnlU > 0 ? "up" : "neutral"}
        />
        <KpiCard label="因子条目" value={factorCount || "—"} hint={`${durationGroups.length} 个时长分组`} />
      </section>

      <div className={`rhr-body${selectedRow ? " has-detail" : ""}`}>
        <section className="rhr-list card-surface" aria-busy={loading}>
          {durationGroups.length === 0 ? (
            <div className="rhr-empty">
              <div className="rhr-empty-icon" aria-hidden />
              <p>{loading ? "正在加载命中率数据…" : "暂无已结算的规则样本"}</p>
              <span>模拟或规则下单并结算后，数据会出现在此处</span>
            </div>
          ) : (
            durationGroups.map((group) => (
              <div key={group.durationKey} className="rhr-duration-block">
                <header className="rhr-duration-head">
                  <div className="rhr-duration-title">
                    <span className="rhr-duration-badge">{group.heading}</span>
                    <h2>结算周期</h2>
                  </div>
                  <span className="rhr-duration-count">{group.rows.length} 个因子</span>
                </header>
                <ul className="rhr-factor-list">
                  {group.rows.map((row) => {
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
                            <span className={`rhr-pnl-pill${row.pnlU < 0 ? " is-loss" : row.pnlU > 0 ? " is-profit" : ""}`}>
                              {row.total === 0 ? "—" : formatPnlU(row.pnlU)}
                            </span>
                          </div>
                          <span className="rhr-factor-chevron" aria-hidden />
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))
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

function KpiCard({ label, value, hint, tone = "neutral" }) {
  return (
    <article className={`rhr-kpi tone-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {hint ? <small>{hint}</small> : null}
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

function groupAiSuccessByTradeDuration(rows) {
  const map = new Map();
  for (const row of rows) {
    const dm = row.durationMinutes;
    const durationKey = dm === UNKNOWN_DURATION ? "unknown" : String(dm);
    if (!map.has(durationKey)) {
      const sortKey = dm === UNKNOWN_DURATION ? Number.POSITIVE_INFINITY : dm;
      const heading =
        dm === UNKNOWN_DURATION ? contractDurationLabel(null) : contractDurationLabel(dm);
      map.set(durationKey, { durationKey, sortKey, heading, rows: [] });
    }
    map.get(durationKey).rows.push(row);
  }
  for (const g of map.values()) {
    g.rows.sort((a, b) => b.pnlU - a.pnlU || aiHistoryRowLabel(a).localeCompare(aiHistoryRowLabel(b), "zh-CN"));
  }
  return [...map.values()].sort((a, b) => a.sortKey - b.sortKey);
}
