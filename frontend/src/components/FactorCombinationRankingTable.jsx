import { useEffect, useMemo, useState } from "react";
import { failureReasonLabel } from "../utils/failureReasonLabels";
import "./FactorCombinationRankingTable.css";

const COMBO_PAGE_SIZE = 6;

export default function FactorCombinationRankingTable({
  highWinrateRanking = [],
  highWinrateSummary = null,
  onPageChange,
  onQueryChange,
  page = 1,
  pageCount = 1,
  query = "",
  ranking,
  total,
  unfilteredTotal,
  passedTotal,
  evaluatedTotal,
}) {
  return (
    <div className="factor-combo-ranking-stack">
      <RankingSection
        emptyText="暂无高胜率目标组合"
        ranking={highWinrateRanking}
        summary={highWinrateSummaryText(highWinrateSummary)}
        title="高胜率目标组合"
        variant="goal"
      />
      <RegularRankingSection
        onPageChange={onPageChange}
        onQueryChange={onQueryChange}
        page={page}
        pageCount={pageCount}
        query={query}
        ranking={ranking}
        total={total}
        unfilteredTotal={unfilteredTotal}
        passedTotal={passedTotal}
        evaluatedTotal={evaluatedTotal}
      />
    </div>
  );
}

function RankingSection({ emptyText, ranking, summary = "", title, variant = "regular" }) {
  const [page, setPage] = useState(1);
  const total = ranking.length;
  const pageCount = Math.max(1, Math.ceil(total / COMBO_PAGE_SIZE));

  useEffect(() => {
    setPage(1);
  }, [ranking]);

  useEffect(() => {
    if (page > pageCount) setPage(pageCount);
  }, [page, pageCount]);

  const pageRows = useMemo(() => {
    const start = (page - 1) * COMBO_PAGE_SIZE;
    return ranking.slice(start, start + COMBO_PAGE_SIZE);
  }, [page, ranking]);

  const rankOffset = (page - 1) * COMBO_PAGE_SIZE;

  return (
    <section className={`factor-combo-ranking factor-combo-ranking-${variant} card-surface`}>
      <header className="factor-combo-ranking-title">
        <div>
          <span className="section-kicker">{variant === "goal" ? "组合缓存" : "排名列表"}</span>
          <h3>{title}</h3>
        </div>
        <span>{countText(total, summary)}</span>
      </header>
      <div className="factor-combo-rank-list">
        {pageRows.map((row, index) => renderRankingRow(row, rankOffset + index))}
        {!total ? <p className="factor-combo-empty">{emptyText}</p> : null}
      </div>
      {total > COMBO_PAGE_SIZE ? (
        <ComboPagination page={page} pageCount={pageCount} total={total} onPageChange={setPage} />
      ) : total ? (
        <p className="factor-combo-rank-total">共 {total} 条</p>
      ) : null}
    </section>
  );
}

function RegularRankingSection({
  onPageChange,
  onQueryChange,
  page,
  pageCount,
  query,
  ranking,
  total,
  unfilteredTotal,
  passedTotal,
  evaluatedTotal,
}) {
  const safeTotal = Number(total ?? ranking.length);
  return (
    <section className="factor-combo-ranking factor-combo-ranking-regular card-surface">
      <header className="factor-combo-ranking-title">
        <div>
          <span className="section-kicker">排名列表</span>
          <h3>普通组合</h3>
        </div>
        <span>{regularCountText(safeTotal, unfilteredTotal, passedTotal, evaluatedTotal)}</span>
      </header>
      <label className="factor-combo-rank-search">
        <span className="sr-only">搜索普通组合</span>
        <input
          value={query}
          onChange={(event) => onQueryChange?.(event.target.value)}
          placeholder="搜索组合名或成员因子"
        />
      </label>
      <div className="factor-combo-rank-list">
        {ranking.map((row, index) => renderRankingRow(row, (page - 1) * COMBO_PAGE_SIZE + index))}
        {!safeTotal ? <p className="factor-combo-empty">暂无普通组合</p> : null}
      </div>
      {safeTotal ? (
        <ComboPagination page={page} pageCount={pageCount} total={safeTotal} onPageChange={onPageChange} />
      ) : null}
    </section>
  );
}

function renderRankingRow(row, index) {
  const sampleMetric = comboSampleMetric(row);
  return (
    <article key={row.factorName || index} className="factor-combo-rank-card">
      <div className="factor-combo-rank-index">
        <b>{index + 1}</b>
        <span className={comboTypeClass(row)}>{comboTypeText(row)}</span>
      </div>
      <div className="factor-combo-rank-body">
        <strong className="factor-combo-rank-name">{row.factorDisplayName || row.factorName}</strong>
        <code className="factor-combo-rank-code">{row.factorName}</code>
        <p className="factor-combo-rank-members">{memberText(row.members)}</p>
        <p className={`factor-combo-rank-gate ${row.walkForwardPassed ? "is-pass" : "is-fail"}`}>
          {walkForwardText(row)}
        </p>
      </div>
      <div className="factor-combo-rank-metrics">
        <Metric label="胜率" value={formatPct(row.winRate, 1)} strong={isGoalCombo(row)} />
        <Metric label={sampleMetric.label} value={sampleMetric.value} />
        <Metric label="评分" value={formatNum(row.factorScore, 1)} />
        <Metric label="盈亏比" value={formatNum(row.profitFactor, 2)} />
        <Metric label="夏普" value={formatNum(row.sharpe, 2)} />
        <Metric label="相关" value={formatPct(row.avgAbsCorrelation, 1)} />
      </div>
    </article>
  );
}

function ComboPagination({ page, pageCount, total, onPageChange }) {
  return (
    <nav className="factor-combo-rank-pagination" aria-label="组合排名分页">
      <span className="factor-combo-rank-page-total">共 {total} 条</span>
      <div className="factor-combo-rank-page-actions">
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

function comboTypeText(row) {
  return isGoalCombo(row) ? "高胜率目标" : "普通组合";
}

function comboTypeClass(row) {
  return `factor-combo-rank-type ${isGoalCombo(row) ? "is-goal" : "is-regular"}`;
}

function countText(count, summary) {
  return summary ? `${count} 项 · ${summary}` : `${count} 项`;
}

function regularCountText(total, unfilteredTotal, passedTotal, evaluatedTotal) {
  if (evaluatedTotal != null || passedTotal != null) {
    const evaluated = Number(evaluatedTotal ?? unfilteredTotal ?? total);
    const passed = Number(passedTotal ?? 0);
    return `${total} 项 · 已评估 ${evaluated} · 通过 ${passed}`;
  }
  const raw = Number(unfilteredTotal ?? total);
  return raw !== total ? `${total} / ${raw} 项` : `${total} 项`;
}

function walkForwardText(row) {
  if (row.walkForwardPassed === true) return "walk-forward 通过，可进入交易候选";
  const reason = row.walkForwardFailureReason
    ? failureReasonLabel(row.walkForwardFailureReason)
    : "未通过 walk-forward";
  return `观察候选 · ${reason}`;
}

function comboSampleMetric(row) {
  if (row.avgTradesPerDay != null && !Number.isNaN(Number(row.avgTradesPerDay))) {
    return { label: "日均单量", value: formatNum(row.avgTradesPerDay, 1) };
  }
  if (row.trades != null && !Number.isNaN(Number(row.trades))) {
    return { label: "交易数", value: formatInt(row.trades) };
  }
  return { label: "样本数", value: formatInt(row.totalPeriods) };
}

function highWinrateSummaryText(summary) {
  const avgTrades = formatNum(summary?.topStrategyAvgTradesPerDay, 1);
  if (avgTrades === "—") return "";
  return `Top1 日均 ${avgTrades} 单`;
}

function isGoalCombo(row) {
  return String(row.factorName || "").startsWith("goal_combo__");
}

function Metric({ label, value, strong = false }) {
  return (
    <span className={strong ? "is-strong" : ""}>
      <small>{label}</small>
      <b>{value}</b>
    </span>
  );
}

function memberText(members) {
  if (!Array.isArray(members) || !members.length) return "—";
  return members.map((member) => member.displayName || member.name).join(" + ");
}

function formatNum(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function formatInt(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Math.trunc(Number(value)).toLocaleString();
}

function formatPct(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}
