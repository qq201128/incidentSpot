import "./FactorCombinationRankingTable.css";

export default function FactorCombinationRankingTable({
  highWinrateRanking = [],
  highWinrateSummary = null,
  ranking,
}) {
  return (
    <div className="factor-combo-ranking-stack">
      <RankingSection
        emptyText="暂无高胜率目标组合"
        ranking={highWinrateRanking}
        summary={highWinrateSummaryText(highWinrateSummary)}
        title="高胜率目标组合"
      />
      <RankingSection emptyText="暂无普通组合" ranking={ranking} title="普通组合" />
    </div>
  );
}

function RankingSection({ emptyText, ranking, summary = "", title }) {
  return (
    <div className="factor-combo-ranking">
      <div className="factor-combo-ranking-title">
        <h3>{title}</h3>
        <span>{countText(ranking.length, summary)}</span>
      </div>
      <div className="factor-combo-rank-list">
        {ranking.map(renderRankingRow)}
        {!ranking.length ? <p className="factor-combo-empty">{emptyText}</p> : null}
      </div>
    </div>
  );
}

function renderRankingRow(row, index) {
  return (
    <article key={row.factorName || index} className="factor-combo-rank-card">
      <div className="factor-combo-rank-index">
        <b>{index + 1}</b>
        <span className={comboTypeClass(row)}>{comboTypeText(row)}</span>
      </div>
      <div className="factor-combo-rank-body">
        <strong className="factor-combo-rank-name">{row.factorDisplayName || row.factorName}</strong>
        <code className="factor-combo-rank-code">{row.factorName}</code>
        <p>{memberText(row.members)}</p>
      </div>
      <div className="factor-combo-rank-metrics">
        <Metric label="胜率" value={formatPct(row.winRate, 1)} strong={isGoalCombo(row)} />
        <Metric label="日均单量" value={formatNum(row.avgTradesPerDay, 1)} />
        <Metric label="评分" value={formatNum(row.factorScore, 1)} />
        <Metric label="盈亏比" value={formatNum(row.profitFactor, 2)} />
        <Metric label="夏普" value={formatNum(row.sharpe, 2)} />
        <Metric label="相关" value={formatPct(row.avgAbsCorrelation, 1)} />
      </div>
    </article>
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

function formatPct(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}
