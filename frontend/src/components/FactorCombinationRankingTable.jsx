const RANKING_PREVIEW_LIMIT = 16;

export default function FactorCombinationRankingTable({ ranking }) {
  return (
    <div className="factor-combo-ranking">
      <div className="factor-combo-ranking-title">
        <h3>综合评分组合</h3>
        <span>{ranking.length} 项</span>
      </div>
      <div className="factor-combo-table-wrap">
        <table className="factors-table factor-combo-table">
          <thead>
            <tr>
              <th>#</th>
              <th>组合因子</th>
              <th>成员</th>
              <th>评分</th>
              <th>胜率</th>
              <th>相关性</th>
              <th>贡献</th>
              <th>夏普</th>
              <th>IR</th>
              <th>盈亏比</th>
            </tr>
          </thead>
          <tbody>{ranking.slice(0, RANKING_PREVIEW_LIMIT).map(renderRankingRow)}</tbody>
        </table>
        {!ranking.length ? <p className="factor-combo-empty">暂无组合排名</p> : null}
      </div>
    </div>
  );
}

function renderRankingRow(row, index) {
  return (
    <tr key={row.factorName || index}>
      <td>{index + 1}</td>
      <td>
        <strong className="factors-name-cn">{row.factorDisplayName || row.factorName}</strong>
        <code className="factors-code">{row.factorName}</code>
      </td>
      <td>{memberText(row.members)}</td>
      <td>{formatNum(row.factorScore, 1)}</td>
      <td>{formatPct(row.winRate, 1)}</td>
      <td>{formatPct(row.avgAbsCorrelation, 1)}</td>
      <td>{formatPct(row.contribution, 1)}</td>
      <td>{formatNum(row.sharpe, 2)}</td>
      <td>{formatNum(row.ir, 4)}</td>
      <td>{formatNum(row.profitFactor, 2)}</td>
    </tr>
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
