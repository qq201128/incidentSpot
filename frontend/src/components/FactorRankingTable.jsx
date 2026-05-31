import { factorTitle, formatNum, formatPct } from "./factorDisplayUtils";
import { factorTableCategoryLabel } from "../utils/factorCatalogLabels";
import "./FactorRankingTable.css";

const RANKING_PAGE_SIZE = 8;

export default function FactorRankingTable({
  onPageChange,
  onQueryChange,
  onSelectFactor,
  page = 1,
  pageCount = 1,
  query = "",
  ranking,
  selectedName,
  total,
  unfilteredTotal,
}) {
  const safeTotal = Number(total ?? ranking.length);
  const rankOffset = (page - 1) * RANKING_PAGE_SIZE;

  return (
    <section className="factors-ranking-block">
      <header className="factors-ranking-header">
        <h3 className="factors-subhead">排名缓存 (按综合评分排名)</h3>
        <span>{rankingCountText(safeTotal, unfilteredTotal)}</span>
      </header>
      <label className="factors-ranking-search">
        <span className="sr-only">搜索排名缓存</span>
        <input
          value={query}
          onChange={(event) => onQueryChange?.(event.target.value)}
          placeholder="搜索因子名/类别/来源"
        />
      </label>
      <div className="factors-ranking-wrap">
        <table className="factors-table factors-ranking-table">
          <thead>
            <tr>
              <th>排名</th>
              <th>因子</th>
              <th>类别</th>
              <th>周期</th>
              <th>评分</th>
              <th>胜率</th>
              <th>IC均值</th>
              <th>IR</th>
              <th title="相关性(均值)">相关性</th>
              <th>贡献度</th>
            </tr>
          </thead>
          <tbody>
            {ranking.map((row, index) => {
              const name = row.factorName || row.name;
              return (
                <tr
                  key={name}
                  className={selectedName === name ? "factors-row-selected" : ""}
                  onClick={() => onSelectFactor(name)}
                >
                  <td>{rankOffset + index + 1}</td>
                  <td className="factors-ranking-factor-cell">
                    <strong className="factors-name-cn" title={factorTitle(row)}>
                      {factorTitle(row)}
                    </strong>
                    <code className="factors-code" title={name}>
                      {name}
                    </code>
                  </td>
                  <td>{factorTableCategoryLabel(row)}</td>
                  <td className="factors-ranking-num">{row.duration || "—"}</td>
                  <td className="factors-ranking-num">{formatNum(row.factorScore, 1)}</td>
                  <td className="factors-ranking-num">{formatPct(row.winRate, 1)}</td>
                  <td className="factors-ranking-num">{formatNum(row.icMean, 4)}</td>
                  <td className="factors-ranking-num">{formatNum(row.ir, 2)}</td>
                  <td className="factors-ranking-num">{formatPct(row.avgAbsCorrelation, 1)}</td>
                  <td className="factors-ranking-num">{formatPct(row.contribution, 1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {!safeTotal ? <p className="factors-empty factors-ranking-empty">暂无排名数据</p> : null}
      </div>
      {safeTotal > RANKING_PAGE_SIZE ? (
        <RankingPagination page={page} pageCount={pageCount} total={safeTotal} onPageChange={onPageChange} />
      ) : safeTotal ? (
        <p className="factors-ranking-page-total">共 {safeTotal} 条</p>
      ) : null}
    </section>
  );
}

function rankingCountText(total, unfilteredTotal) {
  const raw = Number(unfilteredTotal ?? total);
  return raw !== total ? `${total} / ${raw} 项` : `${total} 项`;
}

function RankingPagination({ page, pageCount, total, onPageChange }) {
  return (
    <nav className="factors-ranking-pagination" aria-label="排名缓存分页">
      <span className="factors-ranking-page-total">共 {total} 条</span>
      <div className="factors-ranking-page-actions">
        <button type="button" disabled={page <= 1} onClick={() => onPageChange(1)} aria-label="首页">
          «
        </button>
        <button type="button" disabled={page <= 1} onClick={() => onPageChange(page - 1)} aria-label="上一页">
          ‹
        </button>
        <strong className="factors-ranking-page-indicator">
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
