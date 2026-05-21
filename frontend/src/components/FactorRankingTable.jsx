import { useEffect, useMemo, useState } from "react";
import { factorTitle, formatNum, formatPct } from "./factorDisplayUtils";
import { factorTableCategoryLabel } from "../utils/factorCatalogLabels";
import "./FactorRankingTable.css";

const RANKING_PAGE_SIZE = 8;

export default function FactorRankingTable({ ranking, selectedName, onSelectFactor }) {
  const [page, setPage] = useState(1);

  const total = ranking.length;
  const pageCount = Math.max(1, Math.ceil(total / RANKING_PAGE_SIZE));

  useEffect(() => {
    setPage(1);
  }, [ranking]);

  useEffect(() => {
    if (page > pageCount) setPage(pageCount);
  }, [page, pageCount]);

  const pageRows = useMemo(() => {
    const start = (page - 1) * RANKING_PAGE_SIZE;
    return ranking.slice(start, start + RANKING_PAGE_SIZE);
  }, [page, ranking]);

  const rankOffset = (page - 1) * RANKING_PAGE_SIZE;

  return (
    <section className="factors-ranking-block">
      <h3 className="factors-subhead">排名缓存 (按综合评分排名)</h3>
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
            {pageRows.map((row, index) => {
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
        {!total ? <p className="factors-empty factors-ranking-empty">暂无排名数据</p> : null}
      </div>
      {total > RANKING_PAGE_SIZE ? (
        <RankingPagination page={page} pageCount={pageCount} total={total} onPageChange={setPage} />
      ) : total ? (
        <p className="factors-ranking-page-total">共 {total} 条</p>
      ) : null}
    </section>
  );
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
