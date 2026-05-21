import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchEnsembleRanking } from "../api/client";
import { strategyLabel } from "../utils/strategyLabels";
import "./EnsembleRankingTable.css";

const PAGE_SIZE = 8;

export default function EnsembleRankingTable({ symbol, duration }) {
  const [ranking, setRanking] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [page, setPage] = useState(1);

  const load = useCallback(async () => {
    if (!symbol || symbol.length < 6) return;
    setError("");
    const data = await fetchEnsembleRanking(symbol, duration);
    setRanking(Array.isArray(data?.ranking) ? data.ranking : []);
  }, [duration, symbol]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    load()
      .catch((err) => {
        if (!cancelled) setError(_errorMessage(err, "读取候选信号排名失败"));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [load]);

  useEffect(() => {
    setPage(1);
  }, [duration, ranking.length, symbol]);

  const pageCount = Math.max(1, Math.ceil(ranking.length / PAGE_SIZE));
  useEffect(() => {
    if (page > pageCount) setPage(pageCount);
  }, [page, pageCount]);

  const pageRows = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;
    return ranking.slice(start, start + PAGE_SIZE);
  }, [page, ranking]);

  if (loading) {
    return <p className="ensemble-records-empty">正在加载候选信号排名…</p>;
  }

  return (
    <div className="ensemble-records">
      {error ? <p className="ensemble-records-error">{error}</p> : null}
      <div className="ensemble-records-table" role="table" aria-label="候选信号排名">
        <div className="ensemble-records-rows">
          <div className="ensemble-records-row ensemble-records-row--header" role="row">
            <span role="columnheader">#</span>
            <span role="columnheader">信号</span>
            <span role="columnheader">类型</span>
            <span role="columnheader">样本</span>
            <span role="columnheader">胜率</span>
            <span role="columnheader">盈亏比</span>
            <span role="columnheader">权重</span>
            <span role="columnheader">状态</span>
          </div>
          {pageRows.map((row, index) => (
            <RankingRow key={row.signalKey} index={(page - 1) * PAGE_SIZE + index + 1} row={row} />
          ))}
        </div>
        {!ranking.length ? (
          <p className="ensemble-records-empty">暂无已结算候选，请在综合裁判中刷新统计</p>
        ) : null}
      </div>
      {ranking.length > PAGE_SIZE ? (
        <nav className="ensemble-records-pagination" aria-label="候选信号分页">
          <span>共 {ranking.length} 条</span>
          <div className="ensemble-records-page-actions">
            <button type="button" disabled={page <= 1} onClick={() => setPage(1)}>
              «
            </button>
            <button type="button" disabled={page <= 1} onClick={() => setPage(page - 1)}>
              ‹
            </button>
            <strong>
              {page} / {pageCount}
            </strong>
            <button type="button" disabled={page >= pageCount} onClick={() => setPage(page + 1)}>
              ›
            </button>
            <button type="button" disabled={page >= pageCount} onClick={() => setPage(pageCount)}>
              »
            </button>
          </div>
        </nav>
      ) : null}
    </div>
  );
}

function RankingRow({ index, row }) {
  const badges = _badges(row);
  const title = strategyLabel(row.signalKey);
  const winGood = Number(row.winRate) >= 0.5;
  const pfGood = Number(row.profitFactor) >= 1;
  return (
    <div
      className={`ensemble-records-row${row.lowSample ? " is-low-sample" : ""}`}
      role="row"
    >
      <span role="cell">{index}</span>
      <span className="ensemble-records-signal" role="cell" title={row.signalKey}>
        <strong>{title}</strong>
      </span>
      <span role="cell">{_typeLabel(row.signalType)}</span>
      <span role="cell">{row.sampleCount ?? "—"}</span>
      <span className={winGood ? "is-good" : ""} role="cell">
        {_pct(row.winRate)}
      </span>
      <span className={pfGood ? "is-good" : ""} role="cell">
        {_num(row.profitFactor)}
      </span>
      <span role="cell">{_num(row.weightSuggestion)}</span>
      <span className="ensemble-records-tags" role="cell">
        {badges.length ? badges.join(" · ") : "稳定"}
      </span>
    </div>
  );
}

function _badges(row) {
  const badges = [];
  if (row.lowSample) badges.push("低样本");
  if (row.insufficientSample && !row.lowSample) badges.push("样本不足");
  if (row.weakSignal) badges.push("近期走弱");
  if (Number(row.consecutiveLosses) >= 5) badges.push(`连亏${row.consecutiveLosses}`);
  if (row.degraded && !row.weakSignal) badges.push("降权中");
  return badges;
}

function _typeLabel(type) {
  if (type === "factor_combo") return "多因子";
  if (type === "high_winrate_combo") return "高胜率";
  if (type === "model_family") return "模型族";
  return "其他";
}

function _pct(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : "—";
}

function _num(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(2) : "—";
}

function _errorMessage(err, fallback) {
  return err?.response?.data?.detail || err?.message || fallback;
}
