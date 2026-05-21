import { useEffect, useMemo, useState } from "react";
import {
  copyToClipboard,
  directionDetailLabel,
  factorTitle,
  formatNum,
  formatPct,
} from "./factorDisplayUtils";
import "./FactorDetailPanel.css";

const DURATION_LABELS = { "10m": "10m", "30m": "30m", "60m": "60m", "1d": "1d" };
const DURATION_ORDER = ["10m", "30m", "60m", "1d"];

export default function FactorDetailPanel({
  backtestError,
  detail,
  detailError,
  detailLoading = false,
  displayMetrics,
  duration,
  onDurationChange,
  periodScores,
  periodScoresPending = false,
  selectedFactor,
  selectedName,
}) {
  const [starred, setStarred] = useState(false);

  useEffect(() => {
    setStarred(false);
  }, [selectedName]);

  const activeDetail = useMemo(() => {
    if (detail?.name === selectedName) return detail;
    if (selectedFactor?.name === selectedName) return selectedFactor;
    return detail;
  }, [detail, selectedFactor, selectedName]);

  const detailPending = Boolean(selectedName && detailLoading && detail?.name !== selectedName);

  if (!selectedName) {
    return <p className="factors-placeholder">在左侧目录选择因子，查看定义、回测指标与排名缓存。</p>;
  }

  return (
    <div className={`factor-detail-stack${detailPending ? " is-detail-pending" : ""}`}>
      <DetailHeader
        detail={activeDetail}
        duration={duration}
        onDurationChange={onDurationChange}
        selectedName={selectedName}
        starred={starred}
        onToggleStar={() => setStarred((value) => !value)}
      />
      {detailError ? <p className="factors-error">{detailError}</p> : null}
      {backtestError ? <p className="factors-error">{backtestError}</p> : null}
      <div className="factor-detail-body">
        {activeDetail ? <DefinitionColumn detail={activeDetail} /> : null}
        <MetricsColumn metrics={displayMetrics} />
        {periodScores?.length ? (
          <PeriodScoreChart
            pending={periodScoresPending}
            scores={periodScores}
            activeDuration={duration}
          />
        ) : (
          <section className="factor-period-chart factor-period-chart-empty">
            <h3 className="factors-subhead">各周期评分</h3>
            <p className="factors-placeholder">暂无各周期排名缓存</p>
          </section>
        )}
      </div>
    </div>
  );
}

function DetailHeader({ detail, duration, onDurationChange, selectedName, starred, onToggleStar }) {
  const timeframes = detail?.timeframes?.length ? detail.timeframes : Object.keys(DURATION_LABELS);
  const title = detail ? factorTitle(detail) : selectedName;
  return (
    <header className="factor-detail-head">
      <div className="factor-detail-title-block">
        <h2>
          {title || "—"}
          {selectedName ? <code className="factor-detail-name-tag">{selectedName}</code> : null}
        </h2>
      </div>
      <div className="factor-detail-head-actions">
        {detail?.canStore ? <span className="factor-store-tag">可入库</span> : null}
        <div className="factor-duration-toggle" role="tablist" aria-label="规则周期">
          {timeframes.map((item) => (
            <button
              key={item}
              type="button"
              role="tab"
              aria-selected={duration === item}
              className={duration === item ? "is-active" : ""}
              onClick={() => onDurationChange(item)}
            >
              {DURATION_LABELS[item] || item}
            </button>
          ))}
        </div>
        <button
          type="button"
          className={`factor-star-btn${starred ? " is-starred" : ""}`}
          title="收藏"
          onClick={onToggleStar}
          aria-pressed={starred}
        >
          ★
        </button>
      </div>
    </header>
  );
}

function DefinitionColumn({ detail }) {
  return (
    <section className="factor-definition-column" aria-label="因子定义">
      <dl className="factor-definition-list">
        <DefinitionRow label="英文/字段名" value={<code className="factors-code">{detail.name}</code>} />
        <DefinitionRow label="说明" value={detail.description} />
        <DefinitionRow label="公式" value={<FormulaValue formula={detail.formula} />} />
        <DefinitionRow label="周期" value={(detail.timeframes || []).join("、") || "—"} />
        <DefinitionRow label="方向" value={directionDetailLabel(detail.direction)} />
        <DefinitionRow label="源码" value={<SourceValue path={detail.sourceFile} />} />
      </dl>
    </section>
  );
}

function DefinitionRow({ label, value }) {
  return (
    <div className="factor-definition-row">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function FormulaValue({ formula }) {
  return (
    <div className="factor-copy-row">
      <code className="factors-formula">{formula || "—"}</code>
      <CopyButton text={formula} />
    </div>
  );
}

function SourceValue({ path }) {
  const display = path ? `local://factors/${path}` : "—";
  return (
    <div className="factor-copy-row">
      <code className="factors-code is-path">{display}</code>
      <CopyButton text={display} />
    </div>
  );
}

function CopyButton({ text }) {
  return (
    <button type="button" className="factor-copy-btn" onClick={() => void copyToClipboard(text)} aria-label="复制">
      ⧉
    </button>
  );
}

function MetricsColumn({ metrics }) {
  return (
    <section className="factor-metrics-column" aria-label="回测指标">
      <h3 className="factors-subhead">回测指标</h3>
      {metrics ? <MetricsGrid metrics={metrics} /> : (
        <p className="factors-placeholder">暂无该周期的排名缓存指标，可在顶部点击「运行回测」。</p>
      )}
    </section>
  );
}

function MetricsGrid({ metrics }) {
  const items = [
    { label: "综合评分", value: formatNum(metrics.factorScore, 1), tone: "positive", featured: true },
    { label: "胜率", value: formatPct(metrics.winRate, 1), tone: "positive" },
    { label: "IC均值", value: formatNum(metrics.icMean, 3), tone: "positive" },
    { label: "IR", value: formatNum(metrics.ir, 2), tone: "positive" },
    { label: "多空收益", value: formatNum(metrics.longShortReturn, 4), tone: "positive" },
    { label: "最大回撤", value: formatPct(metrics.maxDrawdown, 1), tone: "negative" },
    { label: "盈亏比", value: formatNum(metrics.profitFactor, 2), tone: "positive" },
    { label: "t统计量", value: formatNum(metrics.tStat, 2), tone: "positive" },
    { label: "p值", value: formatNum(metrics.pValue, 4) },
    { label: "样本期数", value: metrics.totalPeriods ?? "—" },
  ];
  return (
    <dl className="factor-metrics-grid">
      {items.map((item) => (
        <div
          key={item.label}
          className={`factor-metric-cell${item.tone ? ` is-${item.tone}` : ""}${item.featured ? " is-featured" : ""}`}
        >
          <dt>{item.label}</dt>
          <dd>{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function PeriodScoreChart({ scores, activeDuration, pending = false }) {
  const ordered = [...scores].sort(
    (a, b) => DURATION_ORDER.indexOf(a.duration) - DURATION_ORDER.indexOf(b.duration),
  );
  const available = ordered.filter((row) => row.available && row.factorScore != null);
  const maxScore = Math.max(...available.map((row) => Number(row.factorScore) || 0), 1);
  return (
    <section
      className={`factor-period-chart${pending ? " is-pending" : ""}`}
      aria-label="各周期综合评分"
    >
      <h3 className="factors-subhead">各周期评分</h3>
      <div className="factor-period-bars">
        {ordered.map((row) => {
          const height = row.available && row.factorScore != null
            ? `${(Number(row.factorScore) / maxScore) * 100}%`
            : "4px";
          return (
            <div
              key={row.duration}
              className={`factor-period-bar${row.duration === activeDuration ? " is-active" : ""}${row.available ? "" : " is-empty"}`}
            >
              <div className="factor-period-bar-fill" style={{ height }} title={formatNum(row.factorScore, 1)} />
              <span>{DURATION_LABELS[row.duration] || row.duration}</span>
              <small>{row.available ? formatNum(row.factorScore, 1) : "—"}</small>
            </div>
          );
        })}
      </div>
    </section>
  );
}
