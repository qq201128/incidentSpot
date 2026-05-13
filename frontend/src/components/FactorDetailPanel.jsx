import { directionLabel, factorTitle, formatNum, formatPct } from "./factorDisplayUtils";

const RANKING_LIMIT = 40;

export default function FactorDetailPanel({
  backtest,
  backtestError,
  backtestLoading,
  detail,
  detailError,
  onRunBacktest,
  onSelectFactor,
  ranking,
  selectedName,
}) {
  return (
    <section className="factors-detail-panel card-surface">
      <PanelHeader detail={detail} selectedName={selectedName} />
      {detailError ? <p className="factors-error">{detailError}</p> : null}
      {detail ? <FactorDefinition detail={detail} /> : null}
      {!selectedName ? <p className="factors-placeholder">在左侧表格中点击一行查看定义与回测。</p> : null}
      {selectedName ? (
        <BacktestBlock
          backtest={backtest}
          error={backtestError}
          loading={backtestLoading}
          onRun={onRunBacktest}
        />
      ) : null}
      <RankingBlock ranking={ranking} selectedName={selectedName} onSelect={onSelectFactor} />
    </section>
  );
}

function PanelHeader({ detail, selectedName }) {
  return (
    <div className="section-head factors-section-head">
      <div>
        <span className="section-kicker">详情</span>
        <h2>{detail ? factorTitle(detail) : selectedName || "请选择因子"}</h2>
      </div>
    </div>
  );
}

function FactorDefinition({ detail }) {
  return (
    <dl className="factors-dl">
      <DefinitionTerm label="英文/字段名" value={<code className="factors-code">{detail.name}</code>} />
      <DefinitionTerm label="说明" value={detail.description} />
      <DefinitionTerm label="公式" value={<code className="factors-formula">{detail.formula}</code>} />
      <DefinitionTerm label="周期" value={(detail.timeframes || []).join(", ") || "—"} />
      <DefinitionTerm label="方向" value={directionLabel(detail.direction)} />
      <DefinitionTerm label="源码" value={<code className="factors-code">{detail.sourceFile}</code>} />
    </dl>
  );
}

function DefinitionTerm({ label, value }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function BacktestBlock({ backtest, error, loading, onRun }) {
  return (
    <div className="factors-backtest-block">
      <h3 className="factors-subhead">单因子回测</h3>
      <button type="button" disabled={loading} onClick={onRun}>
        {loading ? "计算中…" : "运行回测"}
      </button>
      {error ? <p className="factors-error">{error}</p> : null}
      {backtest ? <BacktestMetrics backtest={backtest} /> : null}
    </div>
  );
}

function BacktestMetrics({ backtest }) {
  return (
    <dl className="factors-metrics">
      <MetricItem label="综合评分" value={formatNum(backtest.factorScore, 1)} />
      <MetricItem label="相关性" value={formatPct(backtest.avgAbsCorrelation, 1)} />
      <MetricItem label="样本期数" value={backtest.totalPeriods} />
      <MetricItem label="IC 均值" value={formatNum(backtest.icMean, 6)} />
      <MetricItem label="IC 标准差" value={formatNum(backtest.icStd, 6)} />
      <MetricItem label="IR" value={formatNum(backtest.ir, 4)} />
      <MetricItem label="IC>0 占比" value={formatNum(backtest.icPositiveRate, 4)} />
      <MetricItem label="多空收益" value={formatNum(backtest.longShortReturn, 6)} />
      <MetricItem label="因子夏普" value={formatNum(backtest.sharpe, 4)} />
      <MetricItem label="胜率" value={formatPct(backtest.winRate, 1)} />
      <MetricItem label="贡献度" value={formatPct(backtest.contribution, 1)} />
      <MetricItem label="最大回撤" value={formatPct(backtest.maxDrawdown, 2)} />
      <MetricItem label="盈亏比" value={formatNum(backtest.profitFactor, 4)} />
      <MetricItem label="换手" value={formatNum(backtest.turnover, 4)} />
      <MetricItem label="t 统计量" value={formatNum(backtest.tStat, 4)} />
      <MetricItem label="p 值" value={formatNum(backtest.pValue, 6)} />
    </dl>
  );
}

function MetricItem({ label, value }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function RankingBlock({ ranking, selectedName, onSelect }) {
  return (
    <div className="factors-ranking-block">
      <h3 className="factors-subhead">按综合评分排序（后台缓存 · 当前筛选与交易对）</h3>
      <div className="factors-ranking-wrap">
        <table className="factors-table factors-ranking-table">
          <thead>
            <tr>
              <th>#</th>
              <th>中文因子</th>
              <th>类别</th>
              <th>评分</th>
              <th>相关性</th>
              <th>贡献</th>
              <th>夏普</th>
              <th>胜率</th>
              <th>IR</th>
              <th>IC 均值</th>
            </tr>
          </thead>
          <tbody>
            {ranking
              .slice(0, RANKING_LIMIT)
              .map((row, index) => renderRankingRow({ row, index, selectedName, onSelect }))}
          </tbody>
        </table>
        {!ranking.length ? <p className="factors-empty factors-ranking-empty">暂无排名数据</p> : null}
      </div>
    </div>
  );
}

function renderRankingRow({ row, index, selectedName, onSelect }) {
  return (
    <tr
      key={row.factorName}
      className={selectedName === row.factorName ? "factors-row-selected" : ""}
      onClick={() => onSelect(row.factorName)}
    >
      <td>{index + 1}</td>
      <td>
        <strong className="factors-name-cn">{factorTitle(row)}</strong>
        <code className="factors-code">{row.factorName}</code>
      </td>
      <td>{row.categoryName || row.category}</td>
      <td>{formatNum(row.factorScore, 1)}</td>
      <td>{formatPct(row.avgAbsCorrelation, 1)}</td>
      <td>{formatPct(row.contribution, 1)}</td>
      <td>{formatNum(row.sharpe, 2)}</td>
      <td>{formatPct(row.winRate, 0)}</td>
      <td>{formatNum(row.ir, 4)}</td>
      <td>{formatNum(row.icMean, 4)}</td>
    </tr>
  );
}
