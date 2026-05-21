import { factorLabel, learningPatternLabel } from "../../utils/factorLearningLabels";
import { formatPct } from "./miningFormatters";

export default function MiningSidebar({ sidebar, operators, ingestionPath }) {
  const loop = sidebar?.closedLoop || {};
  const weights = Object.entries(sidebar?.weights || {}).sort((a, b) => Number(b[1]) - Number(a[1]));

  return (
    <div className="mining-insights-grid">
      <section className="mining-side-card mining-side-card--loop">
        <h3>闭环状态</h3>
        <div className="mining-loop-grid">
          <LoopMetric label="Agent入库" value={loop.agentIngested} />
          <LoopMetric label="组合回测" value={loop.comboBacktest} />
          <LoopMetric
            label="候选晋升"
            value={`${loop.candidatePromoted ?? 0}/${loop.candidateTotal ?? 0}`}
          />
          <LoopMetric label="监控告警" value={loop.monitorAlerts} warn={loop.monitorAlerts > 0} />
          <LoopMetric label="失败帧" value={loop.frameFailures} warn={loop.frameFailures > 0} />
          <LoopMetric label="本地复盘源" value={loop.replaySource} small />
        </div>
        {(sidebar?.alerts || []).map((alert, index) => (
          <div key={`${alert.message}-${index}`} className={`mining-alert is-${alert.level}`}>
            {alert.message}
            {alert.level === "error" && alert.detail ? (
              <button type="button" className="mining-alert-link">
                查看详情
              </button>
            ) : null}
          </div>
        ))}
      </section>

      <section className="mining-side-card mining-side-card--patterns">
        <h3>成功模式 / 禁区</h3>
        <div className="mining-pattern-scroll">
          <div className="mining-pattern-block">
            <h4>成功模式</h4>
            <ul>
              {(sidebar?.successPatterns || []).map((item, index) => (
                <li key={`ok-${index}`}>
                  <i className="is-ok" />
                  {learningPatternLabel(item, "label")}
                </li>
              ))}
              {!sidebar?.successPatterns?.length ? <li className="mining-muted">暂无</li> : null}
            </ul>
          </div>
          <div className="mining-pattern-block">
            <h4>禁区</h4>
            <ul>
              {(sidebar?.forbiddenRegions || []).map((item, index) => (
                <li key={`bad-${index}`}>
                  <i className="is-bad" />
                  {learningPatternLabel(item, "region")}
                </li>
              ))}
              {!sidebar?.forbiddenRegions?.length ? <li className="mining-muted">暂无</li> : null}
            </ul>
          </div>
        </div>
      </section>

      <section className="mining-side-card mining-side-card--weights">
        <h3>自动权重（当前）</h3>
        <div className="mining-weight-scroll">
          {weights.map(([key, value]) => (
            <WeightBar key={key} label={weightLabel(key)} value={Number(value)} />
          ))}
          {!weights.length ? <p className="mining-muted">暂无权重</p> : null}
        </div>
      </section>

      <section className="mining-side-card mining-side-card--operators">
        <h3>
          运算算子库 <b>{operators?.total ?? sidebar?.operatorTotal ?? 0}</b> 个
        </h3>
        <div className="mining-operator-scroll">
          {(operators?.preview || []).map((group) => (
            <div key={group.category} className="mining-operator-group">
              <span>{group.category}</span>
              <div className="mining-operator-tags">
                {group.names.map((name) => (
                  <em key={name}>{name}</em>
                ))}
              </div>
            </div>
          ))}
        </div>
        <footer className="mining-side-link">查看全部运算算子库 →</footer>
      </section>

      <section className="mining-side-card mining-side-card--path">
        <h3>入库路径</h3>
        <div className="mining-ingestion-path" role="list">
          {ingestionPath.map((step, index) => (
            <IngestionStep key={step.key} step={step} showConnector={index < ingestionPath.length - 1} />
          ))}
        </div>
      </section>
    </div>
  );
}

function LoopMetric({ label, value, warn = false, small = false }) {
  return (
    <div className={`mining-loop-metric${warn ? " is-warn" : ""}${small ? " is-small" : ""}`}>
      <span>{label}</span>
      <strong>{value ?? "—"}</strong>
    </div>
  );
}

function WeightBar({ label, value }) {
  const pct = Math.round(value * 100);
  return (
    <div className="mining-weight-row">
      <span>{label}</span>
      <div className="mining-weight-bar">
        <i style={{ width: `${pct}%` }} />
      </div>
      <b>{formatPct(value, 2)}</b>
    </div>
  );
}

const WEIGHT_LABELS = {
  trend: "趋势",
  volatility: "波动",
  liquidity: "流动性",
  shadow: "影子",
  momentum: "动量",
  structure: "结构",
};

function weightLabel(key) {
  if (WEIGHT_LABELS[key]) return WEIGHT_LABELS[key];
  if (String(key).includes("_")) return factorLabel(key);
  return key;
}

function IngestionStep({ step, showConnector }) {
  return (
    <div className={`mining-ingestion-step is-${step.state}`} role="listitem">
      <span className="mining-ingestion-icon" aria-hidden />
      <strong>{step.label}</strong>
      <small>{step.detail}</small>
      {showConnector ? <span className="mining-ingestion-connector" aria-hidden /> : null}
    </div>
  );
}
