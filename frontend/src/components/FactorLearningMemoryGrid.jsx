import {
  factorLabel,
  learningPatternLabel,
} from "../utils/factorLearningLabels";

export default function FactorLearningMemoryGrid({ memory }) {
  return (
    <div className="factor-learning-grid">
      <PatternBox
        title="成功模式"
        items={memory?.factorMining?.successPatterns || []}
        valueKey="label"
        metaKey="support"
      />
      <PatternBox
        title="禁区"
        items={memory?.factorMining?.forbiddenRegions || []}
        valueKey="region"
        metaKey="avgAbsCorrelation"
      />
      <LossPatternBox lossMemory={memory?.lossMemory || {}} />
      <WeightsBox weights={memory?.weights || {}} />
    </div>
  );
}

function LossPatternBox({ lossMemory }) {
  const items = lossMemory.patterns || [];
  return (
    <section className="factor-learning-box">
      <div className="factor-learning-title compact">
        <h3>亏损特征</h3>
        <span>{items.length}</span>
      </div>
      <ul>
        <MetricRow label="状态" value={lossMemoryStatusLabel(lossMemory.status)} />
        <MetricRow label="样本" value={lossMemory.sampleCount ?? "—"} />
        <MetricRow label="亏损" value={lossMemory.lossCount ?? "—"} />
        {items.map((item, index) => (
          <li key={`亏损特征-${index}`}>
            <strong>{learningPatternLabel(item, "feature")}</strong>
            <span>{metaLabel("lossRate", item.lossRate)}</span>
          </li>
        ))}
      </ul>
      {!items.length ? (
        <p className="factor-learning-empty small">{lossMemoryEmptyText(lossMemory.status)}</p>
      ) : null}
    </section>
  );
}

function PatternBox({ title, items, valueKey, metaKey }) {
  return (
    <section className="factor-learning-box">
      <div className="factor-learning-title compact">
        <h3>{title}</h3>
        <span>{items.length}</span>
      </div>
      <ul>
        {items.map((item, index) => (
          <li key={`${title}-${index}`}>
            <strong>{learningPatternLabel(item, valueKey)}</strong>
            <span>{metaLabel(metaKey, item[metaKey])}</span>
          </li>
        ))}
      </ul>
      {!items.length ? <p className="factor-learning-empty small">暂无数据</p> : null}
    </section>
  );
}

function WeightsBox({ weights }) {
  const rows = Object.entries(weights)
    .sort((a, b) => Number(b[1]) - Number(a[1]));
  return (
    <section className="factor-learning-box">
      <div className="factor-learning-title compact">
        <h3>自动权重</h3>
        <span>{Object.keys(weights).length}</span>
      </div>
      <ul>
        {rows.map(([name, value]) => (
          <li key={name}>
            <strong>{factorLabel(name)}</strong>
            <span>{formatPct(value, 1)}</span>
          </li>
        ))}
      </ul>
      {!rows.length ? <p className="factor-learning-empty small">暂无权重</p> : null}
    </section>
  );
}

function MetricRow({ label, value }) {
  return (
    <li>
      <strong>{label}</strong>
      <span>{value}</span>
    </li>
  );
}

function lossMemoryStatusLabel(status) {
  if (status === "learned") return "已学习";
  if (status === "no_separable_loss_pattern") return "无可分离模式";
  if (status === "insufficient_loss_or_win_samples") return "盈亏样本不足";
  if (status === "insufficient_settled_predictions") return "结算样本不足";
  return status || "—";
}

function lossMemoryEmptyText(status) {
  if (status === "no_separable_loss_pattern") return "已复盘，但候选因子未达到亏损提升阈值";
  if (status === "insufficient_loss_or_win_samples") return "亏损或盈利样本不足，暂不学习模式";
  if (status === "insufficient_settled_predictions") return "缺少可对齐的已结算预测";
  return "暂无亏损特征";
}

function metaLabel(key, value) {
  if (key === "lossRate" || key === "avgAbsCorrelation") return formatPct(value, 1);
  return value ?? "—";
}

function formatPct(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}
