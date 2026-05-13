import {
  factorLabel,
  learningPatternLabel,
} from "../utils/factorLearningLabels";

const TOP_WEIGHT_LIMIT = 10;

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
      <PatternBox
        title="亏损特征"
        items={memory?.lossMemory?.patterns || []}
        valueKey="feature"
        metaKey="lossRate"
      />
      <WeightsBox weights={memory?.weights || {}} />
    </div>
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
        {items.slice(0, 8).map((item, index) => (
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
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, TOP_WEIGHT_LIMIT);
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

function metaLabel(key, value) {
  if (key === "lossRate" || key === "avgAbsCorrelation") return formatPct(value, 1);
  return value ?? "—";
}

function formatPct(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}
