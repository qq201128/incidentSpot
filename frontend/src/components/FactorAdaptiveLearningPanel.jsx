import "./FactorAdaptiveLearningPanel.css";

export default function FactorAdaptiveLearningPanel({ learning }) {
  const algorithms = Array.isArray(learning?.algorithms) ? learning.algorithms : [];
  const activeCount = learning?.activeAlgorithmCount ?? 0;
  return (
    <section className="factor-adaptive-panel">
      <div className="factor-adaptive-head">
        <div>
          <span className="section-kicker">自适应学习系统</span>
          <h3>{statusLabel(learning?.status)}</h3>
          <p>{learning?.insight?.message || "等待因子学习记忆生成。"}</p>
        </div>
        <span className="factor-adaptive-badge">学习中 {activeCount}/{learning?.algorithmCount ?? algorithms.length}</span>
      </div>

      <div className="factor-adaptive-summary">
        <Metric label="总体准确率" value={formatPct(learning?.overallAccuracy, 0)} strong />
        <Metric label="学习样本" value={learning?.sampleCount ?? "—"} />
      </div>

      <div className="factor-adaptive-insight">
        <strong>智能学习洞察：</strong>
        <span>{learning?.insight?.bestAlgorithm ? `最佳算法：${learning.insight.bestAlgorithm}` : "暂无最佳算法"}</span>
        {durationText(learning) ? <span>{durationText(learning)}</span> : null}
      </div>

      <div className="factor-adaptive-algorithms">
        <h4>算法权重</h4>
        {algorithms.map((algorithm) => (
          <AlgorithmRow key={algorithm.key} algorithm={algorithm} />
        ))}
        {!algorithms.length ? <p className="factor-learning-empty small">暂无算法评分</p> : null}
      </div>
    </section>
  );
}

function AlgorithmRow({ algorithm }) {
  const pct = Math.round(Number(algorithm.weight || 0) * 100);
  return (
    <div className="factor-adaptive-row">
      <span>{algorithm.label}</span>
      <div className="factor-adaptive-bar" aria-label={`${algorithm.label} 权重 ${pct}%`}>
        <i style={{ width: `${pct}%` }} />
      </div>
      <b>{algorithm.status === "active" ? `${pct}%` : `样本>${algorithm.sampleFloor}`}</b>
    </div>
  );
}

function Metric({ label, value, strong = false }) {
  return (
    <span className={`factor-adaptive-metric${strong ? " strong" : ""}`}>
      <b>{value}</b>
      <small>{label}</small>
    </span>
  );
}

function statusLabel(status) {
  if (status === "learning") return "学习中";
  if (status === "warning") return "低胜率预警";
  if (status === "insufficient_data") return "等待样本";
  return "未初始化";
}

function durationText(learning) {
  const first = learning?.durationPreference?.[0];
  if (!first) return "";
  return `周期偏好：${first.duration} ${formatPct(first.score, 0)}`;
}

function formatPct(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}
