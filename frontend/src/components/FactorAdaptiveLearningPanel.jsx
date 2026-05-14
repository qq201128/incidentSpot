import "./FactorAdaptiveLearningPanel.css";

export default function FactorAdaptiveLearningPanel({ learning, lstm, lstmStatus }) {
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

      <LstmShadowCard lstm={lstm} statusText={lstmStatus} />

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

function LstmShadowCard({ lstm, statusText }) {
  const shadow = lstm?.shadow || lstm || {};
  return (
    <div className={`factor-lstm-card ${statusClass(shadow.status)}`}>
      <div className="factor-lstm-card-head">
        <div>
          <span className="section-kicker">LSTM 影子策略</span>
          <h4>{lstmStatusLabel(shadow.status)}</h4>
          <p>{statusText || shadow.reason || "等待模型训练状态。"}</p>
        </div>
        <span>{shadow.strategyKey || "factor_lstm_shadow"}</span>
      </div>
      <div className="factor-lstm-grid">
        <Metric label="预测状态" value={readinessLabel(shadow)} strong={shadow.shadowPredictionReady} />
        <Metric label="阻断原因" value={blockedReasonLabel(shadow.shadowPredictionBlockedReason)} />
        <Metric label="Torch" value={shadow.torchAvailable ? "可用" : "不可用"} />
        <Metric label="模型版本" value={shortVersion(shadow.modelVersion)} />
        <Metric label="最近训练" value={formatDate(shadow.trainedAt)} />
        <Metric label="训练样本" value={shadow.sampleCounts?.train ?? "—"} />
        <Metric label="测试准确率" value={formatPct(shadow.testAccuracy, 1)} />
        <Metric label="模拟胜率" value={formatPct(shadow.winRate, 1)} strong />
        <Metric label="最近胜率" value={formatPct(shadow.recentWinRate, 1)} />
      </div>
      <LstmComparison rows={shadow.comparison} />
    </div>
  );
}

function LstmComparison({ rows }) {
  const items = Array.isArray(rows) ? rows : [];
  if (!items.length) return null;
  return (
    <div className="factor-lstm-compare">
      {items.map((row) => (
        <span key={row.strategyKey}>
          <small>{compareLabel(row.strategyKey)}</small>
          <b>{formatPct(row.winRate, 1)}</b>
        </span>
      ))}
    </div>
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

function lstmStatusLabel(status) {
  if (status === "training") return "训练中";
  if (status === "trained") return "已训练";
  if (status === "insufficient_samples") return "样本不足";
  if (status === "failed") return "训练失败";
  return "未训练";
}

function readinessLabel(shadow) {
  if (shadow.shadowPredictionReady) return "可模拟下单";
  if (!shadow.status || shadow.status === "untrained") return "未就绪";
  return "已阻断";
}

function blockedReasonLabel(reason) {
  const labels = {
    passed: "—",
    torch_unavailable: "Torch不可用",
    artifacts_incomplete: "模型文件不完整",
    trained_combo_snapshot_missing: "训练组合快照缺失",
    trained_combo_snapshot_incomplete: "训练组合不足Top3",
    current_combo_snapshot_missing: "当前组合排名缺失",
    current_combo_snapshot_incomplete: "当前组合不足Top3",
    combo_snapshot_mismatch: "组合排名已变化",
  };
  return labels[reason] || reason || "—";
}

function statusClass(status) {
  return status === "trained" ? "is-trained" : status === "failed" ? "is-failed" : "";
}

function compareLabel(strategyKey) {
  if (strategyKey?.includes("top2")) return "Top2";
  if (strategyKey?.includes("top3")) return "Top3";
  if (strategyKey?.startsWith("factor_lstm_shadow")) return "LSTM";
  return "Top1";
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

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString();
}

function shortVersion(value) {
  if (!value) return "—";
  return String(value).replace(/^lstm_/, "").slice(0, 24);
}
