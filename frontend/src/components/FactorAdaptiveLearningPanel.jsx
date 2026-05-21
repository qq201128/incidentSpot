import ModelFamilyBoard from "./ModelFamilyBoard";
import "./FactorAdaptiveLearningPanel.css";

export default function FactorAdaptiveLearningPanel({
  learning,
  lstm,
  lstmStatus,
  onSearchCandidates,
  searchStatus,
}) {
  const algorithms = Array.isArray(learning?.algorithms) ? learning.algorithms : [];
  const modelRows = modelFamilyRows(lstm);
  const summary = modelFamilySummary(modelRows);
  return (
    <section className="factor-adaptive-panel">
      <div className="factor-adaptive-head">
        <div>
          <span className="section-kicker">多模型族独立系统</span>
          <h3>{statusLabel(learning?.status)}</h3>
          <p>{lstmStatus || learning?.insight?.message || "等待模型族状态加载。"}</p>
        </div>
        <span className="factor-adaptive-badge">Ready {summary.ready}/{summary.total}</span>
      </div>

      <div className="factor-adaptive-summary">
        <Metric label="总体准确率" value={formatPct(learning?.overallAccuracy, 0)} strong />
        <Metric label="学习样本" value={learning?.sampleCount ?? "—"} />
        <Metric label="搜索中" value={summary.searching} />
        <Metric label="候选记录" value={summary.candidates} />
      </div>

      <ModelFamilyBoard
        families={modelRows}
        onSearchCandidates={onSearchCandidates}
        searchStatus={searchStatus}
        onRescanCandidates={(family) => onSearchCandidates?.(family, { resetHistory: true })}
      />

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

function modelFamilyRows(lstm) {
  if (Array.isArray(lstm?.families)) return lstm.families;
  return [lstm?.shadow || lstm].filter(Boolean);
}

function modelFamilySummary(rows) {
  const total = rows.length;
  const ready = rows.filter((row) => predictionReady(row)).length;
  const searching = rows.filter((row) => ["queued", "running"].includes(row?.candidateSearchProgress?.status)).length;
  const candidates = rows.reduce((sum, row) => sum + Number(row?.candidateLibrary?.total || 0), 0);
  return { total, ready, searching, candidates };
}

function predictionReady(row) {
  return Boolean(row?.shadowPredictionReady || row?.shadowPredictionBlockedReason === "combo_snapshot_mismatch");
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
