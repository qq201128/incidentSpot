import "./FactorAdaptiveLearningPanel.css";

export default function FactorAdaptiveLearningPanel({
  learning,
  lstm,
  lstmStatus,
  onSearchCandidates,
  searchStatus,
}) {
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

      <LstmShadowCard
        lstm={lstm}
        statusText={lstmStatus}
        onSearchCandidates={onSearchCandidates}
        searchStatus={searchStatus}
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

function LstmShadowCard({ lstm, statusText, onSearchCandidates, searchStatus }) {
  const shadow = lstm?.shadow || lstm || {};
  const ready = predictionReady(shadow);
  const progress = shadow.candidateSearchProgress || {};
  const searchActive = ["queued", "running"].includes(progress.status) || searchStatus === "running";
  const searchLabel =
    searchStatus === "running" || progress.status === "queued"
      ? "排队中"
      : progress.status === "running"
        ? "搜索中"
        : "开始搜索";
  return (
    <div className={`factor-lstm-card ${statusClass(shadow.status)}`}>
      <div className="factor-lstm-card-head">
        <div>
          <span className="section-kicker">LSTM 影子策略</span>
          <h4>{lstmStatusLabel(shadow.status)}</h4>
          <p>{statusText || shadow.reason || "等待模型训练状态。"}</p>
        </div>
        <div className="factor-lstm-card-actions">
          <span>{shadow.strategyKey || "factor_lstm_shadow"}</span>
          <button
            type="button"
            className="factor-lstm-search-button"
            disabled={searchActive || !onSearchCandidates}
            onClick={onSearchCandidates}
          >
            {searchLabel}
          </button>
        </div>
      </div>
      <LstmCandidateProgress progress={progress} />
      <div className="factor-lstm-grid">
        <Metric label="预测状态" value={readinessLabel(shadow)} strong={ready} />
        <Metric label="运行闸门" value={gateLabel(shadow)} strong={ready} />
        <Metric label="组合状态" value={comboStatusLabel(shadow.comboSnapshotReason)} />
        <Metric label="Active状态" value={lstmStatusLabel(shadow.activeModelStatus || shadow.status)} />
        <Metric label="最近尝试" value={lstmStatusLabel(shadow.lastAttemptStatus)} />
        <Metric label="验证失败" value={shadow.validationFailureReason || "—"} />
        <Metric label="置信阈值" value={formatNum(shadow.selectedConfidenceThreshold, 2)} />
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

function LstmCandidateProgress({ progress }) {
  if (!progress || progress.status === "idle") return null;
  const pct = Math.round(Number(progress.percent || 0) * 100);
  const counts = progress.counts || {};
  const active = ["queued", "running"].includes(progress.status);
  return (
    <div className={`factor-lstm-progress ${active ? "is-running" : ""}`}>
      <div className="factor-lstm-progress-head">
        <span>{progress.status === "queued" ? "候选搜索排队中" : progress.status === "running" ? "候选搜索进行中" : "最近候选搜索"}</span>
        <b>{progress.completed ?? 0}/{progress.total ?? 0} · {pct}%</b>
      </div>
      <div className="factor-lstm-progress-bar" aria-label={`候选搜索进度 ${pct}%`}>
        <i style={{ width: `${pct}%` }} />
      </div>
      <div className="factor-lstm-progress-meta">
        <span>并发 {progress.parallelWorkers ?? "—"}</span>
        <span>交易 {counts.tradeActive ?? 0}</span>
        <span>影子 {counts.shadowActive ?? 0}</span>
        <span>未过 {counts.validationFailed ?? 0}</span>
      </div>
      {progress.latestCompleted ? (
        <p>{candidateText(progress.latestCompleted)}</p>
      ) : null}
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

function candidateText(candidate) {
  const cfg = candidate.config || {};
  return `最近：${lstmStatusLabel(candidate.status)} · w${cfg.featureWindow ?? "—"} · ${cfg.minMoveBps ?? "—"}bp · ${cfg.epochs ?? "—"}轮`;
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
  if (status === "shadow_active") return "影子激活";
  if (status === "trade_active") return "交易激活";
  if (status === "promoted_shadow_active") return "已发布影子";
  if (status === "promoted_trade_active") return "已发布交易";
  if (status === "rejected_validation") return "验证拒绝";
  if (status === "rejected_insufficient_samples") return "样本拒绝";
  if (status === "queued") return "排队中";
  if (status === "training") return "训练中";
  if (status === "trained") return "已训练";
  if (status === "validation_failed") return "验证失败";
  if (status === "insufficient_samples") return "样本不足";
  if (status === "failed") return "训练失败";
  return "未训练";
}

function readinessLabel(shadow) {
  if (predictionReady(shadow)) return "可模拟下单";
  if (!shadow.status || shadow.status === "untrained") return "未就绪";
  return "已阻断";
}

function gateLabel(shadow) {
  if (predictionReady(shadow)) return "未阻断";
  return blockedReasonLabel(shadow.shadowPredictionBlockedReason);
}

function predictionReady(shadow) {
  return Boolean(shadow.shadowPredictionReady || isComboDriftOnly(shadow));
}

function isComboDriftOnly(shadow) {
  return shadow.shadowPredictionBlockedReason === "combo_snapshot_mismatch";
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
  };
  return labels[reason] || reason || "—";
}

function comboStatusLabel(reason) {
  const labels = {
    passed: "一致",
    trained_combo_snapshot_missing: "训练快照缺失",
    trained_combo_snapshot_incomplete: "训练快照不足",
    current_combo_snapshot_missing: "当前排名缺失",
    current_combo_snapshot_incomplete: "当前排名不足",
    combo_snapshot_mismatch: "已变化，继续学习",
  };
  return labels[reason] || reason || "—";
}

function statusClass(status) {
  if (["shadow_active", "trade_active", "trained"].includes(status)) return "is-trained";
  if (["failed", "validation_failed", "rejected_validation"].includes(status)) return "is-failed";
  return "";
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

function formatNum(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString();
}

function shortVersion(value) {
  if (!value) return "—";
  return String(value).replace(/^lstm_/, "");
}
