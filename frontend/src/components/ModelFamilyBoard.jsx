import {
  blockedReasonLabel,
  comboStatusLabel,
  compareLabel,
  formatDate,
  formatNum,
  formatPct,
  isTrainedStatus,
  modelFamilyLabel,
  publishedSnapshotLabel,
  recentSnapshotLabel,
  shortVersion,
  statusClass,
  statusLabel,
  topCountLabel,
} from "./ModelFamilyBoardLabels";
import "./ModelFamilyBoard.css";

export default function ModelFamilyBoard({ families, onSearchCandidates, searchStatus }) {
  if (!families.length) return <p className="factor-learning-empty small">模型族状态加载中</p>;
  const searchState = normalizedSearchState(searchStatus);
  const allRunning = searchState.status === "running" && searchState.family === "__all__";
  return (
    <div className="factor-model-board">
      <div className="factor-model-board-head">
        <span>训练规则：validation 与 test 胜率都必须 &gt; 70%，全量 search grid 并行执行。</span>
        <button
          type="button"
          className="factor-lstm-search-button"
          disabled={allRunning || !onSearchCandidates}
          onClick={() => onSearchCandidates?.("__all__")}
        >
          {allRunning ? "全部排队中" : "全量搜索全部算法"}
        </button>
      </div>
      <div className="factor-model-card-list">
        {families.map((shadow) => (
          <ModelShadowCard
            key={shadow.modelFamily || shadow.strategyKey}
            shadow={shadow}
            onSearchCandidates={onSearchCandidates}
            searchStatus={searchState}
          />
        ))}
      </div>
    </div>
  );
}

function ModelShadowCard({ shadow, onSearchCandidates, searchStatus }) {
  const ready = predictionReady(shadow);
  const progress = shadow.candidateSearchProgress || {};
  const library = shadow.candidateLibrary || {};
  const rules = shadow.trainingRules || {};
  const searchActive = candidateSearchActive(progress, searchStatus, shadow.modelFamily);
  return (
    <div className={`factor-lstm-card ${statusClass(shadow.status)}`}>
      <ModelCardHead
        shadow={shadow}
        searchActive={searchActive}
        searchLabel={candidateSearchLabel(progress, searchStatus)}
        onSearchCandidates={onSearchCandidates}
      />
      <CandidateProgress progress={progress} />
      <div className="factor-lstm-grid">
        <Metric label="预测状态" value={readinessLabel(shadow)} strong={ready} />
        <Metric label="运行闸门" value={gateLabel(shadow)} strong={ready} />
        <Metric label="组合状态" value={comboStatusLabel(shadow)} />
        <Metric label="当前Top" value={topCountLabel(shadow.comboSnapshotCurrent)} />
        <Metric label="发布快照" value={publishedSnapshotLabel(shadow)} />
        <Metric label="最近快照" value={recentSnapshotLabel(shadow)} />
        <Metric label="Active状态" value={statusLabel(shadow.activeModelStatus || shadow.status)} strong={isTrainedStatus(shadow.status)} />
        <Metric label="最近尝试" value={statusLabel(shadow.lastAttemptStatus)} />
        <Metric label="候选库" value={library.total ?? "—"} />
        <Metric label="搜索空间" value={rules.searchSpaceTotal ?? progress.searchSpaceTotal ?? progress.total ?? "—"} />
        <Metric label="并发" value={rules.parallelWorkers ?? progress.parallelWorkers ?? "—"} />
        <Metric label="胜率门槛" value=">70%" strong />
        <Metric label="置信阈值" value={formatNum(shadow.selectedConfidenceThreshold, 2)} />
        <Metric label="依赖" value={shadow.dependencyAvailable ?? shadow.torchAvailable ? "可用" : "不可用"} />
        <Metric label="模型版本" value={shortVersion(shadow.modelVersion)} />
        <Metric label="最近训练" value={formatDate(shadow.trainedAt)} />
        <Metric label="训练样本" value={shadow.sampleCounts?.train ?? "—"} />
        <Metric label="测试准确率" value={formatPct(shadow.testAccuracy, 1)} />
        <Metric label="模拟胜率" value={formatPct(shadow.winRate, 1)} strong />
        <Metric label="最近胜率" value={formatPct(shadow.recentWinRate, 1)} />
      </div>
      <ComparisonRows rows={shadow.comparison} />
    </div>
  );
}

function ModelCardHead({ shadow, searchActive, searchLabel, onSearchCandidates }) {
  return (
    <div className="factor-lstm-card-head">
      <div>
        <span className="section-kicker">{modelFamilyLabel(shadow.modelFamily)} 影子策略</span>
        <h4>{statusLabel(shadow.status)}</h4>
        <p>{modelFamilyCardText(shadow)}</p>
      </div>
      <div className="factor-lstm-card-actions">
        <span>{shadow.strategyKey || "factor_lstm_shadow"}</span>
        <button
          type="button"
          className="factor-lstm-search-button"
          disabled={searchActive || !onSearchCandidates}
          onClick={() => onSearchCandidates?.(shadow.modelFamily || "lstm")}
        >
          {searchLabel}
        </button>
      </div>
    </div>
  );
}

function CandidateProgress({ progress }) {
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
        <span>基线 {counts.initialBaseline ?? 0}</span>
        <span>未过 {counts.validationFailed ?? 0}</span>
      </div>
      {progress.latestCompleted ? <p>{candidateText(progress.latestCompleted)}</p> : null}
    </div>
  );
}

function ComparisonRows({ rows }) {
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

function Metric({ label, value, strong = false }) {
  return (
    <span className={`factor-adaptive-metric${strong ? " strong" : ""}`}>
      <b>{value}</b>
      <small>{label}</small>
    </span>
  );
}

function normalizedSearchState(searchStatus) {
  return typeof searchStatus === "string" ? { status: searchStatus } : (searchStatus || {});
}

function candidateSearchActive(progress, searchState, family) {
  return ["queued", "running"].includes(progress.status) ||
    (searchState.status === "running" && (!searchState.family || family === searchState.family));
}

function candidateSearchLabel(progress, searchState) {
  if (searchState.status === "running" || progress.status === "queued") return "排队中";
  return progress.status === "running" ? "搜索中" : "开始搜索";
}

function modelFamilyCardText(shadow) {
  if (predictionReady(shadow)) return "影子预测已就绪，可进入模拟下单链路。";
  if (shadow.validationFailureReason) return `验证未通过：${shadow.validationFailureReason}`;
  const reason = blockedReasonLabel(shadow.shadowPredictionBlockedReason);
  return reason === "—" ? "等待训练或候选搜索结果。" : `阻断：${reason}`;
}

function candidateText(candidate) {
  const cfg = candidate.config || {};
  return `最近：${statusLabel(candidate.status)} · w${cfg.featureWindow ?? "—"} · ${cfg.minMoveBps ?? "—"}bp · ${cfg.epochs ?? "—"}轮`;
}

function readinessLabel(shadow) {
  if (predictionReady(shadow)) return "可模拟下单";
  return !shadow.status || shadow.status === "untrained" ? "未就绪" : "已阻断";
}

function gateLabel(shadow) {
  return predictionReady(shadow) ? "未阻断" : blockedReasonLabel(shadow.shadowPredictionBlockedReason);
}

function predictionReady(shadow) {
  return Boolean(shadow.shadowPredictionReady || shadow.shadowPredictionBlockedReason === "combo_snapshot_mismatch");
}
