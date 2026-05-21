import { useCallback, useEffect, useMemo, useState } from "react";
import { confirmEnsembleStage, fetchEnsembleStatus, refreshEnsemble } from "../api/client";
import "./EnsembleJudgePanel.css";

const STAGE_LABELS = {
  observe: "观察中",
  weight_ready: "可启用降权",
  ensemble_ready: "可模拟裁判",
};

const STAGE_HINTS = {
  observe: "样本或覆盖不足，暂不推荐升阶",
  weight_ready: "可对弱势信号降权，继续观察",
  ensemble_ready: "满足裁判模拟条件，可确认启用",
};

const SIGNAL_TYPE_LABELS = {
  factor_combo: "组合",
  high_winrate_combo: "高胜率",
  model_family: "模型族",
  factor_candidate: "候选",
};

export default function EnsembleJudgePanel({ symbol, duration, onConfirmed, onRefreshed }) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setError("");
    const nextStatus = await fetchEnsembleStatus(symbol, duration);
    setStatus(nextStatus);
  }, [duration, symbol]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    load()
      .catch((err) => {
        if (!cancelled) setError(_errorMessage(err, "读取综合裁判失败"));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [load]);

  const canConfirm = Boolean(
    status?.updatedAt &&
      status?.recommendedStage &&
      status.recommendedStage !== "observe" &&
      status.recommendedStage !== status.confirmedStage,
  );

  const stageSynced = Boolean(
    status?.confirmedStage && status.confirmedStage === status.recommendedStage,
  );

  const simulationState = useMemo(() => _simulationState(status), [status]);
  const durationLabel = useMemo(() => _durationLabel(duration), [duration]);

  async function handleRefresh() {
    setRefreshing(true);
    setError("");
    try {
      const data = await refreshEnsemble(symbol, duration);
      setStatus(data?.status || null);
      onRefreshed?.();
    } catch (err) {
      setError(_errorMessage(err, "刷新综合裁判失败"));
    } finally {
      setRefreshing(false);
    }
  }

  async function handleConfirm() {
    if (!canConfirm || !status?.recommendedStage) return;
    setBusy("confirm");
    setError("");
    try {
      const nextStatus = await confirmEnsembleStage(symbol, duration, status.recommendedStage);
      setStatus(nextStatus);
      onConfirmed?.();
    } catch (err) {
      setError(_errorMessage(err, "确认阶段失败"));
    } finally {
      setBusy("");
    }
  }

  if (loading) {
    return <div className="ensemble-card ensemble-card--loading">正在读取综合裁判…</div>;
  }

  return (
    <div className="ensemble-card">
      <header className="ensemble-head">
        <div>
          <strong className="ensemble-title">信号裁判</strong>
          <span className="ensemble-subtitle">
            {symbol} · {durationLabel}
            {status?.updatedAt ? ` · 更新 ${_formatTime(status.updatedAt)}` : ""}
          </span>
        </div>
        <button
          type="button"
          className="ensemble-refresh-btn"
          onClick={() => void handleRefresh()}
          disabled={refreshing || busy === "confirm"}
        >
          {refreshing ? "刷新中…" : "刷新裁判"}
        </button>
      </header>

      <div className="ensemble-stage-grid">
        <StagePill
          label="系统推荐"
          stage={status?.recommendedStage}
          value={STAGE_LABELS[status?.recommendedStage] || "—"}
          hint={STAGE_HINTS[status?.recommendedStage]}
        />
        <StagePill
          label="人工确认"
          stage={status?.confirmedStage || "none"}
          value={status?.confirmedStage ? STAGE_LABELS[status.confirmedStage] : "未确认"}
          hint={status?.confirmedAt ? `确认于 ${_formatTime(status.confirmedAt)}` : "待操作"}
        />
        <StagePill label="模拟状态" stage={simulationState} value={simulationState} />
      </div>

      <CoverageSummary coverage={status?.sampleCoverage} />

      <div className="ensemble-reason-block">
        <span className="ensemble-reason-label">推荐 / 阻断原因</span>
        <p className="ensemble-reason">{status?.recommendationReason || "请点击「刷新裁判」生成推荐"}</p>
      </div>
      <div className="ensemble-actions">
        {canConfirm ? (
          <button
            type="button"
            className="ensemble-confirm-btn"
            onClick={() => void handleConfirm()}
            disabled={busy === "confirm"}
          >
            {busy === "confirm" ? "确认中…" : `确认启用：${STAGE_LABELS[status.recommendedStage]}`}
          </button>
        ) : null}
        {stageSynced ? <p className="ensemble-hint ensemble-hint--ok">已与系统推荐阶段一致，无需重复确认</p> : null}
        {!canConfirm && !stageSynced && status?.recommendedStage === "observe" ? (
          <p className="ensemble-hint">当前仅观察：结算样本或交易日覆盖尚未达标</p>
        ) : null}
      </div>

      {!!error && <div className="predict-error">{error}</div>}
    </div>
  );
}

function StagePill({ label, stage, value, hint = "" }) {
  return (
    <div className={`ensemble-stage-pill ensemble-stage-pill--${_stageVariant(stage)}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {hint ? <small>{hint}</small> : null}
    </div>
  );
}

function CoverageSummary({ coverage }) {
  if (!coverage) return null;
  const sampleCount = coverage.sampleCount ?? 0;
  const days = coverage.distinctTradingDays ?? 0;
  const maxLoss = coverage.maxConsecutiveLosses ?? 0;
  const pfWarn = Boolean(coverage.recentProfitFactorBelowOne);
  const readyCount = coverage.readySignalTypeCount ?? _readySignalTypeCount(coverage.bySignalType);
  const requiredCount = coverage.requiredSignalTypeCount ?? Object.keys(SIGNAL_TYPE_LABELS).length;
  const sourceText = _sourceCoverageText(coverage.bySignalType);
  return (
    <div className="ensemble-coverage">
      <span>结算样本 {sampleCount}</span>
      <span>覆盖 {days} 个交易日</span>
      <span>最大连亏 {maxLoss}</span>
      <span>达标信号源 {readyCount}/{requiredCount}</span>
      {sourceText ? <span title={sourceText}>来源 {sourceText}</span> : null}
      {pfWarn ? <span className="is-warn">近期 PF&lt;1</span> : <span className="is-ok">近期 PF 正常</span>}
    </div>
  );
}

function _readySignalTypeCount(bySignalType = {}) {
  return Object.values(bySignalType).filter((item) => Number(item?.sampleCount || 0) >= 200).length;
}

function _sourceCoverageText(bySignalType = {}) {
  return Object.entries(SIGNAL_TYPE_LABELS)
    .map(([key, label]) => `${label} ${Number(bySignalType[key]?.sampleCount || 0)}`)
    .join(" / ");
}

function _stageVariant(stage) {
  if (stage === "ensemble_ready" || stage === "裁判模拟已启用") return "ready";
  if (stage === "weight_ready" || stage === "可启用降权") return "weight";
  if (stage === "observe" || stage === "观察中") return "observe";
  if (stage === "模拟可用") return "ready";
  if (stage === "未启用模拟") return "idle";
  return "neutral";
}

function _simulationState(status) {
  if (status?.confirmedStage === "ensemble_ready") return "裁判模拟已启用";
  if (status?.confirmedStage === "weight_ready") return "观察期";
  if (status?.confirmedStage === "observe") return "仅观察";
  return "未启用";
}

function _durationLabel(duration) {
  const map = { "10m": "10分钟", "30m": "30分钟", "60m": "60分钟", "1d": "1天" };
  return map[duration] || duration;
}

function _formatTime(value) {
  if (!value) return "—";
  const dt = new Date(value);
  if (!Number.isFinite(dt.getTime())) return String(value);
  return dt.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

function _errorMessage(err, fallback) {
  return err?.response?.data?.detail || err?.message || fallback;
}
