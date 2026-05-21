import { useCallback, useEffect, useMemo, useState } from "react";
import {
  confirmEnsembleStage,
  fetchEnsembleRanking,
  fetchEnsembleStatus,
  refreshEnsemble,
} from "../api/client";

const STAGE_LABELS = {
  observe: "观察中",
  weight_ready: "可启用降权",
  ensemble_ready: "可模拟综合策略",
};

export default function EnsembleJudgePanel({ symbol, duration, onConfirmed }) {
  const [status, setStatus] = useState(null);
  const [ranking, setRanking] = useState([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const canConfirm = Boolean(
    status?.updatedAt &&
      status?.recommendedStage &&
      status.recommendedStage !== "observe" &&
      status.recommendedStage !== status.confirmedStage,
  );
  const simulationState = useMemo(() => _simulationState(status), [status]);

  const load = useCallback(async () => {
    setError("");
    const [nextStatus, nextRanking] = await Promise.all([
      fetchEnsembleStatus(symbol, duration),
      fetchEnsembleRanking(symbol, duration),
    ]);
    setStatus(nextStatus);
    setRanking(Array.isArray(nextRanking?.ranking) ? nextRanking.ranking : []);
  }, [duration, symbol]);

  useEffect(() => {
    let stopped = false;
    setLoading(true);
    load()
      .catch((err) => {
        if (!stopped) setError(_errorMessage(err, "读取综合裁判失败"));
      })
      .finally(() => {
        if (!stopped) setLoading(false);
      });
    return () => {
      stopped = true;
    };
  }, [load]);

  async function handleRefresh() {
    setBusy("refresh");
    setError("");
    try {
      const data = await refreshEnsemble(symbol, duration);
      setStatus(data?.status || null);
      setRanking(Array.isArray(data?.ranking) ? data.ranking : []);
    } catch (err) {
      setError(_errorMessage(err, "刷新综合裁判失败"));
    } finally {
      setBusy("");
    }
  }

  async function handleConfirm() {
    if (!canConfirm) return;
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

  if (loading) return <div className="ensemble-card">正在读取综合裁判...</div>;

  return (
    <div className="ensemble-card">
      <div className="ensemble-head">
        <div>
          <strong>综合裁判</strong>
          <span>{duration} · {STAGE_LABELS[status?.stage] || "观察中"}</span>
        </div>
        <button type="button" onClick={() => void handleRefresh()} disabled={busy === "refresh"}>
          {busy === "refresh" ? "刷新中" : "刷新"}
        </button>
      </div>
      <div className="ensemble-stage-grid">
        <StagePill label="系统推荐" value={STAGE_LABELS[status?.recommendedStage] || "--"} />
        <StagePill label="人工确认" value={STAGE_LABELS[status?.confirmedStage] || "未确认"} />
        <StagePill label="模拟状态" value={simulationState} />
      </div>
      <p className="ensemble-reason">{status?.recommendationReason || "等待刷新统计"}</p>
      {canConfirm && (
        <button
          type="button"
          className="ensemble-confirm-btn"
          onClick={() => void handleConfirm()}
          disabled={busy === "confirm"}
        >
          {busy === "confirm" ? "确认中" : `确认：${STAGE_LABELS[status.recommendedStage]}`}
        </button>
      )}
      <RankingTable rows={ranking} />
      {!!error && <div className="predict-error">{error}</div>}
    </div>
  );
}

function StagePill({ label, value }) {
  return (
    <div className="ensemble-stage-pill">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function RankingTable({ rows }) {
  if (!rows.length) return <div className="strategy-empty">暂无已结算候选信号</div>;
  return (
    <div className="ensemble-ranking">
      {rows.slice(0, 6).map((row) => (
        <div key={row.signalKey} className="ensemble-ranking-row">
          <div>
            <strong>{row.signalKey}</strong>
            <span>{_typeLabel(row.signalType)} · {_badges(row).join(" / ") || "稳定观察"}</span>
          </div>
          <div className="ensemble-metrics">
            <span>{row.sampleCount}样本</span>
            <span>{_pct(row.winRate)}</span>
            <span>PF {_num(row.profitFactor)}</span>
            <span>权重 {_num(row.weightSuggestion)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function _simulationState(status) {
  if (status?.confirmedStage === "ensemble_ready") return "模拟可用";
  if (status?.confirmedStage === "weight_ready") return "观察中";
  return "未启用";
}

function _badges(row) {
  const badges = [];
  if (row.lowSample) badges.push("低样本");
  if (row.insufficientSample && !row.lowSample) badges.push("样本不足");
  if (row.weakSignal) badges.push("近期走弱");
  if (Number(row.consecutiveLosses) >= 5) badges.push("连续亏损");
  if (row.degraded && !row.weakSignal) badges.push("恢复中");
  return badges;
}

function _typeLabel(type) {
  if (type === "factor_combo") return "多因子";
  if (type === "high_winrate_combo") return "高胜率";
  if (type === "model_family") return "模型族";
  return "其他";
}

function _pct(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : "--";
}

function _num(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(2) : "--";
}

function _errorMessage(err, fallback) {
  return err?.response?.data?.detail || err?.message || fallback;
}
