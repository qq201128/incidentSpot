import { useCallback, useEffect, useMemo, useState } from "react";
import { activateModelVersion, fetchModelDashboard, triggerModelTraining } from "../api/client";
import ModelCurrentCard from "./ModelCurrentCard";
import { errorText, formatMetric, formatTime, metricLabel, metricValue, modelLabel, runStatusText, statusText } from "./modelDisplay";
import "./ModelManager.css";

const REFRESH_MS = 10000;

function decisionFailureText(item) {
  const failed = item.decision?.failed;
  if (!Array.isArray(failed) || failed.length === 0) return null;
  return failed.map((x) => {
    if (x.path === "production_target") {
      const winRate = formatMetric(x.candidate?.winRate, "percent");
      const trades = formatMetric(x.candidate?.tradesPerDay);
      return `${metricLabel(x.label)} 未达标（胜率 ${winRate} / 每日 ${trades} 单）`;
    }
    return `${metricLabel(x.label)} 变差 ${formatMetric(x.worseBy)}`;
  }).join("；");
}

function versionBacktestItems(metrics) {
  const production = metrics?.production_gate_backtest;
  if (production) {
    return [
      ["生产胜率", formatMetric(production.win_rate, "percent")],
      ["生产每日", formatMetric(production.trades_per_day)],
    ];
  }
  return [
    ["回测胜率", formatMetric(metricValue(metrics, "backtest_test_split.win_rate"), "percent")],
  ];
}

function VersionRow({ item, activeKey, busyVersion, onActivate }) {
  const canActivate = item.status !== "active" && item.status !== "rejected";
  const busy = busyVersion === item.versionId;
  const failureText = decisionFailureText(item);
  const backtestItems = versionBacktestItems(item.metrics);
  return (
    <li className={`model-version ${item.status}`}>
      <div>
        <div className="version-head">
          <strong>{item.versionId}</strong>
          <span>{statusText(item.status)}</span>
        </div>
        <div className="version-metrics">
          <span>区分度 {formatMetric(item.metrics?.test_auc)}</span>
          <span>综合分数 {formatMetric(item.metrics?.test_f1)}</span>
          {backtestItems.map(([label, value]) => (
            <span key={label}>{label} {value}</span>
          ))}
          <span>{formatTime(item.createdAt)}</span>
        </div>
        {!!failureText && <p className="decision-note">{failureText}</p>}
      </div>
      <button
        type="button"
        className="version-activate"
        disabled={!canActivate || busy}
        onClick={() => onActivate(activeKey, item.versionId)}
      >
        {busy ? "切换中" : "启用"}
      </button>
    </li>
  );
}

function ModelHeader({ schedule, lastRun, trainingBusy, onTrain }) {
  return (
    <div className="model-manager-head">
      <div>
        <h2>模型管理</h2>
        <p>
          下次训练 {formatTime(schedule.nextRunAt)} / 最近运行 {runStatusText(lastRun?.status)}
        </p>
      </div>
      <button type="button" onClick={onTrain} disabled={trainingBusy || schedule.running}>
        {schedule.running ? "训练中" : trainingBusy ? "启动中" : "立即训练"}
      </button>
    </div>
  );
}

function ModelTabs({ models, selectedKey, onSelect }) {
  return (
    <div className="model-tabs">
      {models.map((item) => (
        <button
          type="button"
          key={item.key}
          className={item.key === selectedKey ? "active" : ""}
          onClick={() => onSelect(item.key)}
        >
          {modelLabel(item)}
        </button>
      ))}
    </div>
  );
}

function ModelHistory({ versions, selectedKey, busyVersion, onActivate }) {
  return (
    <div className="model-history">
      <div className="model-title-row">
        <strong>版本与回测</strong>
        <span>{versions.length} 个版本</span>
      </div>
      <ul>
        {versions.map((item) => (
          <VersionRow
            key={`${item.modelKey}:${item.versionId}`}
            item={item}
            activeKey={selectedKey}
            busyVersion={busyVersion}
            onActivate={onActivate}
          />
        ))}
        {!versions.length && <li className="empty-version">暂无历史模型</li>}
      </ul>
    </div>
  );
}

function useDashboardLoader(selectedKey, setSelectedKey, setMessage) {
  const [dashboard, setDashboard] = useState(null);
  const loadDashboard = useCallback(async () => {
    const data = await fetchModelDashboard();
    setDashboard(data);
    if (!data.models.some((item) => item.key === selectedKey)) {
      setSelectedKey(data.models[0]?.key || "10m_enhanced");
    }
  }, [selectedKey, setSelectedKey]);
  useEffect(() => {
    let stopped = false;
    const load = async () => {
      try {
        const data = await fetchModelDashboard();
        if (!stopped) setDashboard(data);
      } catch (err) {
        if (!stopped) setMessage(err?.message || "模型状态读取失败");
      }
    };
    void load();
    const timer = window.setInterval(load, REFRESH_MS);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [setMessage]);
  return { dashboard, loadDashboard };
}

function useModelActions(loadDashboard, setMessage) {
  const [busyVersion, setBusyVersion] = useState("");
  const [trainingBusy, setTrainingBusy] = useState(false);
  const trainNow = async () => {
    setTrainingBusy(true);
    setMessage("");
    try {
      await triggerModelTraining();
      setMessage("训练任务已开始");
      await loadDashboard();
    } finally {
      setTrainingBusy(false);
    }
  };
  const activate = async (modelKey, versionId) => {
    setBusyVersion(versionId);
    setMessage("");
    try {
      await activateModelVersion(modelKey, versionId);
      setMessage("模型已切换，预测缓存已重载");
      await loadDashboard();
    } finally {
      setBusyVersion("");
    }
  };
  return { busyVersion, trainingBusy, trainNow, activate };
}

export default function ModelManager() {
  const [selectedKey, setSelectedKey] = useState("10m_enhanced");
  const [message, setMessage] = useState("");
  const { dashboard, loadDashboard } = useDashboardLoader(selectedKey, setSelectedKey, setMessage);
  const actions = useModelActions(loadDashboard, setMessage);
  const models = dashboard?.models || [];
  const selectedModel = models.find((item) => item.key === selectedKey) || models[0];
  const versions = useMemo(() => {
    const rows = dashboard?.versions || [];
    return rows.filter((item) => item.modelKey === selectedKey);
  }, [dashboard, selectedKey]);

  return (
    <section className="model-manager">
      <ModelHeader
        schedule={dashboard?.schedule || {}}
        lastRun={dashboard?.lastRun}
        trainingBusy={actions.trainingBusy}
        onTrain={() => actions.trainNow().catch((err) => setMessage(errorText(err, "训练启动失败")))}
      />
      <ModelTabs models={models} selectedKey={selectedKey} onSelect={setSelectedKey} />
      <div className="model-body">
        <ModelCurrentCard model={selectedModel} />
        <ModelHistory
          versions={versions}
          selectedKey={selectedKey}
          busyVersion={actions.busyVersion}
          onActivate={(key, id) => actions.activate(key, id).catch((err) => setMessage(errorText(err, "模型切换失败")))}
        />
      </div>
      {!!message && <div className="model-message">{message}</div>}
    </section>
  );
}
