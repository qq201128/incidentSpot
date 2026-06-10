import { modelFamilyLabel } from "../utils/modelFamilies";
import { modelBlockReasonLabel } from "../utils/eventFinalDecisionLabels";
import "./ModelRegimeValidationPanel.css";

const REGIME_KEYS = [
  "trend_up",
  "trend_down",
  "range",
  "uncertain",
  "high_vol",
  "normal_vol",
  "low_vol",
];

export default function ModelRegimeValidationPanel({ statuses }) {
  const rows = Array.isArray(statuses) ? statuses : [];
  if (!rows.length) {
    return (
      <section className="research-side-section">
        <h3>环境分组验证</h3>
        <p className="research-empty small">暂无模型族状态</p>
      </section>
    );
  }
  return (
    <section className="research-side-section research-regime-panel">
      <h3>环境分组验证</h3>
      <p className="research-side-note">展示 regime_* 特征训练与预测阻断状态。</p>
      {rows.map((row) => (
        <ModelRegimeCard key={row.modelFamily} row={row} />
      ))}
    </section>
  );
}

function ModelRegimeCard({ row }) {
  const family = row.modelFamily || "—";
  const blocked = row.shadowPredictionBlockedReason || row.reason;
  const retrain = blocked === "clean_event_retrain_required" || row.cleanEventFeatures === false;
  return (
    <article className="research-regime-card">
      <header>
        <strong>{modelFamilyLabel(family, family)}</strong>
        <span className={retrain ? "is-blocked" : row.cleanEventFeatures ? "is-ready" : "is-warn"}>
          {retrain ? "需重训" : row.cleanEventFeatures ? "regime 已就绪" : "特征未对齐"}
        </span>
      </header>
      {retrain ? (
        <p className="research-regime-block">{modelBlockReasonLabel("clean_event_retrain_required")}</p>
      ) : null}
      {!retrain && blocked && blocked !== "passed" ? (
        <p className="research-regime-block">{modelBlockReasonLabel(blocked)}</p>
      ) : null}
      <RegimeMetrics regimeValidation={row.regimeValidation} />
    </article>
  );
}

function RegimeMetrics({ regimeValidation }) {
  const metrics = normalizeRegimeMetrics(regimeValidation);
  if (!metrics.length) {
    return <p className="research-empty small">尚无按环境分组的验证结果</p>;
  }
  return (
    <table className="research-regime-table">
      <thead>
        <tr>
          <th>环境</th>
          <th>胜率</th>
          <th>样本</th>
        </tr>
      </thead>
      <tbody>
        {metrics.map((row) => (
          <tr key={row.key}>
            <td>{row.label}</td>
            <td>{formatWinRate(row.winRate)}</td>
            <td>{row.sampleCount ?? "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function normalizeRegimeMetrics(regimeValidation) {
  if (!regimeValidation || typeof regimeValidation !== "object") return [];
  const keys = Object.keys(regimeValidation);
  const ordered = REGIME_KEYS.filter((key) => keys.includes(key));
  const extras = keys.filter((key) => !REGIME_KEYS.includes(key));
  return [...ordered, ...extras].map((key) => {
    const payload = regimeValidation[key] || {};
    return {
      key,
      label: regimeMetricLabel(key),
      winRate: payload.winRate ?? payload.accuracy,
      sampleCount: payload.sampleCount ?? payload.n,
    };
  });
}

function regimeMetricLabel(key) {
  if (key.includes(":")) {
    const [trend, vol] = key.split(":");
    return `${trend} · ${vol}`;
  }
  const labels = {
    trend_up: "上升趋势",
    trend_down: "下降趋势",
    range: "震荡",
    uncertain: "趋势不明",
    high_vol: "高波动",
    normal_vol: "正常波动",
    low_vol: "低波动",
  };
  return labels[key] || key;
}

function formatWinRate(rate) {
  const n = Number(rate);
  if (!Number.isFinite(n)) return "—";
  return `${Math.round(n * 100)}%`;
}
