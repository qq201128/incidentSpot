import { formatMetric, formatTime, metricValue, modelLabel } from "./modelDisplay";
import "./ModelProductionGate.css";

const FLIPPED_OPERATOR = { ">=": "<=", "<=": ">=" };
const WINDOW_LABELS = { "1d": "1天", "3d": "3天", "7d": "7天", "30d": "30天" };

function scoreItems(metrics) {
  const production = metrics?.production_gate_backtest;
  return [
    ["测试区分度", formatMetric(metrics?.test_auc)],
    ["综合分数", formatMetric(metrics?.test_f1)],
    ["校准误差", formatMetric(metrics?.test_brier_calibrated, "price")],
    ["回测胜率", formatMetric(metricValue(metrics, "backtest_test_split.win_rate"), "percent")],
    ["方向命中", formatMetric(metricValue(metrics, "backtest_test_split.direction_hit_rate"), "percent")],
    ["每日交易", formatMetric(metricValue(metrics, "backtest_test_split.trades_per_day"))],
    ["生产回测胜率", formatMetric(production?.win_rate, "percent")],
    ["生产回测每日", formatMetric(production?.trades_per_day)],
  ];
}

function targetStatus(metrics) {
  const target = metrics?.production_target;
  if (!target) return null;
  const passed = Boolean(target.passed);
  const winRate = formatMetric(target.winRate, "percent");
  const targetWinRate = formatMetric(target.targetWinRate, "percent");
  const trades = formatMetric(target.tradesPerDay);
  const targetTrades = formatMetric(target.targetTradesPerDay);
  return {
    passed,
    text: passed
      ? `回测达标：胜率 ${winRate} / 每日 ${trades} 单`
      : `回测未达标：胜率 ${winRate} / 每日 ${trades} 单；目标 >${targetWinRate} 且 ≥${targetTrades} 单/天`,
  };
}

function productionGateItems(gate) {
  if (!gate?.enabled) return [];
  const backtest = gate.backtest || {};
  const rules = Array.isArray(gate.rules) ? gate.rules : [];
  const minConfidence = Math.min(...rules.map((item) => Number(item.min_confidence)).filter(Number.isFinite));
  return [
    ["门控", gate.gate_name || "--"],
    ["规则", rules.length ? `${rules.length} 条 / 多空双向` : "--"],
    ["最低置信度", Number.isFinite(minConfidence) ? `≥${formatMetric(minConfidence, "percent")}` : "--"],
    ["回测合并胜率", formatMetric(backtest.win_rate, "percent")],
    ["回测合并每日", formatMetric(backtest.trades_per_day)],
    ["回测交易数", formatMetric(backtest.test_trades)],
  ];
}

function directionText(direction) {
  if (direction === "up") return "看涨";
  if (direction === "down") return "看跌";
  return "--";
}

function conditionText(condition, direction) {
  const feature = condition.feature || "--";
  const op = condition.operator || ">=";
  const value = Number(condition.value);
  if (condition.transform === "signed" && direction === "down" && Number.isFinite(value)) {
    return `${feature} ${FLIPPED_OPERATOR[op] || op} ${formatMetric(-value)}`;
  }
  return `${feature} ${op} ${formatMetric(condition.value)}`;
}

function ProductionGate({ gate }) {
  const rows = productionGateItems(gate);
  const reports = Array.isArray(gate?.rule_reports) ? gate.rule_reports : [];
  if (!rows.length) return null;
  return (
    <div className="production-gate">
      <div className="production-gate-title">生产门控 · 历史回测</div>
      <div className="production-gate-grid">
        {rows.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      <div className="production-rule-list">
        {reports.map((item) => {
          const rule = gate.rules?.find((row) => row.name === item.name) || {};
          return (
            <div className="production-rule" key={item.name}>
              <div>
                <strong>{directionText(item.direction)}</strong>
                <span>{formatMetric(item.backtest?.win_rate, "percent")} / 每日 {formatMetric(item.backtest?.trades_per_day)}</span>
              </div>
              <p>
                置信度 ≥{formatMetric(rule.min_confidence, "percent")}；
                {(rule.conditions || []).map((condition) => conditionText(condition, item.direction)).join("；")}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function monitorText(monitor) {
  if (monitor?.status === "ok") return "正常";
  if (monitor?.status === "warn") return "预警";
  if (monitor?.status === "collecting") return `收集中 ${monitor.sampleTrades || 0}/${monitor.minimumTrades || 0}`;
  return "--";
}

function forwardItems(stats) {
  const gated = stats?.productionGate || {};
  const monitor = stats?.overfitMonitor || {};
  return [
    ["已结算单", formatMetric(gated.trades)],
    ["实盘胜率", formatMetric(gated.winRate, "percent")],
    ["实盘每日", formatMetric(gated.tradesPerDay)],
    ["平均收益", formatMetric(gated.avgReturn, "signed")],
    ["过拟合监控", monitorText(monitor)],
  ];
}

function ForwardValidation({ stats }) {
  if (!stats) return null;
  const windows = Object.entries(stats.windows || {});
  const perRule = stats.perRule || [];
  return (
    <div className="production-gate">
      <div className="production-gate-title">实盘/模拟盘前向验证</div>
      <MetricGrid rows={forwardItems(stats)} />
      <WindowStats rows={windows} />
      <RuleForwardStats rows={perRule} />
    </div>
  );
}

function MetricGrid({ rows }) {
  return (
    <div className="production-gate-grid">
      {rows.map(([label, value]) => (
        <div key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

function WindowStats({ rows }) {
  if (!rows.length) return null;
  return (
    <div className="production-mini-table">
      {rows.map(([key, item]) => (
        <span key={key}>{WINDOW_LABELS[key] || key}: {formatMetric(item.winRate, "percent")} / {formatMetric(item.trades)}单</span>
      ))}
    </div>
  );
}

function RuleForwardStats({ rows }) {
  if (!rows.length) return null;
  return (
    <div className="production-rule-list">
      {rows.map((item) => (
        <div className="production-rule" key={item.rule}>
          <div>
            <strong>{item.rule}</strong>
            <span>{formatMetric(item.winRate, "percent")} / {formatMetric(item.trades)}单</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function FeatureIntegrity({ report }) {
  if (!report) return null;
  const passed = report.status === "passed";
  return (
    <div className="production-gate">
      <div className="production-gate-title">多周期未来函数检查</div>
      <MetricGrid rows={[
        ["状态", passed ? "通过" : "发现风险"],
        ["检查特征", formatMetric(report.checkedFeatureCount)],
        ["样本行", formatMetric(report.sampleRows)],
        ["风险字段", formatMetric((report.leakingFeatures || []).length)],
      ]} />
    </div>
  );
}

export default function ModelCurrentCard({ model }) {
  const target = targetStatus(model?.metrics);
  return (
    <div className="model-current">
      <div className="model-title-row">
        <strong>{modelLabel(model)}</strong>
        <span>{model?.exists ? "已加载" : "缺少文件"}</span>
      </div>
      <div className="model-score-grid">
        {scoreItems(model?.metrics).map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      <p className="model-meta-line">
        周期 {model?.metrics?.train_window_days || "--"} 天 / 样本{" "}
        {model?.metrics?.row_count || "--"} / 更新 {formatTime(model?.updatedAt)}
      </p>
      {target && <div className={`model-target ${target.passed ? "passed" : "failed"}`}>{target.text}</div>}
      <ProductionGate gate={model?.meta?.production_gate} />
      <ForwardValidation stats={model?.metrics?.forward_validation} />
      <FeatureIntegrity report={model?.metrics?.feature_integrity} />
    </div>
  );
}
