import { strategyLabel } from "../../utils/strategyLabels";
import { formatPct } from "./miningFormatters";

export default function MiningModelGrid({ models, summary, busy, onSearchModel }) {
  return (
    <section className="mining-model-section">
      <div className="mining-section-head">
        <h2>
          多模型独立系统
          <span className="mining-ready-badge">
            Ready {summary?.readyModelCount ?? 0}/{summary?.totalModelCount ?? 0}
          </span>
        </h2>
      </div>

      <div className="mining-model-grid">
        {models.map((model) => (
          <ModelCard
            key={model.modelFamily}
            model={model}
            busy={busy === `search-${model.modelFamily}`}
            onSearch={() => onSearchModel(model.modelFamily)}
          />
        ))}
      </div>

      <footer className="mining-model-legend">
        <LegendDot tone="ready" label="可模拟下单" />
        <LegendDot tone="ready-soft" label="就绪" />
        <LegendDot tone="searching" label="搜索中" />
        <LegendDot tone="pending" label="待训练" />
        <LegendDot tone="blocked" label="已阻断" />
        <LegendDot tone="idle" label="不可用" />
      </footer>
    </section>
  );
}

function ModelCard({ model, busy, onSearch }) {
  const progress = model.searchProgress || {};
  const pct = Math.round(Number(progress.percent || 0) * 100);
  const active = ["queued", "running"].includes(model.searchStatus);
  return (
    <article className={`mining-model-card is-${model.cardState}`}>
      <div className="mining-model-card-head">
        <strong>{model.label}</strong>
        <span className={`mining-model-state is-${model.cardState}`}>{model.cardStateLabel}</span>
      </div>
      <dl className="mining-model-metrics">
        <div>
          <dt>策略</dt>
          <dd title={model.strategyKey}>{shortLabel(model.strategyKey)}</dd>
        </div>
        <div>
          <dt>预测就绪</dt>
          <dd>{model.predictionReadyLabel}</dd>
        </div>
        <div>
          <dt>验证胜率</dt>
          <dd>{formatPct(model.validationWinRate, 1)}</dd>
        </div>
        <div>
          <dt>测试胜率</dt>
          <dd>{formatPct(model.testWinRate, 1)}</dd>
        </div>
      </dl>
      <div className={`mining-model-progress${active ? " is-active" : ""}`}>
        <div className="mining-model-progress-head">
          <span>搜索进度</span>
          <b>
            {progress.completed ?? 0}/{progress.total ?? 0}
          </b>
        </div>
        <div className="mining-model-progress-bar">
          <i style={{ width: `${pct}%` }} />
        </div>
      </div>
      <p className="mining-model-latest">
        最新候选: {model.latestCandidateLabel || (model.candidateLibraryTotal ? `${model.candidateLibraryTotal} 条` : "—")}
      </p>
      <button type="button" className="mining-model-search-btn" disabled={busy || active} onClick={onSearch}>
        {busy ? "排队中" : active ? "搜索中" : "搜索候选"}
      </button>
    </article>
  );
}

function shortLabel(key) {
  const label = strategyLabel(key);
  if (!label) return "—";
  return label.length > 14 ? `${label.slice(0, 12)}…` : label;
}

function LegendDot({ tone, label }) {
  return (
    <span className="mining-legend-item">
      <i className={`is-${tone}`} />
      {label}
    </span>
  );
}
