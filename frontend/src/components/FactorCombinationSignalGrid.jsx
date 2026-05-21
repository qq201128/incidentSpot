const DURATION_COLUMN_ORDER = ["10m", "30m", "60m", "1d"];

export default function FactorCombinationSignalGrid({ onSelect, selectedKey, signals }) {
  const columns = groupSignalsIntoDurationColumns(signals);
  return (
    <div className="factor-combo-signals">
      {columns.map(({ duration, signals: columnSignals }) => (
        <div key={duration} className="factor-combo-signal-column" data-duration={duration}>
          {columnSignals.map((signal) => (
            <SignalCard
              key={signalKey(signal)}
              onSelect={onSelect}
              selected={selectedKey === signalKey(signal)}
              signal={signal}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

export function signalKey(signal) {
  return `${signal.duration}-${signal.comboRank || 0}-${signal.factorName || ""}`;
}

function SignalCard({ onSelect, selected, signal }) {
  return (
    <button
      type="button"
      className={`factor-combo-signal ${directionClass(signal.direction)}${selected ? " is-selected" : ""}`}
      onClick={() => onSelect(signal)}
    >
      <div className="factor-combo-signal-top">
        <span>{signal.duration} · Top{signal.comboRank || "—"}</span>
        <strong>{directionText(signal.direction)}</strong>
      </div>
      <span className={`factor-combo-signal-family ${familyClass(signal)}`}>
        {familyText(signal)}
      </span>
      <h3 title={signal.factorDisplayName}>{signal.factorDisplayName || signal.factorName}</h3>
      <p>{memberText(signal.members)}</p>
      <div className="factor-combo-signal-metrics">
        <Metric label="评分" value={formatNum(signal.factorScore, 1)} />
        <Metric label="胜率" value={formatPct(signal.historicalWinRate, 1)} />
        <Metric label="夏普" value={formatNum(signal.historicalSharpe, 2)} />
        <Metric label="模拟" value={simulationLabel(signal)} />
      </div>
    </button>
  );
}

function groupSignalsIntoDurationColumns(signals) {
  const byDuration = new Map();
  for (const signal of signals) {
    const duration = signal.duration || "—";
    const existing = byDuration.get(duration);
    if (existing) {
      existing.push(signal);
      continue;
    }
    byDuration.set(duration, [signal]);
  }
  for (const list of byDuration.values()) {
    list.sort((a, b) => (a.comboRank ?? 0) - (b.comboRank ?? 0));
  }
  return orderedDurations(byDuration).map((duration) => ({
    duration,
    signals: byDuration.get(duration),
  }));
}

function orderedDurations(byDuration) {
  const ordered = DURATION_COLUMN_ORDER.filter((duration) => byDuration.has(duration));
  const rest = [...byDuration.keys()]
    .filter((duration) => !DURATION_COLUMN_ORDER.includes(duration))
    .sort((a, b) => String(a).localeCompare(String(b)));
  return [...ordered, ...rest];
}

function Metric({ label, value }) {
  return (
    <span>
      <small>{label}</small>
      <b>{value}</b>
    </span>
  );
}

function simulationLabel(signal) {
  if (signal.comboStrategyFamily === "high_winrate_goal") {
    return signal.comboRank === 1 ? "高胜率主策略" : `高胜率影子${signal.comboRank}`;
  }
  if (signal.simulationStrategyKey) return signal.comboRank === 1 ? "实盘主策略" : `实盘影子${signal.comboRank}`;
  return "未标记";
}

function familyText(signal) {
  if (signal.comboStrategyFamily === "high_winrate_goal") return "高胜率目标";
  return "普通组合";
}

function familyClass(signal) {
  if (signal.comboStrategyFamily === "high_winrate_goal") return "is-goal";
  return "is-regular";
}

function memberText(members) {
  if (!Array.isArray(members) || !members.length) return "—";
  return members.map((member) => member.displayName || member.name).join(" + ");
}

function directionText(direction) {
  return direction === "down" ? "做空" : "做多";
}

function directionClass(direction) {
  return direction === "down" ? "is-down" : "is-up";
}

function formatNum(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function formatPct(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}
