import { useState } from "react";
import FactorLearningPanel from "../components/FactorLearningPanel";
import "./FactorLearningPage.css";

const DEFAULT_SYMBOL = "BTCUSDT";
const DEFAULT_DURATION = "10m";
const DURATIONS = [
  { value: "10m", label: "10 分钟" },
  { value: "30m", label: "30 分钟" },
  { value: "60m", label: "60 分钟" },
  { value: "1d", label: "1 天" },
];

export default function FactorLearningPage() {
  const [symbol, setSymbol] = useState(DEFAULT_SYMBOL);
  const [duration, setDuration] = useState(DEFAULT_DURATION);

  return (
    <main className="factor-learning-page layout">
      <header className="topbar factor-learning-page-topbar">
        <div>
          <span className="eyebrow">自动挖掘</span>
          <h1>因子学习与候选挖掘</h1>
        </div>
        <div className="factor-learning-page-controls">
          <label>
            交易对
            <input
              value={symbol}
              onChange={(event) => setSymbol(event.target.value.toUpperCase())}
              placeholder="BTCUSDT"
            />
          </label>
          <label>
            规则周期
            <select value={duration} onChange={(event) => setDuration(event.target.value)}>
              {DURATIONS.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>
      <section className="factor-learning-page-body">
        <FactorLearningPanel symbol={symbol} duration={duration} />
      </section>
    </main>
  );
}
