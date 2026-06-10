
import { formatFinalScore, regimePartLabel } from "../utils/eventFinalDecisionLabels";

const FIXED_PAYOUT_PERCENT = 80;

export default function TradeControls({
  symbol,
  currentPrice,
  durationMinutes,
  amount,
  prediction,
  predictLoading,
  predictInfo,
  predictError,
  onAmountChange,
  onDurationChange,
  onPredictClick,
  onTrade,
}) {
  const priceText = Number(currentPrice || 0).toFixed(2);

  return (
    <div className="card trade-card">
      <header className="trade-panel-top">
        <button
          type="button"
          className="trade-mode-btn sim"
          aria-pressed={false}
          disabled
        >
          <span className="mode-dot" />
          <strong>仅模拟交易</strong>
        </button>
        <p className="toggle-hint trade-mode-hint trade-live-mode-hint">
          当前阶段仅创建本地事件与订单记录，不请求交易所。
        </p>
      </header>

      <div className="trade-symbol-strip">
        <strong className="trade-symbol">{symbol}</strong>
        <span className="trade-price">当前价 {priceText}</span>
      </div>

      <div className="duration-row" role="group" aria-label="结算周期">
        {[10, 30, 60, 1440].map((value) => (
          <button
            key={value}
            type="button"
            className={durationMinutes === value ? "chip active" : "chip"}
            onClick={() => onDurationChange(value)}
          >
            {durationLabel(value)}
          </button>
        ))}
      </div>

      <label className="trade-amount-row">
        <span>数量（USDT）</span>
        <input
          type="number"
          min="1"
          value={amount}
          onChange={(e) => onAmountChange(Number(e.target.value))}
        />
      </label>

      <div className="rate-row">
        <RateCell label="上涨支付率" value={`${FIXED_PAYOUT_PERCENT}%`} />
        <RateCell label="下跌支付率" value={`${FIXED_PAYOUT_PERCENT}%`} />
        <RateCell label="结算周期" value={durationLabel(durationMinutes)} />
      </div>

      <div className="action-row">
        <button type="button" className="up-btn" onClick={() => onTrade("UP")}>
          上涨
        </button>
        <button type="button" className="down-btn" onClick={() => onTrade("DOWN")}>
          下跌
        </button>
      </div>

      <div className="predict-row">
        <button type="button" className="predict-btn" onClick={onPredictClick} disabled={predictLoading}>
          {predictLoading ? "计算下单中…" : `规则下单 · ${durationLabel(durationMinutes)}`}
        </button>
        <PredictionResult prediction={prediction} />
        {!!predictInfo && <div className="predict-info">{predictInfo}</div>}
        {!!predictError && <div className="predict-error">规则或下单失败：{predictError}</div>}
      </div>
    </div>
  );
}

function RateCell({ label, value }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PredictionResult({ prediction }) {
  if (!prediction) return null;
  return (
    <div className="predict-result">
      <PredictionDirection prediction={prediction} />
      <span>更新：{formatPredictionTime(prediction.generatedAt)}</span>
      {prediction.tradeQualityScore != null && (
        <span>
          规则评分：{(prediction.tradeQualityScore * 100).toFixed(1)}%
          {prediction.tradeQualityPassed ? "（通过）" : "（未通过）"}
        </span>
      )}
      {prediction.highWinrateGateEnabled && (
        <span>高胜率门控：{prediction.highWinrateGatePassed ? "通过" : "未通过"}</span>
      )}
      <MarketRegimeLine regime={prediction.marketRegime} />
    </div>
  );
}

function MarketRegimeLine({ regime }) {
  if (!regime) return null;
  if (regime.ready === false) {
    return <span>当前环境：{regime.reason || "数据不足"}</span>;
  }
  const trend = regimePartLabel(regime.trendState);
  const vol = regimePartLabel(regime.volatilityState);
  return <span>当前环境：{trend} · {vol} · 置信度 {formatFinalScore(regime.confidence)}</span>;
}

function PredictionDirection({ prediction }) {
  const up = (prediction.probabilityUp * 100).toFixed(2);
  const down = ((1 - prediction.probabilityUp) * 100).toFixed(2);
  if (isNoTradeSignal(prediction)) {
    return <span>观察涨 {up}% / 跌 {down}%</span>;
  }
  return <span>{prediction.direction === "up" ? "涨" : "跌"} {up}% / {down}%</span>;
}

function isNoTradeSignal(prediction) {
  const confidence = Number(prediction?.confidence);
  return prediction?.certaintyLabel === "RULE_WAIT" || (Number.isFinite(confidence) && confidence <= 0.5);
}

function formatPredictionTime(value) {
  if (!value) return "--";
  const dt = new Date(value);
  if (!Number.isFinite(dt.getTime())) return "--";
  return dt.toLocaleTimeString("zh-CN", { hour12: false });
}

function durationLabel(minutes) {
  return minutes === 1440 ? "1天" : `${minutes}分钟`;
}
