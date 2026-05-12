import { formatPnlU } from "../utils/eventSettlement";
import { strategyLabel } from "../utils/strategyLabels";

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
  aiHistorySuccess,
  liveTradingEnabled,
  onLiveTradingChange,
  onAmountChange,
  onDurationChange,
  onPredictClick,
  onTrade,
}) {
  return (
    <div className="card trade-card">
      <div className="trade-live-mode-block">
        <button
          type="button"
          className={liveTradingEnabled ? "trade-mode-btn live" : "trade-mode-btn sim"}
          aria-pressed={liveTradingEnabled}
          onClick={() => onLiveTradingChange((value) => !value)}
        >
          <span className="mode-dot" />
          <strong>{liveTradingEnabled ? "真实交易：开启" : "模拟交易：关闭"}</strong>
        </button>
        <p className="toggle-hint trade-mode-hint trade-live-mode-hint">
          {liveTradingEnabled
            ? "本页手动下单与「规则计算并下单」将调用 Binance 事件合约接口。"
            : "本页仅创建本地事件与订单记录，不请求交易所下单。"}
        </p>
      </div>
      <div className="symbol-row">
        <strong>{symbol}</strong>
        <span>当前价 {Number(currentPrice || 0).toFixed(2)}</span>
      </div>

      <div className="duration-row">
        {[10, 30, 60, 1440].map((value) => (
          <button
            key={value}
            className={durationMinutes === value ? "chip active" : "chip"}
            onClick={() => onDurationChange(value)}
          >
            {durationLabel(value)}
          </button>
        ))}
      </div>

      <label>数量（USDT）</label>
      <input type="number" min="1" value={amount} onChange={(e) => onAmountChange(Number(e.target.value))} />

      <div className="rate-row">
        <RateCell label="上涨支付率" value={`${FIXED_PAYOUT_PERCENT}%`} />
        <RateCell label="下跌支付率" value={`${FIXED_PAYOUT_PERCENT}%`} />
        <RateCell label="结算周期" value={durationLabel(durationMinutes)} />
      </div>

      <div className="action-row">
        <button className="up-btn" onClick={() => onTrade("UP")}>
          上涨
        </button>
        <button className="down-btn" onClick={() => onTrade("DOWN")}>
          下跌
        </button>
      </div>

      <div className="predict-row">
        <button className="predict-btn" onClick={onPredictClick} disabled={predictLoading}>
          {predictLoading ? "规则计算并下单中..." : `规则计算并下单${durationLabel(durationMinutes)}`}
        </button>
        <PredictionResult prediction={prediction} />
        {!!predictInfo && <div className="predict-info">{predictInfo}</div>}
        {!!predictError && <div className="predict-error">规则或下单失败：{predictError}</div>}
        <AiSuccessSummary symbol={symbol} aiHistorySuccess={aiHistorySuccess} />
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
    </div>
  );
}

function AiSuccessSummary({ symbol, aiHistorySuccess }) {
  const { overall, byStrategy } = aiHistorySuccess;
  return (
    <div className="ai-success-summary">
      <div className="ai-success-row ai-success-overall">
        <span>规则命中率（{symbol}，已结算·合计）</span>
        <strong>
          {overall.total === 0
            ? "暂无样本"
            : `${Math.round(overall.rate * 100)}%（${overall.hits}/${overall.total}） · 盈亏 ${formatPnlU(overall.pnlU)}`}
        </strong>
      </div>
      {byStrategy.length > 0 && (
        <ul className="ai-success-by-strategy">
          {byStrategy.map((row) => (
            <li key={row.strategyKey} className="ai-success-row">
              <span>{strategyLabel(row.strategyKey)}</span>
              <strong>
                {row.total === 0
                  ? "—"
                  : `${Math.round(row.rate * 100)}%（${row.hits}/${row.total}） · 盈亏 ${formatPnlU(row.pnlU)}`}
              </strong>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function PredictionDirection({ prediction }) {
  const up = (prediction.probabilityUp * 100).toFixed(2);
  const down = ((1 - prediction.probabilityUp) * 100).toFixed(2);
  if (isNoTradeSignal(prediction)) {
    return <span>当前不下单（观察涨 {up}% / 跌 {down}%）</span>;
  }
  return <span>结果：{prediction.direction === "up" ? "涨" : "跌"}（涨 {up}% / 跌 {down}%）</span>;
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
