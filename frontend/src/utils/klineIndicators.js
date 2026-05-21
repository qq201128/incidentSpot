export function computeMovingAverage(candles, period) {
  const periodSize = Math.max(1, Math.floor(Number(period) || 1));
  if (!Array.isArray(candles) || !candles.length) return [];
  const rows = [];
  let sum = 0;
  for (let index = 0; index < candles.length; index += 1) {
    const close = Number(candles[index]?.close);
    if (!Number.isFinite(close)) continue;
    sum += close;
    if (index >= periodSize - 1) {
      const remove = Number(candles[index - periodSize + 1]?.close);
      if (index >= periodSize) sum -= remove;
      rows.push({ time: candles[index].time, value: sum / periodSize });
    }
  }
  return rows;
}
