const MS_PER_SECOND = 1000;

export function normalizeChartCandle(item) {
  return {
    time: Math.floor(item.openTime / MS_PER_SECOND),
    open: Number(item.open),
    high: Number(item.high),
    low: Number(item.low),
    close: Number(item.close),
  };
}

export function mergeKlineCandle(previous, next) {
  if (!previous || previous.openTime !== next.openTime) {
    return next;
  }
  const prevHigh = Number(previous.high);
  const prevLow = Number(previous.low);
  const nextHigh = Number(next.high);
  const nextLow = Number(next.low);
  return {
    ...next,
    open: previous.open,
    high: Math.max(prevHigh, nextHigh),
    low: Math.min(prevLow, nextLow),
  };
}
