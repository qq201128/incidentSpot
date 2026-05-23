const MS_PER_MINUTE = 60_000;

const INTERVAL_MS = Object.freeze({
  "10m": 10 * MS_PER_MINUTE,
  "30m": 30 * MS_PER_MINUTE,
  "60m": 60 * MS_PER_MINUTE,
  "1d": 24 * 60 * MS_PER_MINUTE,
});

export function chartIntervalMs(interval) {
  return INTERVAL_MS[interval] ?? INTERVAL_MS["30m"];
}

export function currentIntervalBucketMs(interval, nowMs = Date.now()) {
  const barMs = chartIntervalMs(interval);
  return Math.floor(nowMs / barMs) * barMs;
}

/**
 * When REST/WS lags, synthesize the in-progress bucket from the live index price
 * so the chart does not show a blank gap to the right edge.
 */
export function ensureFormingKline(latest, interval, currentPrice, nowMs = Date.now()) {
  if (!(currentPrice > 0) || !Number.isFinite(currentPrice)) {
    return latest ?? null;
  }

  const bucket = currentIntervalBucketMs(interval, nowMs);
  const price = Number(currentPrice);

  if (!latest) {
    return {
      openTime: bucket,
      open: price,
      high: price,
      low: price,
      close: price,
      volume: 0,
      closeTime: nowMs,
      isClosed: false,
    };
  }

  const openTime = Number(latest.openTime);
  if (!Number.isFinite(openTime)) {
    return latest;
  }

  if (openTime === bucket) {
    return latest.isClosed === true ? { ...latest, isClosed: false } : latest;
  }

  if (openTime > bucket) {
    return latest;
  }

  return {
    openTime: bucket,
    open: price,
    high: price,
    low: price,
    close: price,
    volume: 0,
    closeTime: nowMs,
    isClosed: false,
  };
}

/** Merge live/forming candle into OHLC series used by the chart and MAs. */
export function mergeChartSeries(historyRows, chartCandle) {
  const rows = Array.isArray(historyRows) ? historyRows : [];
  if (!chartCandle) {
    return rows;
  }
  const last = rows[rows.length - 1];
  if (!last) {
    return [chartCandle];
  }
  if (last.time === chartCandle.time) {
    return [...rows.slice(0, -1), chartCandle];
  }
  if (chartCandle.time > last.time) {
    return [...rows, chartCandle];
  }
  return rows;
}
