const TTL_MS = 60_000;
const cache = new Map();

function cacheKey(symbol, duration) {
  return `${String(symbol || "").trim().toUpperCase()}:${duration}`;
}

export function peekResearchDashboardCache(symbol, duration) {
  const entry = cache.get(cacheKey(symbol, duration));
  if (!entry) return null;
  if (Date.now() - entry.at > TTL_MS) {
    cache.delete(cacheKey(symbol, duration));
    return null;
  }
  return entry;
}

export function storeResearchDashboardCache(symbol, duration, report) {
  if (!report) return;
  cache.set(cacheKey(symbol, duration), { at: Date.now(), report });
}

export function clearResearchDashboardCache() {
  cache.clear();
}
