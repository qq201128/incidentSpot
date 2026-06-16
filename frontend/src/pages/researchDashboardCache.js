const TTL_MS = 60_000;
const cache = new Map();

function cacheKey(symbol, duration, pagination = {}) {
  const page = pagination.page ?? 1;
  const pageSize = pagination.pageSize ?? "default";
  return `${String(symbol || "").trim().toUpperCase()}:${duration}:${page}:${pageSize}`;
}

export function peekResearchDashboardCache(symbol, duration, pagination = {}) {
  const key = cacheKey(symbol, duration, pagination);
  const entry = cache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.at > TTL_MS) {
    cache.delete(key);
    return null;
  }
  return entry;
}

export function storeResearchDashboardCache(symbol, duration, report, pagination = {}) {
  if (!report) return;
  cache.set(cacheKey(symbol, duration, pagination), { at: Date.now(), report });
}

export function clearResearchDashboardCache() {
  cache.clear();
}
