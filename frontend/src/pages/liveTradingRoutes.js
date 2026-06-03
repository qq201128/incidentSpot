const DEFAULT_DURATION = "10m";
const DEFAULT_SYMBOL = "BTCUSDT";
const DURATION_ORDER = Object.freeze(["10m", "30m", "60m", "1d"]);

const DURATION_RANK = new Map(DURATION_ORDER.map((duration, index) => [duration, index]));

export function requestFromParams(searchParams) {
  const durationParam = trimmedParam(searchParams, "duration");
  const strategyKey = trimmedParam(searchParams, "strategyKey");
  const symbolParam = trimmedParam(searchParams, "symbol");
  return {
    duration: durationParam || DEFAULT_DURATION,
    hasDurationParam: Boolean(durationParam),
    hasSlotRequest: Boolean(strategyKey),
    hasSymbolParam: Boolean(symbolParam),
    strategyKey,
    symbol: normalizeSymbol(symbolParam || DEFAULT_SYMBOL),
  };
}

export function filterSlotRows(rows, request) {
  return rows.filter((row) => slotMatchesRequest(row, request)).sort(compareSlots);
}

export function liveTradingPath(slot) {
  const identity = slotIdentity(slot);
  const params = new URLSearchParams({
    duration: identity.duration,
    strategyKey: identity.strategyKey,
    symbol: identity.symbol,
  });
  return `/live-trading?${params.toString()}`;
}

export function slotIdentity(slot) {
  return {
    duration: requiredSlotValue(slot, "duration"),
    strategyKey: requiredSlotValue(slot, "strategyKey"),
    symbol: normalizeSymbol(requiredSlotValue(slot, "symbol")),
  };
}

function slotMatchesRequest(row, request) {
  const symbol = normalizeSymbol(row?.symbol || "");
  if (request.hasSymbolParam && symbol !== request.symbol) return false;
  if (request.hasDurationParam && String(row?.duration || "") !== request.duration) return false;
  return true;
}

function compareSlots(left, right) {
  return (
    compareText(left.symbol, right.symbol) ||
    durationRank(left.duration) - durationRank(right.duration) ||
    compareText(left.strategyKey, right.strategyKey)
  );
}

function durationRank(duration) {
  return DURATION_RANK.get(String(duration)) ?? DURATION_ORDER.length;
}

function compareText(left, right) {
  return String(left || "").localeCompare(String(right || ""));
}

function normalizeSymbol(symbol) {
  return String(symbol || "").trim().toUpperCase();
}

function requiredSlotValue(slot, field) {
  const value = String(slot?.[field] || "").trim();
  if (!value) throw new Error(`auto-trade slot missing ${field}`);
  return value;
}

function trimmedParam(searchParams, key) {
  return String(searchParams.get(key) || "").trim();
}
