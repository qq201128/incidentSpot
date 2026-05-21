const STORAGE_KEY = "incidentSpot.klineChartUi.v1";

export const DEFAULT_INDICATORS = Object.freeze({
  ma7: true,
  ma20: true,
  ma60: false,
  volume: false,
});

export const DEFAULT_SETTINGS = Object.freeze({
  showGrid: true,
  priceScaleMode: "normal",
  crosshairMagnet: true,
  autoScroll: true,
});

const INDICATOR_KEYS = Object.keys(DEFAULT_INDICATORS);
const SETTING_KEYS = Object.keys(DEFAULT_SETTINGS);
const PERSISTED_DRAWING_TOOLS = new Set(["cursor", "trend", "hline"]);
const PRICE_SCALE_MODES = new Set(["normal", "percent", "log"]);

export function loadKlineChartUiState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

export function saveKlineChartUiState(payload) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    /* quota / private mode */
  }
}

export function mergeIndicators(stored) {
  const source = stored?.indicators;
  if (!source || typeof source !== "object") {
    return { ...DEFAULT_INDICATORS };
  }
  const next = { ...DEFAULT_INDICATORS };
  for (const key of INDICATOR_KEYS) {
    if (typeof source[key] === "boolean") {
      next[key] = source[key];
    }
  }
  return next;
}

export function mergeSettings(stored) {
  const source = stored?.settings;
  if (!source || typeof source !== "object") {
    return { ...DEFAULT_SETTINGS };
  }
  const next = { ...DEFAULT_SETTINGS };
  for (const key of SETTING_KEYS) {
    if (key === "priceScaleMode") {
      if (PRICE_SCALE_MODES.has(source[key])) {
        next.priceScaleMode = source[key];
      }
      continue;
    }
    if (typeof source[key] === "boolean") {
      next[key] = source[key];
    }
  }
  return next;
}

export function mergeDrawingTool(stored) {
  const tool = stored?.drawingTool;
  return PERSISTED_DRAWING_TOOLS.has(tool) ? tool : "cursor";
}

export function mergeDrawingsLocked(stored) {
  return Boolean(stored?.drawingsLocked);
}

export function mergeDrawingMode(stored) {
  return Boolean(stored?.drawingMode);
}

export function snapshotKlineChartUiState({
  indicators,
  settings,
  drawingTool,
  drawingsLocked,
  drawingMode,
}) {
  return {
    indicators,
    settings,
    drawingTool,
    drawingsLocked,
    drawingMode,
  };
}
