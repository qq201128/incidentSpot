import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_INDICATORS,
  DEFAULT_SETTINGS,
  loadKlineChartUiState,
  mergeDrawingMode,
  mergeDrawingTool,
  mergeDrawingsLocked,
  mergeIndicators,
  mergeSettings,
  saveKlineChartUiState,
  snapshotKlineChartUiState,
} from "../utils/klineChartUiStorage";

export { DEFAULT_INDICATORS, DEFAULT_SETTINGS };

export const DRAWING_TOOLS = Object.freeze([
  { id: "cursor", icon: "＋", label: "十字光标" },
  { id: "trend", icon: "╱", label: "趋势线" },
  { id: "hline", icon: "⌁", label: "水平线" },
  { id: "measure", icon: "☰", label: "适应窗口" },
  { id: "reset", icon: "⊙", label: "重置视图" },
  { id: "settings", icon: "◇", label: "打开设置" },
  { id: "lock", icon: "🔒", label: "锁定画线" },
  { id: "clear", icon: "⌫", label: "清除画线" },
]);

export function useKlineChartUi() {
  const storedRef = useRef(loadKlineChartUiState());
  const stored = storedRef.current;

  const [indicators, setIndicators] = useState(() => mergeIndicators(stored));
  const [settings, setSettings] = useState(() => mergeSettings(stored));
  const [drawingTool, setDrawingTool] = useState(() => mergeDrawingTool(stored));
  const [drawingsLocked, setDrawingsLocked] = useState(() => mergeDrawingsLocked(stored));
  const [drawingMode, setDrawingMode] = useState(() => mergeDrawingMode(stored));
  const [activePanel, setActivePanel] = useState(null);
  const [fitToken, setFitToken] = useState(0);
  const [clearDrawingsToken, setClearDrawingsToken] = useState(0);

  useEffect(() => {
    saveKlineChartUiState(
      snapshotKlineChartUiState({
        indicators,
        settings,
        drawingTool,
        drawingsLocked,
        drawingMode,
      }),
    );
  }, [indicators, settings, drawingTool, drawingsLocked, drawingMode]);

  const togglePanel = useCallback((panel) => {
    setActivePanel((current) => (current === panel ? null : panel));
  }, []);

  const toggleIndicator = useCallback((key) => {
    setIndicators((current) => ({ ...current, [key]: !current[key] }));
  }, []);

  const patchSettings = useCallback((patch) => {
    setSettings((current) => ({ ...current, ...patch }));
  }, []);

  const requestFit = useCallback(() => {
    setFitToken((value) => value + 1);
  }, []);

  const toggleDrawingMode = useCallback(() => {
    setDrawingMode((value) => !value);
    setActivePanel(null);
  }, []);

  const selectDrawingTool = useCallback((toolId) => {
    if (toolId === "lock") {
      setDrawingsLocked((value) => !value);
      return;
    }
    if (toolId === "settings") {
      setActivePanel("settings");
      return;
    }
    if (toolId === "measure" || toolId === "reset") {
      setFitToken((value) => value + 1);
      setDrawingTool("cursor");
      return;
    }
    if (toolId === "clear") {
      setDrawingTool("cursor");
      setClearDrawingsToken((value) => value + 1);
      return;
    }
    setDrawingTool(toolId);
    setDrawingMode(true);
  }, []);

  const togglePriceScaleMode = useCallback((mode) => {
    setSettings((current) => ({
      ...current,
      priceScaleMode: current.priceScaleMode === mode ? "normal" : mode,
    }));
  }, []);

  return {
    activePanel,
    clearDrawingsToken,
    drawingMode,
    drawingTool,
    drawingsLocked,
    fitToken,
    indicators,
    settings,
    patchSettings,
    requestFit,
    selectDrawingTool,
    setActivePanel,
    setDrawingMode,
    setDrawingTool,
    setDrawingsLocked,
    toggleDrawingMode,
    toggleIndicator,
    togglePanel,
    togglePriceScaleMode,
  };
}
