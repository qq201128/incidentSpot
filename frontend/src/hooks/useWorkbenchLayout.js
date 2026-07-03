/**
 * 工作台布局保存Hook
 */
import { useState, useEffect } from 'react';

const LAYOUT_STORAGE_KEY = 'incidentSpot:workbench_layout';

const DEFAULT_LAYOUT = {
  // 面板显示状态
  panels: {
    orderbook: true,
    chart: true,
    trades: true,
    positions: true,
    orders: true,
  },
  // 面板尺寸
  sizes: {
    leftPanel: 300,
    rightPanel: 300,
    chartHeight: 400,
  },
  // 图表设置
  chart: {
    interval: '30m',
    indicators: ['EMA', 'Volume'],
  },
  // 订单簿设置
  orderbook: {
    depth: 5,
    view: 'both', // both / bids / asks
  },
};

export function useWorkbenchLayout() {
  const [layout, setLayout] = useState(() => {
    try {
      const saved = localStorage.getItem(LAYOUT_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        // 合并默认值（处理新增字段）
        return mergeDeep(DEFAULT_LAYOUT, parsed);
      }
    } catch (err) {
      console.error('Failed to load layout:', err);
    }
    return DEFAULT_LAYOUT;
  });

  useEffect(() => {
    try {
      localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(layout));
    } catch (err) {
      console.error('Failed to save layout:', err);
    }
  }, [layout]);

  const updateLayout = (updates) => {
    setLayout((prev) => mergeDeep(prev, updates));
  };

  const resetLayout = () => {
    setLayout(DEFAULT_LAYOUT);
  };

  const togglePanel = (panelName) => {
    setLayout((prev) => ({
      ...prev,
      panels: {
        ...prev.panels,
        [panelName]: !prev.panels[panelName],
      },
    }));
  };

  return {
    layout,
    setLayout,
    updateLayout,
    resetLayout,
    togglePanel,
  };
}

function mergeDeep(target, source) {
  const output = { ...target };

  if (isObject(target) && isObject(source)) {
    Object.keys(source).forEach((key) => {
      if (isObject(source[key])) {
        if (!(key in target)) {
          output[key] = source[key];
        } else {
          output[key] = mergeDeep(target[key], source[key]);
        }
      } else {
        output[key] = source[key];
      }
    });
  }

  return output;
}

function isObject(item) {
  return item && typeof item === 'object' && !Array.isArray(item);
}
