/**
 * 订单快捷键Hook
 */
import { useEffect } from 'react';

export function useOrderHotkeys({
  onBuyMarket,
  onSellMarket,
  onBuyLimit,
  onSellLimit,
  onCancelAll,
  enabled = true,
}) {
  useEffect(() => {
    if (!enabled) return;

    function handleKeyDown(event) {
      // 忽略输入框中的快捷键
      const target = event.target;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        return;
      }

      const ctrl = event.ctrlKey || event.metaKey;

      // Ctrl/Cmd + B: 市价买入
      if (ctrl && event.key === 'b') {
        event.preventDefault();
        onBuyMarket?.();
      }

      // Ctrl/Cmd + S: 市价卖出
      if (ctrl && event.key === 's') {
        event.preventDefault();
        onSellMarket?.();
      }

      // Ctrl/Cmd + Shift + B: 限价买入
      if (ctrl && event.shiftKey && event.key === 'B') {
        event.preventDefault();
        onBuyLimit?.();
      }

      // Ctrl/Cmd + Shift + S: 限价卖出
      if (ctrl && event.shiftKey && event.key === 'S') {
        event.preventDefault();
        onSellLimit?.();
      }

      // Esc: 取消所有订单
      if (event.key === 'Escape') {
        event.preventDefault();
        onCancelAll?.();
      }

      // ? : 显示快捷键帮助
      if (event.key === '?' && !ctrl) {
        event.preventDefault();
        showHotkeyHelp();
      }
    }

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [enabled, onBuyMarket, onSellMarket, onBuyLimit, onSellLimit, onCancelAll]);
}

function showHotkeyHelp() {
  const helpText = `
快捷键列表：

Ctrl/Cmd + B          市价买入
Ctrl/Cmd + S          市价卖出
Ctrl/Cmd + Shift + B  限价买入
Ctrl/Cmd + Shift + S  限价卖出
Esc                   取消所有订单
?                     显示此帮助
  `.trim();

  alert(helpText);
}

/**
 * 快捷键帮助组件
 */
export function HotkeyHelp() {
  return (
    <div className="hotkey-help">
      <h3>快捷键</h3>
      <dl className="hotkey-list">
        <div className="hotkey-item">
          <dt><kbd>Ctrl</kbd> + <kbd>B</kbd></dt>
          <dd>市价买入</dd>
        </div>
        <div className="hotkey-item">
          <dt><kbd>Ctrl</kbd> + <kbd>S</kbd></dt>
          <dd>市价卖出</dd>
        </div>
        <div className="hotkey-item">
          <dt><kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>B</kbd></dt>
          <dd>限价买入</dd>
        </div>
        <div className="hotkey-item">
          <dt><kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>S</kbd></dt>
          <dd>限价卖出</dd>
        </div>
        <div className="hotkey-item">
          <dt><kbd>Esc</kbd></dt>
          <dd>取消所有订单</dd>
        </div>
      </dl>
    </div>
  );
}
