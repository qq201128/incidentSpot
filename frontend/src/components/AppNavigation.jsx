const NAV_ITEMS = Object.freeze([
  { key: "trade", label: "工作台" },
  { key: "hit-rate", label: "规则命中率" },
  { key: "governance", label: "样本观测" },
  { key: "research", label: "研究驾驶舱" },
  { key: "factors", label: "因子库" },
  { key: "learning", label: "自动挖掘" },
]);

export default function AppNavigation({ appView, onViewChange }) {
  return (
    <header className="app-shell-header">
      <a className="brand-lockup" href="/" aria-label="incidentSpot 工作台">
        <span className="brand-mark" aria-hidden>
          <span />
          <span />
          <span />
          <span />
        </span>
        <strong>incidentSpot</strong>
      </a>
      <nav className="app-nav" aria-label="主导航">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.key}
            type="button"
            className={`app-nav-link${appView === item.key ? " app-nav-link-active" : ""}`}
            onClick={() => onViewChange(item.key)}
            aria-current={appView === item.key ? "page" : undefined}
          >
            {item.label}
          </button>
        ))}
      </nav>
    </header>
  );
}
