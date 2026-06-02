import { NavLink } from "react-router-dom";

const NAV_ITEMS = Object.freeze([
  { key: "trade", label: "工作台", path: "/" },
  { key: "hit-rate", label: "规则命中率", path: "/rule-hit-rate" },
  { key: "governance", label: "样本观测", path: "/event-governance" },
  { key: "research", label: "研究驾驶舱", path: "/research-dashboard" },
  { key: "live-trading", label: "实盘配置", path: "/live-trading" },
  { key: "factors", label: "因子库", path: "/factors" },
  { key: "learning", label: "自动挖掘", path: "/learning" },
]);

export default function AppNavigation({ appView }) {
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
          <NavLink
            key={item.key}
            className={`app-nav-link${appView === item.key ? " app-nav-link-active" : ""}`}
            to={item.path}
            end={item.path === "/"}
            aria-current={appView === item.key ? "page" : undefined}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
