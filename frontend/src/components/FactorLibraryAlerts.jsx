import { useState } from "react";
import "./FactorLibraryAlerts.css";

const VISIBLE_ALERT_LIMIT = 2;

export default function FactorLibraryAlerts({ alerts }) {
  const [showAll, setShowAll] = useState(false);
  if (!alerts?.length) return null;

  const visible = showAll ? alerts : alerts.slice(0, VISIBLE_ALERT_LIMIT);
  const hiddenCount = alerts.length - VISIBLE_ALERT_LIMIT;

  return (
    <footer className="factor-alerts" aria-label="系统通知">
      {visible.map((alert, index) => (
        <FactorAlert key={alertKey(alert, index)} alert={alert} />
      ))}
      {!showAll && hiddenCount > 0 ? (
        <button type="button" className="factor-alerts-more" onClick={() => setShowAll(true)}>
          还有 {hiddenCount} 条告警，点击展开
        </button>
      ) : null}
      {showAll && alerts.length > VISIBLE_ALERT_LIMIT ? (
        <button type="button" className="factor-alerts-more" onClick={() => setShowAll(false)}>
          收起告警
        </button>
      ) : null}
    </footer>
  );
}

function FactorAlert({ alert }) {
  const [expanded, setExpanded] = useState(false);
  const detailText = formatAlertDetail(alert.detail);
  return (
    <div className={`factor-alert is-${alert.level}`} role="alert">
      <div className="factor-alert-body">
        <strong>{alert.title}</strong>
        <p>{alert.message}</p>
        {expanded && detailText ? <pre className="factor-alert-detail">{detailText}</pre> : null}
      </div>
      <button
        type="button"
        className="factor-alert-btn"
        onClick={() => setExpanded((value) => !value)}
      >
        {expanded ? "收起" : "查看详情"}
      </button>
    </div>
  );
}

function alertKey(alert, index) {
  if (alert.id) return alert.id;
  const detail = alert.detail;
  if (detail && typeof detail === "object") {
    const parts = [alert.code, detail.table, detail.group, detail.missingReason].filter(Boolean);
    if (parts.length) return parts.join(":");
  }
  return `${alert.code ?? "alert"}-${alert.title ?? index}-${index}`;
}

function formatAlertDetail(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  try {
    return JSON.stringify(detail, null, 2);
  } catch {
    return String(detail);
  }
}
