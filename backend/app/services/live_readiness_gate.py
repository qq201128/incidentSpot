from __future__ import annotations

from typing import Any

from app.services.high_winrate_strategy_metrics import high_winrate_decision, high_winrate_thresholds

LIVE_READINESS_VERSION = "pre_live_readiness_gate_v1"
LIVE_READINESS_STATUS_ELIGIBLE = "eligible"
LIVE_READINESS_STATUS_BLOCKED = "blocked"
LIVE_READINESS_REASON_PASSED = "passed"
LIVE_READINESS_REASON_STATUS_NOT_STABLE = "paper_live_status_not_stable"
LIVE_READINESS_REASON_POLICY_DISABLED = "real_trading_disabled_by_project_policy"
LIVE_READINESS_REASON_PNL_MISSING = "paper_live_total_pnl_missing"
LIVE_READINESS_REASON_PNL_BELOW_TARGET = "paper_live_total_pnl_below_target"
MIN_TOTAL_EVENT_PNL = 0.0
REQUIRED_PAPER_LIVE_STATUS = "paper_stable"


def live_readiness_gate(
    metrics: dict[str, Any],
    status: str | None = None,
    *,
    status_reason: str | None = None,
    real_trading_enabled: bool = False,
) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")
    decision = high_winrate_decision(metrics)
    reasons = _gate_reasons(metrics, decision, status, status_reason, real_trading_enabled)
    eligible = not reasons
    return {
        "version": LIVE_READINESS_VERSION,
        "eligible": eligible,
        "status": LIVE_READINESS_STATUS_ELIGIBLE if eligible else LIVE_READINESS_STATUS_BLOCKED,
        "reason": LIVE_READINESS_REASON_PASSED if eligible else reasons[0],
        "reasons": reasons,
        "paperLiveStatus": status,
        "realTradingEnabled": False,
        "manualEnableRequired": True,
        "thresholds": _thresholds(),
        "metrics": metrics,
        "decision": decision,
    }


def _gate_reasons(
    metrics: dict[str, Any],
    decision: dict[str, str],
    status: str | None,
    status_reason: str | None,
    real_trading_enabled: bool,
) -> list[str]:
    reasons: list[str] = []
    if status is not None and status != REQUIRED_PAPER_LIVE_STATUS:
        _append_reason(reasons, status_reason or LIVE_READINESS_REASON_STATUS_NOT_STABLE)
    _append_reason(reasons, decision["reason"] if decision["status"] != REQUIRED_PAPER_LIVE_STATUS else None)
    _append_reason(reasons, _pnl_reason(metrics))
    if status is not None and status != REQUIRED_PAPER_LIVE_STATUS:
        _append_reason(reasons, LIVE_READINESS_REASON_STATUS_NOT_STABLE)
    if real_trading_enabled:
        _append_reason(reasons, LIVE_READINESS_REASON_POLICY_DISABLED)
    return reasons


def _thresholds() -> dict[str, Any]:
    thresholds = dict(high_winrate_thresholds())
    thresholds["minTotalEventPnlU"] = MIN_TOTAL_EVENT_PNL
    thresholds["requiresTotalEventPnlU"] = True
    thresholds["requiredStatus"] = REQUIRED_PAPER_LIVE_STATUS
    thresholds["manualEnableRequired"] = True
    return thresholds


def _pnl_reason(metrics: dict[str, Any]) -> str | None:
    value = metrics.get("totalEventPnlU")
    if value is None:
        return LIVE_READINESS_REASON_PNL_MISSING
    return LIVE_READINESS_REASON_PNL_BELOW_TARGET if float(value) <= MIN_TOTAL_EVENT_PNL else None


def _append_reason(reasons: list[str], reason: str | None) -> None:
    if reason and reason not in reasons:
        reasons.append(reason)
