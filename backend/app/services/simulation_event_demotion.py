from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from app.services.high_winrate_strategy_metrics import (
    ACTIVE_SAMPLE_COUNT,
    high_winrate_decision,
    high_winrate_metrics,
    high_winrate_thresholds,
)

STATUS_ACTIVE = "active"
STATUS_DEMOTED = "demoted"
STATUS_COLLECTING = "collecting"
STATUS_INSUFFICIENT = "insufficient_samples"
OBSERVE_ONLY = True


def evaluate_simulation_event_demotion(
    symbol: str,
    duration: str,
    *,
    source: str,
    list_strategy_keys: Callable[[Any, str, str], list[str]],
    validate_strategy_key: Callable[[str], bool],
) -> dict[str, Any]:
    """Evaluate settled event PnL for simulation strategies; observe-only."""
    from app.db.session import get_conn
    from app.services.event_pnl_rows import settled_event_metric_rows

    sym = symbol.strip().upper()
    conn = get_conn()
    try:
        keys = list_strategy_keys(conn, sym, duration)
        evaluations = [
            _evaluate_strategy(
                conn,
                sym,
                duration,
                key,
                settled_event_metric_rows=settled_event_metric_rows,
                validate_strategy_key=validate_strategy_key,
            )
            for key in keys
        ]
        watchlist = [item for item in evaluations if item["status"] == STATUS_DEMOTED]
        return {
            "source": source,
            "symbol": sym,
            "duration": duration,
            "evaluatedAt": _utc_now(),
            "observeOnly": OBSERVE_ONLY,
            "thresholds": high_winrate_thresholds(),
            "evaluatedCount": len(evaluations),
            "watchlistCount": len(watchlist),
            "demotedCount": len(watchlist),
            "evaluations": evaluations,
            "watchlist": watchlist,
        }
    finally:
        conn.close()


def _evaluate_strategy(
    conn: Any,
    symbol: str,
    duration: str,
    strategy_key: str,
    *,
    settled_event_metric_rows: Callable[..., list[dict[str, Any]]],
    validate_strategy_key: Callable[[str], bool],
) -> dict[str, Any]:
    if not validate_strategy_key(strategy_key):
        return _evaluation_payload(strategy_key, STATUS_INSUFFICIENT, "unsupported_strategy_key", [], {})
    rows = settled_event_metric_rows(conn, symbol, duration, strategy_key=strategy_key)
    metrics = high_winrate_metrics(rows)
    if metrics["sampleCount"] < ACTIVE_SAMPLE_COUNT:
        return _evaluation_payload(
            strategy_key,
            STATUS_COLLECTING,
            "insufficient_event_samples",
            rows,
            metrics,
        )
    decision = high_winrate_decision(metrics)
    status = STATUS_DEMOTED if decision["status"] == "demoted" else STATUS_ACTIVE
    return _evaluation_payload(strategy_key, status, decision["reason"], rows, metrics)


def _evaluation_payload(
    strategy_key: str,
    status: str,
    reason: str,
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "strategyKey": strategy_key,
        "displayRule": _display_rule(rows),
        "status": status,
        "reason": reason,
        "metricsSource": "events",
        "metrics": metrics,
        "sampleCount": metrics.get("sampleCount", 0),
        "winRate": metrics.get("winRate"),
        "profitFactor": metrics.get("profitFactor"),
        "consecutiveLosses": metrics.get("consecutiveLosses", 0),
        "totalPnlU": _total_pnl(rows),
        "autoTradeAction": "none",
    }


def _display_rule(rows: list[dict[str, Any]]) -> str | None:
    for row in rows:
        rule = row.get("high_winrate_rule")
        if rule:
            return str(rule)
    return None


def _total_pnl(rows: list[dict[str, Any]]) -> float:
    return round(sum(float(row["event_pnl"]) for row in rows if row.get("event_pnl") is not None), 6)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
