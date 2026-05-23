from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.event_pnl_rows import batch_combo_strategy_keys, settled_event_metric_rows
from app.services.factor_combo_simulation_keys import is_batch_combo_simulation_strategy
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


def evaluate_batch_combo_event_demotion(symbol: str, duration: str) -> dict[str, Any]:
    """Evaluate batch combo health from event PnL; observe-only (never disables strategies)."""
    from app.db.session import get_conn

    sym = symbol.strip().upper()
    conn = get_conn()
    try:
        keys = batch_combo_strategy_keys(conn, sym, duration)
        evaluations = [_evaluate_strategy(conn, sym, duration, key) for key in keys]
        watchlist = [item for item in evaluations if item["status"] == STATUS_DEMOTED]
        return {
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


def _evaluate_strategy(conn: Any, symbol: str, duration: str, strategy_key: str) -> dict[str, Any]:
    if not is_batch_combo_simulation_strategy(strategy_key):
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
        "status": status,
        "reason": reason,
        "metricsSource": "events",
        "metrics": metrics,
        "sampleCount": metrics.get("sampleCount", 0),
        "winRate": metrics.get("winRate"),
        "profitFactor": metrics.get("profitFactor"),
        "consecutiveLosses": metrics.get("consecutiveLosses"),
        "totalPnlU": _total_pnl(rows),
        "autoTradeAction": "none",
    }


def _total_pnl(rows: list[dict[str, Any]]) -> float:
    return round(sum(float(row["event_pnl"]) for row in rows if row.get("event_pnl") is not None), 6)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
