from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.services.high_winrate_strategy_metrics import high_winrate_thresholds
from app.services.rule_config import DURATION_TO_MINUTES
from app.services.strategy_registry import HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY

STATUS_TRADABLE = "tradable"
STATUS_PAUSED = "paused"
STATUS_DEMOTED = "demoted"
DEFAULT_QTY = 5.0
RANKING_REFRESH_FAILED_REASON = "candidate_pool_exhausted_refresh_failed"


def status_payload(
    status: str,
    reason: str,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thresholds = high_winrate_thresholds()
    payload = {
        "status": status,
        "reason": reason,
        "metrics": metrics,
        "thresholds": thresholds,
        "sampleCount": metrics.get("sampleCount"),
        "settledSampleCount": metrics.get("sampleCount"),
        "winRate": metrics.get("winRate"),
        "profitFactor": metrics.get("profitFactor"),
        "consecutiveLosses": metrics.get("consecutiveLosses"),
        "metricsSource": metrics.get("metricsSource"),
        "totalEventPnlU": metrics.get("totalEventPnlU"),
        "paperStability": metrics.get("paperStability"),
        "requiredSampleCount": thresholds["requiredSampleCount"],
        "tradable": status == STATUS_TRADABLE,
        "evaluatedAt": utc_now(),
    }
    if extra:
        payload.update(extra)
    return payload


def refresh_failed_payload(payload: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "status": STATUS_DEMOTED,
        "reason": RANKING_REFRESH_FAILED_REASON,
        "evaluatedAt": utc_now(),
        "refreshReport": refresh_report_summary(report),
    }


def refresh_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    ranking = report.get("ranking") if isinstance(report.get("ranking"), list) else []
    summary = {
        "updatedAt": report.get("updatedAt"),
        "rankingTotal": len(ranking),
        "rankingFailure": report.get("rankingFailure"),
        "validationGate": report.get("validationGate"),
        "candidateDiagnostics": report.get("candidateDiagnostics"),
        "rankingDiagnostics": report.get("rankingDiagnostics"),
        "promotion": report.get("promotion"),
        "target": report.get("target"),
    }
    return {key: value for key, value in summary.items() if value is not None}


def has_rankings(report: dict[str, Any]) -> bool:
    ranking = report.get("ranking")
    return isinstance(ranking, list) and bool(ranking)


def write_status(conn: Any, symbol: str, duration: str, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    conn.execute(
        """
        INSERT INTO high_winrate_strategy_status(
          strategy_key, symbol, duration, status, reason, details_json,
          sample_count, win_rate, profit_factor, consecutive_losses, evaluated_at, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(strategy_key, symbol, duration) DO UPDATE SET
          status = excluded.status, reason = excluded.reason, details_json = excluded.details_json,
          sample_count = excluded.sample_count, win_rate = excluded.win_rate,
          profit_factor = excluded.profit_factor, consecutive_losses = excluded.consecutive_losses,
          evaluated_at = excluded.evaluated_at, updated_at = excluded.updated_at
        """,
        status_values(symbol, duration, payload, metrics),
    )


def status_values(symbol: str, duration: str, payload: dict[str, Any], metrics: dict[str, Any]) -> tuple[Any, ...]:
    return (
        HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
        symbol,
        duration,
        payload["status"],
        payload["reason"],
        json.dumps(payload, ensure_ascii=False),
        int(metrics["sampleCount"]),
        metrics["winRate"],
        metrics["profitFactor"],
        int(metrics["consecutiveLosses"]),
        payload["evaluatedAt"],
        utc_now(),
    )


def current_status(conn: Any, symbol: str, duration: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT status, reason, details_json, evaluated_at
        FROM high_winrate_strategy_status
        WHERE strategy_key = ? AND symbol = ? AND duration = ?
        """,
        (HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, symbol, duration),
    ).fetchone()
    return {} if row is None else dict(row)


def public_status(row: dict[str, Any], symbol: str, duration: str) -> dict[str, Any]:
    if not row:
        return {"symbol": symbol, "duration": duration, "status": "unknown", "reason": "not_evaluated"}
    details = json.loads(row["details_json"]) if row.get("details_json") else {}
    return {
        "symbol": symbol,
        "duration": duration,
        "status": details.get("status") or row.get("status"),
        "reason": details.get("reason") or row.get("reason"),
        **details,
    }


def set_strategy_slot(conn: Any, symbol: str, duration: str, *, enabled: bool, live_trading_enabled: bool | None) -> None:
    row = strategy_slot(conn, duration)
    live_enabled = live_enabled_for_row(row, live_trading_enabled)
    conn.execute(
        """
        INSERT OR REPLACE INTO auto_trade_strategies(
          strategy_key, duration, enabled, live_trading_enabled, symbol, duration_minutes, qty, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
            duration,
            int(enabled),
            int(live_enabled),
            symbol,
            int(DURATION_TO_MINUTES[duration]),
            float(row["qty"]) if row else DEFAULT_QTY,
            utc_now(),
        ),
    )


def sync_strategy_slot_for_status(conn: Any, symbol: str, duration: str, status: str) -> None:
    if status == STATUS_PAUSED:
        set_strategy_slot(conn, symbol, duration, enabled=False, live_trading_enabled=False)
        return
    live_override = None if status == STATUS_TRADABLE else False
    set_strategy_slot(conn, symbol, duration, enabled=True, live_trading_enabled=live_override)


def strategy_slot(conn: Any, duration: str) -> Any | None:
    return conn.execute(
        """
        SELECT qty, live_trading_enabled
        FROM auto_trade_strategies
        WHERE strategy_key = ? AND duration = ?
        """,
        (HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, duration),
    ).fetchone()


def live_enabled_for_row(row: Any | None, override: bool | None) -> bool:
    if override is not None:
        return bool(override)
    return bool(row["live_trading_enabled"]) if row else False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
