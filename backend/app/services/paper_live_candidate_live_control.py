from __future__ import annotations

import sqlite3
from typing import Any

from app.db.session import get_conn
from app.services.auto_trade_settings_payloads import DEFAULT_QTY, write_settings
from app.services.auto_trade_settings_validation import validated_auto_trade_settings
from app.services.auto_trade_types import AutoTradeSettings
from app.services.live_readiness_gate import live_readiness_gate
from app.services.paper_live_candidate_service import (
    STATUS_STABLE,
    paper_live_candidate_report,
    refresh_paper_live_candidate_states,
)
from app.services.paper_live_report_cache import store_paper_live_report_cache
from app.services.rule_config import DURATION_TO_MINUTES

_REPORT_CANDIDATE_FIELDS = ("allCandidates", "candidates", "stable", "collecting", "failed")
_LIVE_READINESS_METRIC_KEYS = (
    "consecutiveLosses",
    "sampleCount",
    "winRate",
    "profitFactor",
    "avgReturn",
    "paperStability",
)


def set_candidate_live_trading(
    symbol: str,
    duration: str,
    *,
    candidate_key: str,
    live_trading_enabled: bool,
) -> dict[str, Any]:
    sym = symbol.strip().upper()
    report = _live_control_report(sym, duration, candidate_key)
    candidate = _candidate_for_live_control(report, candidate_key, live_trading_enabled)
    settings = _settings(candidate, sym, duration, live_trading_enabled=live_trading_enabled)
    live_state = _write_validated_settings(settings)
    updated_report = _report_with_live_state(report, candidate_key, settings, live_state)
    store_paper_live_report_cache(sym, duration, updated_report)
    return {
        "ok": True,
        "candidateKey": candidate_key,
        "strategyKey": settings.strategy_key,
        "liveTradingEnabled": live_trading_enabled,
        "report": updated_report,
    }


def _live_control_report(symbol: str, duration: str, candidate_key: str) -> dict[str, Any]:
    if _candidate_status_snapshot_exists(symbol, duration, candidate_key):
        report = _snapshot_report_or_full_refresh(symbol, duration)
        if _candidate_by_key(report, candidate_key) is not None:
            return report
    return refresh_paper_live_candidate_states(symbol, duration)


def _candidate_status_snapshot_exists(symbol: str, duration: str, candidate_key: str) -> bool:
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT 1
            FROM paper_live_candidate_status
            WHERE candidate_key = ? AND symbol = ? AND duration = ?
            LIMIT 1
            """,
            (candidate_key, symbol.strip().upper(), duration),
        ).fetchone()
        return row is not None
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return False
        raise
    finally:
        conn.close()


def _snapshot_report_or_full_refresh(symbol: str, duration: str) -> dict[str, Any]:
    try:
        return paper_live_candidate_report(symbol, duration)
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return refresh_paper_live_candidate_states(symbol, duration)
        raise


def _candidate_for_live_control(
    report: dict[str, Any],
    candidate_key: str,
    live_trading_enabled: bool,
) -> dict[str, Any]:
    candidate = _candidate_by_key(report, candidate_key)
    if candidate is None:
        raise ValueError(f"paper-live candidate not found: {candidate_key}")
    if live_trading_enabled and candidate.get("status") != STATUS_STABLE:
        raise ValueError(f"candidate is not stable: {candidate_key}")
    if not candidate.get("strategyKey"):
        raise ValueError(f"candidate has no strategy key: {candidate_key}")
    return candidate


def _candidate_by_key(report: dict[str, Any], candidate_key: str) -> dict[str, Any] | None:
    rows = report.get("allCandidates") if isinstance(report.get("allCandidates"), list) else []
    return next((row for row in rows if row.get("candidateKey") == candidate_key), None)


def _settings(
    candidate: dict[str, Any],
    symbol: str,
    duration: str,
    *,
    live_trading_enabled: bool,
) -> AutoTradeSettings:
    row = _slot_row(str(candidate["strategyKey"]), symbol, duration)
    return AutoTradeSettings(
        strategy_key=str(candidate["strategyKey"]),
        enabled=True,
        symbol=symbol,
        duration=duration,
        duration_minutes=int(DURATION_TO_MINUTES[duration]),
        qty=float(row["qty"]) if row else DEFAULT_QTY,
        live_trading_enabled=live_trading_enabled,
    )


def _slot_row(strategy_key: str, symbol: str, duration: str) -> Any | None:
    conn = get_conn()
    try:
        return conn.execute(
            """
            SELECT qty
            FROM auto_trade_strategies
            WHERE strategy_key = ? AND symbol = ? AND duration = ?
            """,
            (strategy_key, symbol, duration),
        ).fetchone()
    finally:
        conn.close()


def _write_validated_settings(settings: AutoTradeSettings) -> dict[str, Any]:
    validated = validated_auto_trade_settings(settings)
    conn = get_conn()
    try:
        write_settings(conn, validated)
        row = conn.execute(
            """
            SELECT enabled, live_trading_enabled, qty, updated_at
            FROM auto_trade_strategies
            WHERE strategy_key = ? AND symbol = ? AND duration = ?
            """,
            (validated.strategy_key, validated.symbol, validated.duration),
        ).fetchone()
        conn.commit()
        return _live_state_from_slot(row, validated)
    finally:
        conn.close()


def _live_state_from_slot(row: Any | None, settings: AutoTradeSettings) -> dict[str, Any]:
    if row is None:
        return {
            "autoTradeEnabled": settings.enabled,
            "liveTradingEnabled": settings.live_trading_enabled,
            "qty": settings.qty,
            "updatedAt": None,
        }
    return {
        "autoTradeEnabled": bool(row["enabled"]),
        "liveTradingEnabled": bool(row["live_trading_enabled"]),
        "qty": float(row["qty"]),
        "updatedAt": row["updated_at"],
    }


def _report_with_live_state(
    report: dict[str, Any],
    candidate_key: str,
    settings: AutoTradeSettings,
    live_state: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(report)
    for field in _REPORT_CANDIDATE_FIELDS:
        rows = report.get(field)
        if isinstance(rows, list):
            updated[field] = [
                _candidate_with_live_state(row, candidate_key, settings, live_state)
                for row in rows
            ]
    updated["realTradingEnabled"] = _report_has_live_enabled(updated, candidate_key, live_state)
    return updated


def _candidate_with_live_state(
    row: Any,
    candidate_key: str,
    settings: AutoTradeSettings,
    live_state: dict[str, Any],
) -> Any:
    if not isinstance(row, dict) or not _same_candidate(row, candidate_key, settings.strategy_key):
        return row
    updated = {
        **row,
        "autoTradeEnabled": live_state["autoTradeEnabled"],
        "liveTradingEnabled": live_state["liveTradingEnabled"],
        "liveTradingUpdatedAt": live_state["updatedAt"],
    }
    metrics = updated.get("metrics") if isinstance(updated.get("metrics"), dict) else {}
    if _has_live_readiness_metrics(metrics):
        updated["liveReadiness"] = live_readiness_gate(
            metrics,
            updated.get("status") or updated.get("paperLiveStatus"),
            status_reason=updated.get("reason"),
            real_trading_enabled=bool(live_state["liveTradingEnabled"]),
        )
    return updated


def _same_candidate(row: dict[str, Any], candidate_key: str, strategy_key: str) -> bool:
    return row.get("candidateKey") == candidate_key or row.get("strategyKey") == strategy_key


def _has_live_readiness_metrics(metrics: dict[str, Any]) -> bool:
    return all(key in metrics for key in _LIVE_READINESS_METRIC_KEYS)


def _report_has_live_enabled(
    report: dict[str, Any],
    candidate_key: str,
    live_state: dict[str, Any],
) -> bool:
    if bool(live_state["liveTradingEnabled"]):
        return True
    for field in _REPORT_CANDIDATE_FIELDS:
        rows = report.get(field)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("candidateKey") != candidate_key and row.get("liveTradingEnabled"):
                return True
    return False
