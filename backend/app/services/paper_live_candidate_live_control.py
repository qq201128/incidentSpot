from __future__ import annotations

from typing import Any

from app.db.session import get_conn
from app.services.auto_trade_settings_payloads import DEFAULT_QTY, write_settings
from app.services.auto_trade_settings_validation import validated_auto_trade_settings
from app.services.auto_trade_types import AutoTradeSettings
from app.services.paper_live_candidate_service import STATUS_STABLE, refresh_paper_live_candidate_states
from app.services.rule_config import DURATION_TO_MINUTES


def set_candidate_live_trading(
    symbol: str,
    duration: str,
    *,
    candidate_key: str,
    live_trading_enabled: bool,
) -> dict[str, Any]:
    sym = symbol.strip().upper()
    report = refresh_paper_live_candidate_states(sym, duration)
    candidate = _stable_candidate(report, candidate_key)
    settings = _settings(candidate, sym, duration, live_trading_enabled=live_trading_enabled)
    _write_validated_settings(settings)
    updated_report = refresh_paper_live_candidate_states(sym, duration)
    return {
        "ok": True,
        "candidateKey": candidate_key,
        "strategyKey": settings.strategy_key,
        "liveTradingEnabled": live_trading_enabled,
        "report": updated_report,
    }


def _stable_candidate(report: dict[str, Any], candidate_key: str) -> dict[str, Any]:
    candidate = _candidate_by_key(report, candidate_key)
    if candidate is None:
        raise ValueError(f"paper-live candidate not found: {candidate_key}")
    if candidate.get("status") != STATUS_STABLE:
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


def _write_validated_settings(settings: AutoTradeSettings) -> None:
    validated = validated_auto_trade_settings(settings)
    conn = get_conn()
    try:
        write_settings(conn, validated)
        conn.commit()
    finally:
        conn.close()
