from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.auto_trade_types import AutoTradeSettings
from app.services.ensemble_judge_constants import ENSEMBLE_RANKER_STRATEGY_KEY, STAGE_ENSEMBLE_READY
from app.services.rule_config import DURATION_TO_MINUTES
from app.services.runtime_symbols import DEFAULT_RUNTIME_SYMBOLS
from app.services.strategy_registry import DEFAULT_STRATEGY_KEY, strategy_definition

DEFAULT_SYMBOL = DEFAULT_RUNTIME_SYMBOLS[0]
DEFAULT_DURATION = "10m"
DEFAULT_DURATION_MINUTES = 10
DEFAULT_QTY = 5.0
AUTO_TRADE_SLOT_DURATIONS: tuple[str, ...] = ("10m", "30m", "60m", "1d")


def write_settings(conn: Any, settings: AutoTradeSettings) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO auto_trade_strategies(
          strategy_key, symbol, duration, enabled, live_trading_enabled, duration_minutes, qty, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            settings.strategy_key,
            settings.symbol,
            settings.duration,
            int(settings.enabled),
            int(settings.live_trading_enabled),
            settings.duration_minutes,
            settings.qty,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def settings_from_row(row: Any) -> AutoTradeSettings:
    return AutoTradeSettings(
        strategy_key=row_strategy_key(row),
        enabled=bool(row["enabled"]),
        symbol=str(row["symbol"]).upper(),
        duration=str(row["duration"]),
        duration_minutes=int(row["duration_minutes"]),
        qty=float(row["qty"]),
        live_trading_enabled=bool(row["live_trading_enabled"]),
    )


def default_settings(
    strategy_key: str = DEFAULT_STRATEGY_KEY,
    duration: str = DEFAULT_DURATION,
    symbol: str = DEFAULT_SYMBOL,
) -> AutoTradeSettings:
    minutes = int(DURATION_TO_MINUTES.get(duration, DEFAULT_DURATION_MINUTES))
    return AutoTradeSettings(
        strategy_key=strategy_key,
        enabled=False,
        symbol=symbol.strip().upper(),
        duration=duration,
        duration_minutes=minutes,
        qty=DEFAULT_QTY,
        live_trading_enabled=False,
    )


def strategy_payload(settings: AutoTradeSettings) -> dict[str, Any]:
    strategy = strategy_definition(settings.strategy_key)
    return {
        **settings.to_response(),
        "name": strategy.name,
        "description": strategy.description,
        "requiresVegasConfirmation": strategy.requires_vegas_confirmation,
        "requiresHighWinrateGate": strategy.requires_high_winrate_gate,
        "requiresTradeQualityGate": strategy.requires_trade_quality_gate,
        "signalSource": strategy.signal_source,
        "ruleNames": strategy.rule_names,
        "tradable": strategy.tradable,
        "disabledReason": strategy.disabled_reason,
        "backtestSummary": strategy.backtest_summary,
        "minDailyTrades": strategy.min_daily_trades,
        "requiresKlineFeatures": strategy.requires_kline_features,
        "usesTradePolicyGates": strategy.uses_trade_policy_gates,
        "entryGraceMs": strategy.entry_grace_ms,
        "supportedDurations": sorted(strategy.supported_durations),
    }


def payload_durations(payload: dict[str, Any]) -> list[str]:
    durations = payload.get("supportedDurations") or AUTO_TRADE_SLOT_DURATIONS
    return [duration for duration in AUTO_TRADE_SLOT_DURATIONS if duration in set(durations)]


def ensemble_ranker_settings(conn: Any, by_slot: dict[tuple[str, str, str], Any], symbols: tuple[str, ...]) -> list[AutoTradeSettings]:
    if not ensemble_ranker_visible(conn):
        return []
    return [
        settings_from_row(row) if row is not None else default_settings(ENSEMBLE_RANKER_STRATEGY_KEY, duration, symbol)
        for symbol in symbols
        for duration in AUTO_TRADE_SLOT_DURATIONS
        for row in [by_slot.get((ENSEMBLE_RANKER_STRATEGY_KEY, symbol, duration))]
    ]


def ensemble_ranker_visible(conn: Any) -> bool:
    if not table_exists(conn, "ensemble_stage_status"):
        return False
    row = conn.execute(
        """
        SELECT 1
        FROM ensemble_stage_status
        WHERE confirmed_stage = ?
        LIMIT 1
        """,
        (STAGE_ENSEMBLE_READY,),
    ).fetchone()
    return row is not None


def table_exists(conn: Any, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def row_strategy_key(row: Any) -> str:
    return str(row["strategy_key"] or DEFAULT_STRATEGY_KEY)
