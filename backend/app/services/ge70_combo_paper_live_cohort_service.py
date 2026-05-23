from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.auto_trade_types import AutoTradeSettings

from app.db.session import get_conn
from app.services.factor_combo_batch_simulation_service import create_batch_combo_simulation_trade
from app.services.factor_combo_simulation_keys import simulation_strategy_key_for_factor_name
from app.services.factor_combination_live_service import rebuild_combination_signal_watchlist
from app.services.factor_learning_common import utc_now
from app.services.factor_mined_library import MINED_FACTOR_LIBRARY_PATH, load_mined_factor_library
from app.services.high_winrate_combo_cache_service import (
    get_cached_high_winrate_combo_ranking,
    save_cached_high_winrate_combo_ranking,
)
from app.services.high_winrate_strategy_demotion import promote_high_winrate_strategy
from app.services.factor_combo_strategy import predict_factor_combo_row_direction
from app.services.kline_timing import current_rule_entry_open_time_for_duration
from app.services.prediction_cache_service import save_prediction
from app.services.rule_config import DURATION_TO_MINUTES, SUPPORTED_RULE_DURATIONS
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY, strategy_entry_grace_ms

GE70_MIN_WIN_RATE = 0.70
COHORT_VERSION = "ge70_mined_library_paper_live_v1"
# Batch shadow path only runs comboRank > 1; start at 2 so every cohort combo is simulated.
COHORT_COMBO_RANK_START = 2


def load_ge70_mined_combo_rows(*, min_win_rate: float = GE70_MIN_WIN_RATE) -> list[dict[str, Any]]:
    library = load_mined_factor_library(MINED_FACTOR_LIBRARY_PATH)
    rows: list[dict[str, Any]] = []
    for row in library.get("factors") or []:
        metrics = row.get("metrics") or {}
        win_rate = metrics.get("winRate")
        if win_rate is None or float(win_rate) < min_win_rate:
            continue
        if not row.get("members"):
            continue
        rows.append(row)
    rows.sort(
        key=lambda item: (
            str(item.get("duration") or ""),
            -float((item.get("metrics") or {}).get("winRate") or 0.0),
        ),
    )
    return rows


def bootstrap_ge70_paper_live_cohort(
    symbol: str,
    *,
    duration: str | None = None,
    seed_predictions: bool = False,
    rebuild_watchlist: bool = False,
) -> dict[str, Any]:
    sym = symbol.strip().upper()
    selected = load_ge70_mined_combo_rows()
    if duration is not None:
        if duration not in SUPPORTED_RULE_DURATIONS:
            raise ValueError(f"unsupported duration: {duration}")
        selected = [row for row in selected if str(row.get("duration")) == duration]
    grouped = _group_rows_by_duration(selected)
    if not grouped:
        return _empty_bootstrap_report(sym, duration, seed_predictions=seed_predictions)

    cache_reports: list[dict[str, Any]] = []
    slot_reports: list[dict[str, Any]] = []
    for dur, rows in sorted(grouped.items()):
        ranking = [_ranking_row_from_mined(row, rank) for rank, row in enumerate(rows, start=COHORT_COMBO_RANK_START)]
        report = {
            "version": COHORT_VERSION,
            "updatedAt": utc_now(),
            "symbol": sym,
            "duration": dur,
            "target": {
                "targetCount": len(ranking),
                "minWinRate": GE70_MIN_WIN_RATE,
                "cohort": "mined_factor_library_ge70",
            },
            "ranking": ranking,
            "paperLiveSimulation": [],
        }
        save_cached_high_winrate_combo_ranking(report)
        promote_high_winrate_strategy(sym, dur)
        slot_reports.append(_ensure_batch_simulation_slots(sym, dur, ranking))
        cache_reports.append(
            {
                "duration": dur,
                "comboCount": len(ranking),
                "topWinRate": ranking[0].get("winRate"),
                "strategyKeys": [simulation_strategy_key_for_factor_name(str(row["factorName"])) for row in ranking[:3]],
            }
        )

    watchlist = _safe_rebuild_watchlist(sym) if rebuild_watchlist else {"skipped": True, "reason": "rebuild_watchlist_disabled"}
    prediction_report = (
        _seed_batch_paper_live_predictions(sym, grouped)
        if seed_predictions
        else {"skipped": True, "reason": "seed_predictions_disabled"}
    )
    return {
        "symbol": sym,
        "duration": duration,
        "totalCombos": len(selected),
        "durations": cache_reports,
        "batchStrategySlots": slot_reports,
        "watchlist": {
            "eligibleTotal": watchlist.get("eligibleTotal"),
            "total": watchlist.get("total"),
            "signalFailures": len(watchlist.get("signalFailures") or []),
            "cacheIssues": watchlist.get("cacheIssues") or [],
        },
        "predictions": prediction_report,
        "message": (
            f"已将 {len(selected)} 条 ≥70% 组合写入高胜率模拟缓存并启用批量模拟策略；"
            "信号触发后会在「事件合约记录」生成 SIM 模拟事件（规则列显示组合名）。"
            "请保持后端自动预测运行。"
        ),
        "eventRecordHint": "事件合约记录 · 模式 SIM · 规则列=组合因子名",
    }


def _group_rows_by_duration(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dur = str(row.get("duration") or "")
        if dur in SUPPORTED_RULE_DURATIONS:
            grouped[dur].append(row)
    return dict(grouped)


def _ranking_row_from_mined(row: dict[str, Any], rank: int) -> dict[str, Any]:
    metrics = row.get("metrics") or {}
    return {
        "rank": rank,
        "comboRank": rank,
        "factorName": str(row["factorName"]),
        "factorDisplayName": row.get("factorDisplayName") or row.get("factorName"),
        "description": row.get("description") or row.get("factorDisplayName"),
        "formula": str(row.get("formula") or row["factorName"]),
        "method": str(row.get("method") or ""),
        "members": list(row.get("members") or []),
        "threshold": row.get("threshold"),
        "winRate": float(metrics["winRate"]),
        "profitFactor": metrics.get("profitFactor"),
        "trades": int(metrics.get("totalPeriods") or 0),
        "totalPeriods": int(metrics.get("totalPeriods") or 0),
        "minTrades": int(metrics.get("totalPeriods") or 0),
        "avgReturn": metrics.get("longShortReturn"),
        "source": "mined_factor_library_ge70_cohort",
    }


def _ensure_batch_simulation_slots(
    symbol: str,
    duration: str,
    ranking: list[dict[str, Any]],
) -> dict[str, Any]:
    ts = utc_now()
    enabled = 0
    conn = get_conn()
    try:
        for row in ranking:
            strategy_key = simulation_strategy_key_for_factor_name(str(row["factorName"]))
            conn.execute(
                """
                INSERT OR REPLACE INTO auto_trade_strategies(
                  strategy_key, duration, enabled, live_trading_enabled, symbol, duration_minutes, qty, updated_at
                )
                VALUES(?, ?, 1, 0, ?, ?, 5, ?)
                """,
                (
                    strategy_key,
                    duration,
                    symbol,
                    int(DURATION_TO_MINUTES[duration]),
                    ts,
                ),
            )
            enabled += 1
        conn.commit()
    finally:
        conn.close()
    return {"duration": duration, "enabledSlots": enabled}


def _safe_rebuild_watchlist(symbol: str) -> dict[str, Any]:
    try:
        return rebuild_combination_signal_watchlist(symbol, limit=None, top_per_duration=None)
    except Exception as exc:
        return {
            "eligibleTotal": 0,
            "total": 0,
            "signalFailures": [{"stage": "rebuild_watchlist", "error": str(exc)}],
            "cacheIssues": [],
        }


def _seed_batch_paper_live_predictions(
    symbol: str,
    grouped: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    saved: list[str] = []
    trades: list[str] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    parent_by_duration = _factor_combo_parent_settings(symbol)

    for dur, rows in sorted(grouped.items()):
        entry_open_time = current_rule_entry_open_time_for_duration(dur)
        grace_ms = strategy_entry_grace_ms(FACTOR_COMBO_STRATEGY_KEY)
        parent = parent_by_duration.get(dur)
        if parent is None:
            failures.append({"duration": dur, "error": "factor_combo_parent_settings_missing"})
            continue
        cached = get_cached_high_winrate_combo_ranking(symbol, dur)
        ranking = (cached or {}).get("ranking") if (cached or {}).get("version") == COHORT_VERSION else rows
        if not isinstance(ranking, list):
            ranking = rows
        for row in ranking:
            factor_name = str(row.get("factorName") or "")
            if int(row.get("comboRank") or 0) < COHORT_COMBO_RANK_START:
                skipped.append({"duration": dur, "factorName": factor_name, "reason": "combo_rank_below_batch_threshold"})
                continue
            try:
                prediction = predict_factor_combo_row_direction(
                    symbol,
                    dur,
                    row,
                    entry_open_time=entry_open_time,
                    entry_grace_ms=grace_ms,
                )
                if not save_prediction(prediction, allow_existing=True):
                    skipped.append(
                        {
                            "duration": dur,
                            "factorName": factor_name,
                            "strategyKey": prediction["strategy_key"],
                            "reason": "prediction_already_exists",
                        }
                    )
                    continue
                saved.append(prediction["strategy_key"])
                trade = create_batch_combo_simulation_trade(parent, prediction)
                if trade is not None:
                    trades.append(prediction["strategy_key"])
            except Exception as exc:
                failures.append({"duration": dur, "factorName": factor_name, "error": str(exc)})

    return {
        "savedPredictions": len(saved),
        "simulationTrades": len(trades),
        "skipped": len(skipped),
        "failures": failures[:20],
        "failureCount": len(failures),
    }


def _factor_combo_parent_settings(symbol: str) -> dict[str, AutoTradeSettings]:
    from app.services.auto_trade_service import list_auto_trade_settings

    selected: dict[str, AutoTradeSettings] = {}
    for settings in list_auto_trade_settings():
        if settings.strategy_key != FACTOR_COMBO_STRATEGY_KEY:
            continue
        if settings.symbol.upper() != symbol.upper():
            continue
        selected[settings.duration] = settings
    return selected


def _empty_bootstrap_report(
    symbol: str,
    duration: str | None,
    *,
    seed_predictions: bool,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "duration": duration,
        "totalCombos": 0,
        "durations": [],
        "batchStrategySlots": [],
        "watchlist": {"eligibleTotal": 0, "total": 0, "signalFailures": 0, "cacheIssues": []},
        "predictions": {"skipped": True, "reason": "no_ge70_combos"},
        "message": "未在 mined_factor_library 中找到 ≥70% 组合。",
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
