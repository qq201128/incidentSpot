from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.binance_service import fetch_index_price_klines, fetch_premium_index
from app.services.blind_reverse_martingale_strategy import (
    BLIND_RM_MAX_CONSECUTIVE_LOSSES,
    load_blind_rm_settlement_state,
)
from app.services.kline_timing import (
    RULE_INTERVAL_MS,
    align_to_rule_interval_bucket,
    current_rule_entry_open_time,
    is_within_entry_grace,
)
from app.services.rule_config import RULE_DURATION
from app.services.strategy_registry import (
    FIVE_BAR_10M_RM_STRATEGY_KEY,
    FOUR_BAR_10M_RM_STRATEGY_KEY,
    THREE_BAR_10M_RM_STRATEGY_KEY,
    strategy_definition,
)

PRICE_DECIMALS = 8
PROBABILITY_DECIMALS = 4
# Binance 无原生 10m：indexPriceKlines 的 10m 由 1m 指数 K 按 UTC 聚合成 10m 桶（与 APP 「指数 / Index」10 分钟一致），不用合约成交价 K。
FETCH_10M_KLINE_LIMIT = 80


def streak_and_last_n_10m_rest(
    symbol: str, entry_open_time: int, n: int
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """
    用 GET /fapi/v1/indexPriceKlines 的 **指数价 10m OHLC** 判最近 n 根是否同向连涨/连跌（与「指数 K 线」一致）。
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    entry_open_time = align_to_rule_interval_bucket(int(entry_open_time))
    pair = symbol.upper()
    bars = fetch_index_price_klines(pair, "10m", limit=FETCH_10M_KLINE_LIMIT)
    if len(bars) < n:
        return None, None
    completed = [b for b in bars if int(b["openTime"]) < int(entry_open_time)]
    if len(completed) < n:
        return None, None
    last_n = completed[-n:]
    kinds: list[str] = []
    for b in last_n:
        o = float(b["open"])
        c = float(b["close"])
        if c > o:
            kinds.append("bull")
        elif c < o:
            kinds.append("bear")
        else:
            return None, last_n
    if kinds == ["bull"] * n:
        return "bull", last_n
    if kinds == ["bear"] * n:
        return "bear", last_n
    return None, last_n


def _event_start_to_boundary_ms(start_time: str) -> int:
    normalized = str(start_time).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ms = int(dt.timestamp() * 1000)
    bar = int(RULE_INTERVAL_MS)
    return (ms // bar) * bar


def _last_win_prediction_open_time_ms(strategy_key: str, symbol: str) -> int | None:
    """最近一次「猜对」对应的预测 open_time（10m 边界）；predictions 未回填时用 events 起始时间对齐到边界。"""
    conn = get_conn()
    row = conn.execute(
        """
        SELECT open_time FROM predictions
        WHERE strategy_key = ? AND symbol = ? AND prediction_correct = 1
        ORDER BY id DESC
        LIMIT 1
        """,
        (strategy_key, symbol.upper()),
    ).fetchone()
    if row is not None and row["open_time"] is not None:
        conn.close()
        return int(row["open_time"])
    row = conn.execute(
        """
        SELECT start_time FROM events
        WHERE strategy_key = ? AND symbol = ? AND status = 'SETTLED'
          AND ai_prediction_correct = 1
        ORDER BY id DESC
        LIMIT 1
        """,
        (strategy_key, symbol.upper()),
    ).fetchone()
    conn.close()
    if row is None or not row["start_time"]:
        return None
    return _event_start_to_boundary_ms(str(row["start_time"]))


def _post_win_wait_label(streak_length: int) -> str:
    if streak_length == 3:
        return "THREE_BAR_RM_POST_WIN_WAIT_FRESH_TRIPLE"
    if streak_length == 4:
        return "FOUR_BAR_RM_POST_WIN_WAIT_FRESH_STREAK"
    if streak_length == 5:
        return "FIVE_BAR_RM_POST_WIN_WAIT_FRESH_STREAK"
    return f"{streak_length}_BAR_RM_POST_WIN_WAIT_FRESH_STREAK"


def _bar_rm_label_prefix(streak_length: int) -> str:
    return {3: "THREE_BAR_RM", 4: "FOUR_BAR_RM", 5: "FIVE_BAR_RM"}.get(
        streak_length, f"{streak_length}_BAR_RM"
    )


def predict_n_bar_10m_reverse_martingale_direction(
    symbol: str,
    duration: str = RULE_DURATION,
    *,
    streak_length: int,
    strategy_registry_key: str,
    entry_open_time: int | None = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    if duration != RULE_DURATION:
        raise ValueError(
            f"n-bar 10m reverse martingale supports only {RULE_DURATION}, got {duration}"
        )
    if streak_length not in (3, 4, 5):
        raise ValueError(f"streak_length must be 3, 4, or 5, got {streak_length}")
    sym = symbol.upper()
    strategy = strategy_definition(strategy_registry_key)
    rule_name = strategy.rule_names[0] if strategy.rule_names else strategy.key
    pfx = _bar_rm_label_prefix(streak_length)

    state = load_blind_rm_settlement_state(strategy.key, sym)
    if entry_open_time is None:
        open_time = current_rule_entry_open_time(now_ms)
    else:
        open_time = align_to_rule_interval_bucket(int(entry_open_time))
    entry_price = float(fetch_premium_index(sym).get("indexPrice") or 0)
    if entry_price <= 0:
        raise ValueError("latest index price unavailable")

    entry_window_passed = is_within_entry_grace(
        open_time,
        now_ms,
        grace_ms=int(strategy.entry_grace_ms),
    )

    n_loss = state.consecutive_losses
    # 本轮首笔已结算亏损单的预测方向；倍投各档与该方向相同（同向加码）。
    cycle_anchor_predicted = state.cycle_anchor_direction()
    label: str
    direction: str
    pattern_ok = False
    post_win_block = False
    last_win_ot: int | None = None
    oldest_bar_open: int | None = None
    fresh_floor: int | None = None

    # 倍投阶段（已有连亏、尚未满 4 笔）：每一根新的 10m 都应继续下单并加码档，不再要求「前 n 根连」形态。
    if 1 <= n_loss < BLIND_RM_MAX_CONSECUTIVE_LOSSES:
        anchor = cycle_anchor_predicted
        if anchor:
            direction = anchor
            label = f"{pfx}_SAME_AS_CYCLE"
        elif state.rows_considered:
            raw = str((state.rows_considered[-1].get("pred") or "")).lower()
            if raw in ("up", "down"):
                direction = raw
                label = f"{pfx}_SAME_ANCHOR_FALLBACK"
            else:
                direction = "up"
                label = f"{pfx}_SAME_LAST_UNKNOWN"
        else:
            direction = "up"
            label = f"{pfx}_SAME_NO_ROWS"
        pattern_ok = True
    else:
        streak_raw, last_bars = streak_and_last_n_10m_rest(sym, open_time, streak_length)
        last_win_ot = _last_win_prediction_open_time_ms(strategy.key, sym)
        streak = streak_raw
        post_win_block = False
        oldest_bar_open = None
        fresh_floor = None
        if streak_raw is not None and last_win_ot is not None and last_bars:
            oldest_bar_open = min(int(b["openTime"]) for b in last_bars)
            fresh_floor = last_win_ot + int(RULE_INTERVAL_MS)
            if oldest_bar_open < fresh_floor:
                streak = None
                post_win_block = True

        if post_win_block:
            direction = "up"
            label = _post_win_wait_label(streak_length)
            pattern_ok = False
        elif streak == "bull":
            direction = "down"
            label = f"{pfx}_STREAK_BULL_REV"
            pattern_ok = True
        elif streak == "bear":
            direction = "up"
            label = f"{pfx}_STREAK_BEAR_REV"
            pattern_ok = True
        else:
            direction = "up"
            label = f"{pfx}_NO_PATTERN"
            pattern_ok = False

    confidence = 0.5
    probability_up = confidence if direction == "up" else 1.0 - confidence
    trade_quality_passed = entry_window_passed and pattern_ok

    return {
        "symbol": sym,
        "strategy_key": strategy.key,
        "duration": duration,
        "open_time": int(open_time),
        "entry_price": round(float(entry_price), PRICE_DECIMALS),
        "direction": direction,
        "probability_up": round(probability_up, PROBABILITY_DECIMALS),
        "confidence": confidence,
        "certainty_label": (f"{pfx}_WAIT_WINDOW" if not entry_window_passed else label),
        "threshold": None,
        "trade_quality_score": 1.0 if trade_quality_passed else 0.0,
        "trade_quality_passed": trade_quality_passed,
        "trade_quality_gate": rule_name,
        "high_winrate_gate": None,
        "high_winrate_rule": rule_name,
        "high_winrate_gate_passed": None,
        "high_winrate_gate_value": None,
        "high_winrate_gate_min": None,
        "signal_source": strategy.signal_source,
        "rule_score": float(n_loss),
        "rule_reasons": [
            f"rule={rule_name}",
            f"consecutive_losses={n_loss}",
            "mg_steps_usdt=10/20/45",
            f"index_streak_len={streak_length}",
            "streak_source=binance_index_price_10m",
            f"pattern_ok={pattern_ok}",
            f"entry_window_passed={entry_window_passed}",
            f"direction_mode={label}",
            *(
                [
                    f"cycle_anchor_predicted={cycle_anchor_predicted}",
                    "recovery_direction=same_as_cycle_anchor",
                ]
                if 1 <= n_loss < BLIND_RM_MAX_CONSECUTIVE_LOSSES
                else []
            ),
            *(
                [
                    "post_win_pattern_suppressed=1",
                    f"last_win_prediction_open_time_ms={last_win_ot}",
                    f"require_streak_bars_open_on_or_after_ms={fresh_floor}",
                    f"oldest_bar_among_streak_candidates_ms={oldest_bar_open}",
                ]
                if post_win_block and last_win_ot is not None and fresh_floor is not None and oldest_bar_open is not None
                else []
            ),
        ],
        "orderbook": {
            "strategy": rule_name,
            "consecutiveLosses": n_loss,
            "entryWindowPassed": entry_window_passed,
            "patternOk": pattern_ok,
            "streakLength": streak_length,
            "cycleAnchorPredicted": cycle_anchor_predicted
            if 1 <= n_loss < BLIND_RM_MAX_CONSECUTIVE_LOSSES
            else None,
            "recoverySameAsAnchor": bool(1 <= n_loss < BLIND_RM_MAX_CONSECUTIVE_LOSSES),
        },
        "timeframe_votes": [],
    }


def predict_three_bar_10m_reverse_martingale_direction(
    symbol: str,
    duration: str = RULE_DURATION,
    *,
    entry_open_time: int | None = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    return predict_n_bar_10m_reverse_martingale_direction(
        symbol,
        duration,
        streak_length=3,
        strategy_registry_key=THREE_BAR_10M_RM_STRATEGY_KEY,
        entry_open_time=entry_open_time,
        now_ms=now_ms,
    )


def predict_four_bar_10m_reverse_martingale_direction(
    symbol: str,
    duration: str = RULE_DURATION,
    *,
    entry_open_time: int | None = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    return predict_n_bar_10m_reverse_martingale_direction(
        symbol,
        duration,
        streak_length=4,
        strategy_registry_key=FOUR_BAR_10M_RM_STRATEGY_KEY,
        entry_open_time=entry_open_time,
        now_ms=now_ms,
    )


def predict_five_bar_10m_reverse_martingale_direction(
    symbol: str,
    duration: str = RULE_DURATION,
    *,
    entry_open_time: int | None = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    return predict_n_bar_10m_reverse_martingale_direction(
        symbol,
        duration,
        streak_length=5,
        strategy_registry_key=FIVE_BAR_10M_RM_STRATEGY_KEY,
        entry_open_time=entry_open_time,
        now_ms=now_ms,
    )
