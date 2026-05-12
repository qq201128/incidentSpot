from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.binance_service import fetch_index_price_klines, fetch_premium_index
from app.services.blind_reverse_martingale_strategy import load_blind_rm_settlement_state
from app.services.kline_timing import (
    align_to_rule_interval_bucket_for_duration,
    current_rule_entry_open_time_for_duration,
    is_within_entry_grace,
)
from app.services.n_bar_rm_htf_vol_gate import (
    evaluate_htf_counter_trend_suppress,
    evaluate_volatility_spike_suppress,
    n_bar_rm_htf_vol_gate_enabled,
)
from app.services.kline_timing import rule_interval_ms_for_duration
from app.services.rule_config import RULE_DURATION, SUPPORTED_RULE_DURATIONS
from app.services.strategy_registry import (
    FIVE_BAR_10M_RM_STRATEGY_KEY,
    FOUR_BAR_10M_RM_STRATEGY_KEY,
    THREE_BAR_10M_RM_STRATEGY_KEY,
    strategy_definition,
)

PRICE_DECIMALS = 8
PROBABILITY_DECIMALS = 4
FETCH_KLINE_LIMIT = 80

# duration 到 Binance API interval 的映射
DURATION_TO_BINANCE_INTERVAL: dict[str, str] = {
    "10m": "10m",
    "30m": "30m",
    "60m": "1h",
    "1d": "1d",
}


def _binance_interval_for_duration(duration: str) -> str:
    """将 duration (10m/30m/60m/1d) 转换为 Binance API 的 interval 格式。"""
    interval = DURATION_TO_BINANCE_INTERVAL.get(duration)
    if not interval:
        raise ValueError(f"unsupported duration for index klines: {duration}")
    return interval


def streak_and_last_n_bars(
    symbol: str, entry_open_time: int, n: int, duration: str = RULE_DURATION
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """
    用 GET /fapi/v1/indexPriceKlines 的指数价 OHLC 判最近 n 根是否同向连涨/连跌。
    根据 duration 参数获取对应周期的 K 线数据（10m/30m/1h/1d）。
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    entry_open_time = align_to_rule_interval_bucket_for_duration(int(entry_open_time), duration)
    pair = symbol.upper()
    interval = _binance_interval_for_duration(duration)
    bars = fetch_index_price_klines(pair, interval, limit=FETCH_KLINE_LIMIT)
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


def streak_and_last_n_10m_rest(
    symbol: str, entry_open_time: int, n: int
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """兼容旧代码：使用 10m 周期。"""
    return streak_and_last_n_bars(symbol, entry_open_time, n, RULE_DURATION)


def _event_start_to_boundary_ms(start_time: str, duration: str = RULE_DURATION) -> int:
    normalized = str(start_time).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ms = int(dt.timestamp() * 1000)
    bar = rule_interval_ms_for_duration(duration)
    return (ms // bar) * bar


def last_settled_event_boundary_ms(strategy_key: str, symbol: str, duration: str = RULE_DURATION) -> int | None:
    """最近一次已结算事件的桶起点（不论输赢）。用于要求下一笔的 n 连形态 K 全部落在该桶之后，避免与上一笔共用重叠 K。"""
    conn = get_conn()
    row = conn.execute(
        """
        SELECT start_time FROM events
        WHERE strategy_key = ? AND symbol = ? AND status = 'SETTLED'
        ORDER BY id DESC
        LIMIT 1
        """,
        (strategy_key, symbol.upper()),
    ).fetchone()
    conn.close()
    if row is None or not row["start_time"]:
        return None
    return _event_start_to_boundary_ms(str(row["start_time"]), duration)


def _post_settle_wait_label(streak_length: int) -> str:
    if streak_length == 3:
        return "THREE_BAR_RM_POST_SETTLE_WAIT_FRESH_TRIPLE"
    if streak_length == 4:
        return "FOUR_BAR_RM_POST_SETTLE_WAIT_FRESH_STREAK"
    if streak_length == 5:
        return "FIVE_BAR_RM_POST_SETTLE_WAIT_FRESH_STREAK"
    return f"{streak_length}_BAR_RM_POST_SETTLE_WAIT_FRESH_STREAK"


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
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(
            f"n-bar reverse martingale supports only {sorted(SUPPORTED_RULE_DURATIONS)}, got {duration}"
        )
    if streak_length not in (3, 4, 5):
        raise ValueError(f"streak_length must be 3, 4, or 5, got {streak_length}")
    sym = symbol.upper()
    strategy = strategy_definition(strategy_registry_key)
    rule_name = strategy.rule_names[0] if strategy.rule_names else strategy.key
    pfx = _bar_rm_label_prefix(streak_length)

    state = load_blind_rm_settlement_state(strategy.key, sym)
    if entry_open_time is None:
        open_time = current_rule_entry_open_time_for_duration(duration, now_ms)
    else:
        open_time = align_to_rule_interval_bucket_for_duration(int(entry_open_time), duration)
    entry_price = float(fetch_premium_index(sym).get("indexPrice") or 0)
    if entry_price <= 0:
        raise ValueError("latest index price unavailable")

    entry_window_passed = is_within_entry_grace(
        open_time,
        now_ms,
        grace_ms=int(strategy.entry_grace_ms),
    )

    n_loss = state.consecutive_losses
    label: str
    direction: str
    pattern_ok = False
    post_settle_block = False
    last_settled_ot: int | None = None
    oldest_bar_open: int | None = None
    fresh_floor: int | None = None

    # 使用对应 duration 的指数 K 线判定连续阳/阴形态
    # 连亏后不补单：下一笔仍须再次满足形态；且形态所含 K 须全部落在「上一笔已结算事件」对应桶之后（与猜赢后等待新鲜形态同理）。
    streak_raw, last_bars = streak_and_last_n_bars(sym, open_time, streak_length, duration)
    last_settled_ot = last_settled_event_boundary_ms(strategy.key, sym, duration)
    streak = streak_raw
    if streak_raw is not None and last_settled_ot is not None and last_bars:
        oldest_bar_open = min(int(b["openTime"]) for b in last_bars)
        fresh_floor = last_settled_ot + rule_interval_ms_for_duration(duration)
        if oldest_bar_open < fresh_floor:
            streak = None
            post_settle_block = True

    if post_settle_block:
        direction = "up"
        label = _post_settle_wait_label(streak_length)
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

    # HTF 1h + 10m TR/ATR：三连 / 四连 / 五连均走本函数，共用 gate（见 n_bar_rm_htf_vol_gate）。
    vol_suppress = False
    htf_suppress = False
    vol_gate_meta: dict[str, Any] = {}
    htf_gate_meta: dict[str, Any] = {}
    suppress_label: str | None = None
    if pattern_ok and n_bar_rm_htf_vol_gate_enabled():
        vol_suppress, vol_gate_meta = evaluate_volatility_spike_suppress(sym, int(open_time))
        htf_suppress, htf_gate_meta = evaluate_htf_counter_trend_suppress(
            sym, int(open_time), direction
        )
        if vol_suppress:
            suppress_label = f"{pfx}_VOL_SPIKE_SKIP_TR_VS_ATR"
        elif htf_suppress:
            suppress_label = f"{pfx}_HTF_1H_SMA_COUNTER_TREND_SKIP"

    confidence = 0.5
    probability_up = confidence if direction == "up" else 1.0 - confidence
    trade_quality_passed = (
        entry_window_passed
        and pattern_ok
        and not vol_suppress
        and not htf_suppress
    )

    if not entry_window_passed:
        certainty_label_out = f"{pfx}_WAIT_WINDOW"
    elif suppress_label:
        certainty_label_out = suppress_label
    else:
        certainty_label_out = label

    return {
        "symbol": sym,
        "strategy_key": strategy.key,
        "duration": duration,
        "open_time": int(open_time),
        "entry_price": round(float(entry_price), PRICE_DECIMALS),
        "direction": direction,
        "probability_up": round(probability_up, PROBABILITY_DECIMALS),
        "confidence": confidence,
        "certainty_label": certainty_label_out,
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
            "order_qty=flat_panel_base_no_mg",
            "entry_policy=pattern_only_no_recovery",
            f"index_streak_len={streak_length}",
            "streak_source=binance_index_price_10m",
            f"pattern_ok={pattern_ok}",
            f"entry_window_passed={entry_window_passed}",
            f"direction_mode={label}",
            f"n_bar_htf_vol_gate_enabled={n_bar_rm_htf_vol_gate_enabled()}",
            f"vol_spike_suppress={vol_suppress}",
            f"htf_counter_trend_suppress={htf_suppress}",
            *(
                [f"vol_gate={vol_gate_meta.get('reason')}", f"vol_meta={vol_gate_meta}"]
                if vol_gate_meta
                else []
            ),
            *(
                [f"htf_gate={htf_gate_meta.get('reason')}", f"htf_meta={htf_gate_meta}"]
                if htf_gate_meta
                else []
            ),
            *(
                [
                    "post_settle_pattern_suppressed=1",
                    f"last_settled_event_boundary_ms={last_settled_ot}",
                    f"require_streak_bars_open_on_or_after_ms={fresh_floor}",
                    f"oldest_bar_among_streak_candidates_ms={oldest_bar_open}",
                ]
                if post_settle_block
                and last_settled_ot is not None
                and fresh_floor is not None
                and oldest_bar_open is not None
                else []
            ),
        ],
        "orderbook": {
            "strategy": rule_name,
            "consecutiveLosses": n_loss,
            "entryWindowPassed": entry_window_passed,
            "patternOk": pattern_ok,
            "streakLength": streak_length,
            "cycleAnchorPredicted": None,
            "recoverySameAsAnchor": False,
            "volSpikeSuppress": vol_suppress,
            "htfCounterTrendSuppress": htf_suppress,
            "volGate": vol_gate_meta,
            "htfGate": htf_gate_meta,
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
