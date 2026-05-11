from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from app.db.session import get_conn
from app.services.binance_service import fetch_premium_index
from app.services.kline_timing import is_within_entry_grace
from app.services.rule_config import RULE_DURATION, SUPPORTED_RULE_DURATIONS
from app.services.strategy_registry import (
    BLIND_REVERSE_MARTINGALE_RULE_NAME,
    BLIND_REVERSE_MARTINGALE_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS,
    strategy_definition,
)

PRICE_DECIMALS = 8
PROBABILITY_DECIMALS = 4

# 首轮之后的固定倍投档位（USDT）
BLIND_RM_MARTINGALE_AMOUNTS_USDT: tuple[float, float, float] = (10.0, 20.0, 45.0)
# 首轮随意 + 最多 3 笔反向倍投（10/20/45）；连亏满 4 笔（含 45 未中）→ 下一轮重新随机首单
# 连续亏损在遇到任意一笔「猜对」时即清零 → 下一轮重新随机首单（与 45 踏空后一致）
BLIND_RM_MAX_CONSECUTIVE_LOSSES = 4


@dataclass(frozen=True)
class BlindRmSettlementState:
    """当前已结算行情中，从最新一笔往回数的连续亏损笔数。"""

    consecutive_losses: int
    rows_considered: list[dict[str, Any]]

    def cycle_anchor_direction(self) -> str | None:
        """当前亏损链中最早那一笔（本轮首单）的预测方向 up/down；无链或非 recover 阶段为 None。"""
        n = self.consecutive_losses
        if n < 1 or n >= BLIND_RM_MAX_CONSECUTIVE_LOSSES:
            return None
        if n - 1 >= len(self.rows_considered):
            return None
        raw = self.rows_considered[n - 1].get("pred") or ""
        d = str(raw).lower()
        return d if d in ("up", "down") else None


def _effective_prediction_correct(row: dict[str, Any]) -> int | None:
    """用于连亏统计：优先读 ai_prediction_correct；为空时用结算结果 YES/NO 与方向推算（与 settlement._prediction_correct 一致）。"""
    raw = row.get("correct")
    if raw is not None:
        return int(raw)
    pred = str(row.get("pred") or "").lower()
    outcome = str(row.get("outcome") or "").upper()
    if pred in ("up", "down") and outcome in ("YES", "NO"):
        hit = (pred == "up" and outcome == "YES") or (pred == "down" and outcome == "NO")
        return 1 if hit else 0
    return None


def load_blind_rm_settlement_state(strategy_key: str, symbol: str) -> BlindRmSettlementState:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT e.ai_predicted_direction AS pred, e.ai_prediction_correct AS correct, e.result AS outcome
            FROM events e
            WHERE e.strategy_key = ? AND e.symbol = ? AND e.status = 'SETTLED'
            ORDER BY e.id DESC
            LIMIT 16
            """,
            (strategy_key, symbol.upper()),
        ).fetchall()
    finally:
        conn.close()

    streak = 0
    dict_rows: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        eff = _effective_prediction_correct(r)
        if eff is None:
            continue
        if eff == 1:
            break
        dict_rows.append(r)
        streak += 1

    return BlindRmSettlementState(consecutive_losses=streak, rows_considered=dict_rows)


def blind_rm_order_qty_usdt(settings_qty: float, state: BlindRmSettlementState) -> float:
    """下一笔下单名义：无连亏或已重置→基础 qty；连亏后按固定档 10→20→45；连亏满 4 次→基础 qty。

    连亏计数在遇到任意一笔结算为赢时会被打断（中间中了即重新从基础档开始）。

    若设置的基础数量等于 10，原先「连亏 1 次后下一档仍为 10」会与首单金额相同，看起来像没有倍投；
    因此在**首次 Recovery（连亏数 n==1）**时改为：使用阶梯里第一个**严格大于基础数量**的名义金额；
    例如 base=10 → 第一笔 Recovery 用 20；base=5 → 仍用 10。
    """
    base = float(settings_qty)
    n = state.consecutive_losses
    if n <= 0 or n >= BLIND_RM_MAX_CONSECUTIVE_LOSSES:
        return base
    ladder = BLIND_RM_MARTINGALE_AMOUNTS_USDT
    idx = n - 1
    if not (0 <= idx < len(ladder)):
        return base
    step = ladder[idx]
    if idx == 0:
        for amt in ladder:
            if amt > base + 1e-9:
                return float(amt)
        return float(max(step, base))
    return float(step)


def predict_blind_reverse_martingale_direction(
    symbol: str,
    duration: str = RULE_DURATION,
    *,
    entry_open_time: int | None = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(
            f"blind reverse martingale supports only {sorted(SUPPORTED_RULE_DURATIONS)}, got {duration}"
        )
    sym = symbol.upper()
    strategy = strategy_definition(BLIND_REVERSE_MARTINGALE_STRATEGY_KEY)
    state = load_blind_rm_settlement_state(strategy.key, sym)
    open_time = int(entry_open_time if entry_open_time is not None else time.time() * 1000)
    entry_price = _entry_price(fetch_premium_index(sym))
    entry_window_passed = is_within_entry_grace(
        open_time,
        now_ms,
        grace_ms=int(ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS),
    )

    n = state.consecutive_losses
    # n==0：上一笔赢或尚无记录 → 新轮随机首单
    # n>=4：首单+10+20+45 皆亏（含 45 未中）→ 新轮随机首单
    if n <= 0 or n >= BLIND_RM_MAX_CONSECUTIVE_LOSSES:
        direction = _stable_random_direction(sym, open_time)
        label = "BLIND_RM_RANDOM_NEW_CYCLE"
    else:
        anchor = state.cycle_anchor_direction()
        if not anchor:
            direction = _stable_random_direction(sym, open_time)
            label = "BLIND_RM_RANDOM_FALLBACK"
        else:
            direction = "down" if anchor == "up" else "up"
            label = "BLIND_RM_INVERT_CYCLE"

    confidence = 0.5
    probability_up = confidence if direction == "up" else 1.0 - confidence
    trade_quality_passed = entry_window_passed

    return {
        "symbol": sym,
        "strategy_key": strategy.key,
        "duration": duration,
        "open_time": int(open_time),
        "entry_price": round(float(entry_price), PRICE_DECIMALS),
        "direction": direction,
        "probability_up": round(probability_up, PROBABILITY_DECIMALS),
        "confidence": confidence,
        "certainty_label": label if trade_quality_passed else "BLIND_RM_WAIT_WINDOW",
        "threshold": None,
        "trade_quality_score": 1.0 if trade_quality_passed else 0.0,
        "trade_quality_passed": trade_quality_passed,
        "trade_quality_gate": BLIND_REVERSE_MARTINGALE_RULE_NAME,
        "high_winrate_gate": None,
        "high_winrate_rule": BLIND_REVERSE_MARTINGALE_RULE_NAME,
        "high_winrate_gate_passed": None,
        "high_winrate_gate_value": None,
        "high_winrate_gate_min": None,
        "signal_source": strategy.signal_source,
        "rule_score": float(n),
        "rule_reasons": [
            f"rule={BLIND_REVERSE_MARTINGALE_RULE_NAME}",
            f"consecutive_losses={n}",
            "cycle_policy=any_win_resets_chain;four_losses_or_45_miss_resets_chain",
            f"entry_window_passed={entry_window_passed}",
            f"direction_mode={label}",
        ],
        "orderbook": {
            "strategy": BLIND_REVERSE_MARTINGALE_RULE_NAME,
            "consecutiveLosses": n,
            "entryWindowPassed": entry_window_passed,
        },
        "timeframe_votes": [],
    }


def _stable_random_direction(symbol: str, open_time: int) -> str:
    raw = hashlib.sha256(f"{symbol.upper()}:{int(open_time)}".encode()).hexdigest()
    return "up" if int(raw[:8], 16) % 2 == 0 else "down"


def _entry_price(row: dict[str, Any]) -> float:
    price = float(row.get("indexPrice") or 0)
    if price <= 0:
        raise ValueError("latest index price unavailable")
    return price
