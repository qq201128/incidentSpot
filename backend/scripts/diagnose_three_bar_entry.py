#!/usr/bin/env python3
"""
复核「三连 10m 反向（不倍投）」在某一 UTC 10m 桶的判定（仅形态入场、无 Recovery；与自动预测/自动下单逻辑一致）。

在 backend 目录下执行（需已安装依赖、且存在与运行中服务相同的 SQLite 库）:

  python scripts/diagnose_three_bar_entry.py BTCUSDT --bucket-utc "2026-05-11T01:10:00+00:00"

若不指定 --bucket-utc，则使用当前时刻对齐后的 10m 桶起点。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from app.services.blind_reverse_martingale_strategy import load_blind_rm_settlement_state
from app.services.kline_timing import RULE_INTERVAL_MS, current_rule_entry_open_time
from app.services.strategy_registry import THREE_BAR_10M_RM_STRATEGY_KEY
from app.services.three_bar_10m_reverse_martingale_strategy import (
    last_settled_event_boundary_ms,
    predict_three_bar_10m_reverse_martingale_direction,
    streak_and_last_n_10m_rest,
)


def _parse_bucket_ms(raw: str | None) -> int:
    if not raw or not str(raw).strip():
        return current_rule_entry_open_time()
    s = str(raw).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ms = int(dt.timestamp() * 1000)
    step = int(RULE_INTERVAL_MS)
    return (ms // step) * step


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose three-bar 10m RM at one UTC bucket.")
    parser.add_argument("symbol", help="e.g. BTCUSDT")
    parser.add_argument(
        "--bucket-utc",
        dest="bucket_utc",
        default=None,
        help='UTC bucket open, ISO8601 e.g. "2026-05-11T01:10:00+00:00"',
    )
    args = parser.parse_args()

    sym = args.symbol.upper()
    bucket_ms = _parse_bucket_ms(args.bucket_utc)

    state = load_blind_rm_settlement_state(THREE_BAR_10M_RM_STRATEGY_KEY, sym)
    pred = predict_three_bar_10m_reverse_martingale_direction(
        sym, entry_open_time=bucket_ms, now_ms=bucket_ms + 60_000
    )
    streak_raw, last_bars = streak_and_last_n_10m_rest(sym, bucket_ms, 3)

    print(f"symbol={sym} bucket_open_time_ms={bucket_ms}")
    print(f"consecutive_losses(n_loss)={state.consecutive_losses} (仅展示，不影响是否入场)")
    lb = last_settled_event_boundary_ms(THREE_BAR_10M_RM_STRATEGY_KEY, sym)
    print(f"last_settled_event_boundary_ms={lb!r} (形态 K 须全部在该 10m 桶之后)")
    print(f"streak_raw={streak_raw!r} pattern_ok implicit from pred rule_reasons")
    print(f"trade_quality_passed={pred.get('trade_quality_passed')}")
    print(f"certainty_label={pred.get('certainty_label')}")
    print(f"direction={pred.get('direction')}")
    for line in pred.get("rule_reasons") or []:
        print(f"  {line}")
    if last_bars:
        print("last 3 completed index 10m bars before bucket (OHLC close vs open => bull/bear):")
        for b in last_bars:
            o = float(b["open"])
            c = float(b["close"])
            tag = "bull" if c > o else ("bear" if c < o else "doji")
            print(
                f"  openTime={b.get('openTime')} O={o} C={c} => {tag}",
            )
    else:
        print("last_bars=None (not enough completed bars)")


if __name__ == "__main__":
    main()
