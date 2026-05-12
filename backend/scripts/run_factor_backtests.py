#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.factor_backtest_batch_service import (
    BACKTEST_DURATION_ORDER,
    run_all_factor_backtests,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all registered factor backtests.")
    parser.add_argument("symbol", nargs="?", default="BTCUSDT", help="e.g. BTCUSDT")
    parser.add_argument(
        "--durations",
        default=",".join(BACKTEST_DURATION_ORDER),
        help="comma-separated durations, default: 10m,30m,60m,1d",
    )
    parser.add_argument("--output", default=None, help="optional JSON output path")
    return parser.parse_args()


def _duration_tuple(raw: str) -> tuple[str, ...]:
    durations = tuple(part.strip() for part in raw.split(",") if part.strip())
    return durations or BACKTEST_DURATION_ORDER


def _write_report(report: dict[str, Any], output: str | None) -> None:
    if not output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    args = _parse_args()
    report = run_all_factor_backtests(args.symbol, durations=_duration_tuple(args.durations))
    _write_report(report, args.output)


if __name__ == "__main__":
    main()
