#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.data_coverage_report import CoverageOptions, build_data_coverage_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report feature data coverage against kline open_time.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="10m")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_data_coverage_report(CoverageOptions(symbol=args.symbol, interval=args.interval))
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(payload)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
