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
from app.services.runtime_symbols import configured_runtime_symbols, parse_symbol_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report feature data coverage against kline open_time.")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--interval", default="10m")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reports = [
        build_data_coverage_report(CoverageOptions(symbol=symbol, interval=args.interval))
        for symbol in _selected_symbols(args)
    ]
    report = reports[0] if args.symbol and len(reports) == 1 else {"symbols": reports}
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(payload)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False))


def _selected_symbols(args: argparse.Namespace) -> tuple[str, ...]:
    if args.symbols and args.symbol:
        raise ValueError("use either --symbol or --symbols, not both")
    if args.symbols:
        return parse_symbol_csv(args.symbols)
    if args.symbol:
        return parse_symbol_csv(args.symbol)
    return configured_runtime_symbols()


if __name__ == "__main__":
    main()
