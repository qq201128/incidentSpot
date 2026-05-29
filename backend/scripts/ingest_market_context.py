#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db.session import init_db
from app.services.market_context_ingest_service import ingest_market_context_data
from app.services.runtime_symbols import configured_runtime_symbols, parse_symbol_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest real non-kline factor data.")
    parser.add_argument("symbol", nargs="?")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--period", default="5m")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    init_db()
    reports = [
        ingest_market_context_data(symbol, period=args.period, limit=args.limit)
        for symbol in _selected_symbols(args)
    ]
    print(json.dumps({"symbols": reports}, ensure_ascii=False, indent=2))


def _selected_symbols(args: argparse.Namespace) -> tuple[str, ...]:
    if args.symbols and args.symbol:
        raise ValueError("use either positional symbol or --symbols, not both")
    if args.symbols:
        return parse_symbol_csv(args.symbols)
    if args.symbol:
        return parse_symbol_csv(args.symbol)
    return configured_runtime_symbols()


if __name__ == "__main__":
    main()
