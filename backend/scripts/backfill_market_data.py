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
from app.services.data_coverage_report import CoverageOptions, build_data_coverage_report
from app.services.factor_page_alerts import coverage_gaps
from app.services.market_data_backfill_service import backfill_symbol_market_data
from app.services.rule_config import SUPPORTED_RULE_DURATIONS
from app.services.runtime_symbols import configured_runtime_symbols, parse_symbol_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill klines and factor dependency tables.")
    parser.add_argument("symbol", nargs="?")
    parser.add_argument("--symbols", default=None)
    parser.add_argument(
        "--durations",
        default=",".join(sorted(SUPPORTED_RULE_DURATIONS)),
        help="comma-separated rule durations (default: all)",
    )
    parser.add_argument("--target-1m-rows", type=int, default=None)
    parser.add_argument("--market-context-limit", type=int, default=None)
    parser.add_argument("--skip-multi", action="store_true")
    parser.add_argument("--skip-feature-fill", action="store_true")
    args = parser.parse_args()

    durations = tuple(part.strip() for part in args.durations.split(",") if part.strip())
    init_db()

    reports = [
        _backfill_symbol(
            symbol,
            durations=durations,
            target_1m_rows=args.target_1m_rows,
            market_context_limit=args.market_context_limit,
            sync_multi=not args.skip_multi,
            fill_bar_features=not args.skip_feature_fill,
        )
        for symbol in _selected_symbols(args)
    ]
    print(json.dumps({"symbols": reports}, ensure_ascii=False, indent=2))


def _backfill_symbol(symbol: str, **kwargs) -> dict:
    report = backfill_symbol_market_data(symbol, **kwargs)
    coverage = [_coverage_summary(symbol, duration) for duration in kwargs["durations"]]
    return {"backfill": report, "coverage": coverage}


def _coverage_summary(symbol: str, duration: str) -> dict:
    coverage = build_data_coverage_report(CoverageOptions(symbol.upper(), duration))
    gaps = coverage_gaps(coverage, primary_interval=duration)
    return {
        "symbol": symbol.upper(),
        "duration": duration,
        "mainKlineRows": coverage["mainRange"]["rowCount"],
        "remainingGaps": len(gaps),
        "gaps": gaps[:10],
    }


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
