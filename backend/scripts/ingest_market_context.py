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


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest real non-kline factor data.")
    parser.add_argument("symbol", nargs="?", default="BTCUSDT")
    parser.add_argument("--period", default="5m")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    init_db()
    report = ingest_market_context_data(args.symbol, period=args.period, limit=args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
