from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.model_family_config import MODEL_FAMILIES  # noqa: E402
from app.services.model_search_job_store import enqueue_model_search_jobs  # noqa: E402
from app.services.runtime_symbols import parse_symbol_csv  # noqa: E402

DEFAULT_DURATIONS = ("10m", "30m", "60m", "1d")


def main() -> int:
    args = _parse_args()
    payload = enqueue_model_search_jobs(
        symbols=parse_symbol_csv(args.symbols),
        durations=tuple(args.durations),
        families=tuple(args.families),
        profile=args.profile,
        priority=args.priority,
        reset_existing=args.reset,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enqueue model-family search jobs without training immediately.")
    parser.add_argument("--symbols", required=True, help="CSV symbols, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--durations", nargs="+", default=list(DEFAULT_DURATIONS))
    parser.add_argument("--families", nargs="+", default=list(MODEL_FAMILIES))
    parser.add_argument("--profile", choices=("fast", "full"), default="full")
    parser.add_argument("--priority", type=int, default=100)
    parser.add_argument("--reset", action="store_true", help="Reset existing matching jobs to pending.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
