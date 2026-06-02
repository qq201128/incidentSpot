from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.model_family_config import MODEL_FAMILIES  # noqa: E402
from app.services.model_search_job_store import (  # noqa: E402
    enqueue_model_search_jobs,
    update_pending_model_search_job_resources,
)
from app.services.runtime_symbols import parse_symbol_csv  # noqa: E402

DEFAULT_DURATIONS = ("10m", "30m", "60m", "1d")


def main() -> int:
    args = _parse_args()
    resource = _resource_payload(args)
    if args.update_pending_resources:
        payload = update_pending_model_search_job_resources(
            symbols=parse_symbol_csv(args.symbols),
            durations=tuple(args.durations),
            families=tuple(args.families),
            profile=args.profile,
            priority=args.priority,
            resource=resource,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    payload = enqueue_model_search_jobs(
        symbols=parse_symbol_csv(args.symbols),
        durations=tuple(args.durations),
        families=tuple(args.families),
        profile=args.profile,
        priority=args.priority,
        reset_existing=args.reset,
        resource=resource,
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
    parser.add_argument("--internal-threads", type=int, default=1)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--xgboost-process-workers", type=int, default=1)
    parser.add_argument("--resource-profile", default="local_safe")
    parser.add_argument("--reset", action="store_true", help="Reset existing matching jobs to pending.")
    parser.add_argument("--update-pending-resources", action="store_true")
    return parser.parse_args()


def _resource_payload(args: argparse.Namespace) -> dict:
    return {
        "resourceProfile": args.resource_profile,
        "internalThreads": args.internal_threads,
        "parallelWorkers": args.parallel_workers,
        "xgboostProcessWorkers": args.xgboost_process_workers,
    }


if __name__ == "__main__":
    raise SystemExit(main())
