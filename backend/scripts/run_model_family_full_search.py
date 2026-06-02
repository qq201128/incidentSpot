from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.experiment_profiles import normalize_experiment_profile  # noqa: E402
from app.services.model_family_config import MODEL_FAMILIES  # noqa: E402
from app.services.model_search_resource import (  # noqa: E402
    ModelSearchResourceConfig,
    resource_payload,
    validated_resource_config,
)
from app.services.model_search_status_service import MODEL_SEARCH_WORKER_COMMAND  # noqa: E402
from app.services.model_search_untrained_enqueue import enqueue_untrained_model_search_jobs  # noqa: E402
from app.services.runtime_symbols import configured_runtime_symbols, parse_symbol_csv  # noqa: E402

DEFAULT_DURATIONS = ("10m", "60m")


def main() -> int:
    args = _parse_args()
    payload = enqueue_untrained_model_search_jobs(
        symbols=_selected_symbols(args),
        durations=tuple(args.durations),
        families=tuple(args.families),
        profile=normalize_experiment_profile(args.profile),
        reset_existing=args.reset_existing,
        reset_history=args.reset_history,
        resource=_resource(args),
    )
    print(json.dumps(_response(payload), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a model-family full search into queued single-target jobs."
    )
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--durations", nargs="+", default=list(DEFAULT_DURATIONS))
    parser.add_argument("--families", nargs="+", default=list(MODEL_FAMILIES))
    parser.add_argument("--profile", choices=("fast", "full"), default="full")
    parser.add_argument("--internal-threads", type=int, default=1)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--xgboost-process-workers", type=int, default=1)
    parser.add_argument("--resource-profile", default="local_safe")
    parser.add_argument("--reset-existing", action="store_true")
    parser.add_argument("--reset-history", action="store_true")
    return parser.parse_args()


def _selected_symbols(args: argparse.Namespace) -> tuple[str, ...]:
    if args.symbols and args.symbol:
        raise ValueError("use either --symbol or --symbols, not both")
    if args.symbols:
        return parse_symbol_csv(args.symbols)
    if args.symbol:
        return parse_symbol_csv(args.symbol)
    return configured_runtime_symbols()


def _resource(args: argparse.Namespace) -> dict[str, Any]:
    config = ModelSearchResourceConfig(
        resource_profile=args.resource_profile,
        internal_threads=args.internal_threads,
        parallel_workers=args.parallel_workers,
        xgboost_process_workers=args.xgboost_process_workers,
    )
    return resource_payload(validated_resource_config(config))


def _response(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "executionMode": "queued_single_target_jobs",
        "trainingRunsInBackendProcess": False,
        "workerCommand": MODEL_SEARCH_WORKER_COMMAND,
        "message": "已拆分为队列任务；训练由独立 worker 单次认领执行。",
    }


if __name__ == "__main__":
    raise SystemExit(main())
