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
from app.services.model_search_resource import (  # noqa: E402
    ModelSearchResourceConfig,
    resource_payload,
    validated_resource_config,
)
from app.services.model_search_status_service import MODEL_SEARCH_WORKER_COMMAND  # noqa: E402
from app.services.model_search_untrained_enqueue import enqueue_untrained_model_search_jobs  # noqa: E402
from app.services.runtime_symbols import configured_runtime_symbols, parse_symbol_csv  # noqa: E402


def main() -> int:
    args = _parse_args()
    _reject_inline_search_grid_args(args)
    payload = enqueue_untrained_model_search_jobs(
        symbols=_symbols(args.symbols),
        durations=_csv_strings(args.durations),
        families=("lstm",),
        profile=normalize_experiment_profile(args.profile),
        reset_existing=args.reset_existing,
        reset_history=args.reset_history,
        resource=_resource(args),
    )
    print(json.dumps(_response(payload), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enqueue LSTM candidate-search jobs without training inline.")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--durations", default="10m,30m,60m")
    parser.add_argument("--profile", default="full", choices=("fast", "full"))
    parser.add_argument("--feature-windows", default=None)
    parser.add_argument("--min-move-bps", default=None)
    parser.add_argument("--epochs", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--candidates-per-duration", type=int, default=None)
    parser.add_argument("--internal-threads", type=int, default=1)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--xgboost-process-workers", type=int, default=1)
    parser.add_argument("--resource-profile", default="local_safe")
    parser.add_argument("--reset-existing", action="store_true")
    parser.add_argument("--reset-history", action="store_true")
    return parser.parse_args()


def _reject_inline_search_grid_args(args: argparse.Namespace) -> None:
    rejected = (
        args.feature_windows,
        args.min_move_bps,
        args.epochs,
        args.seeds,
        args.candidates_per_duration,
    )
    if any(value is not None for value in rejected):
        raise ValueError("custom LSTM inline search-grid arguments are disabled; enqueue model-search jobs instead")


def _csv_strings(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("CSV argument must include at least one value")
    return values


def _symbols(raw: str | None) -> tuple[str, ...]:
    return parse_symbol_csv(raw) if raw is not None else configured_runtime_symbols()


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
        "executionMode": "queued_lstm_model_search_jobs",
        "trainingRunsInBackendProcess": False,
        "workerCommand": MODEL_SEARCH_WORKER_COMMAND,
        "message": "LSTM候选搜索已入队；训练由独立 worker 单次认领执行。",
    }


if __name__ == "__main__":
    raise SystemExit(main())
