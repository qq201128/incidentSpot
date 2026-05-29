from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.model_search_job_runner import ModelSearchWorkerConfig, run_one_model_search_job  # noqa: E402
from app.services.model_search_resource import ModelSearchResourceConfig  # noqa: E402

DEFAULT_POLL_SECONDS = 5.0


def main() -> int:
    args = _parse_args()
    config = _worker_config(args)
    if args.loop:
        return _run_forever(config, args.poll_seconds)
    if args.run_until_empty:
        return _run_until_empty(config)
    return _run_once(config)


def _run_once(config: ModelSearchWorkerConfig) -> int:
    _print_report(run_one_model_search_job(config))
    return 0


def _run_until_empty(config: ModelSearchWorkerConfig) -> int:
    while True:
        report = run_one_model_search_job(config)
        _print_report(report)
        if _is_idle(report):
            return 0


def _run_forever(config: ModelSearchWorkerConfig, poll_seconds: float) -> int:
    while True:
        report = run_one_model_search_job(config)
        _print_report(report)
        if _is_idle(report):
            time.sleep(poll_seconds)


def _is_idle(report: dict) -> bool:
    return report.get("status") == "idle" and report.get("reason") == "no_pending_job"


def _print_report(report: dict) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def _worker_config(args: argparse.Namespace) -> ModelSearchWorkerConfig:
    return ModelSearchWorkerConfig(
        max_running_jobs=args.max_running_jobs,
        resource=ModelSearchResourceConfig(
            resource_profile=args.resource_profile,
            internal_threads=args.internal_threads,
            parallel_workers=args.parallel_workers,
            xgboost_process_workers=args.xgboost_process_workers,
            torch_jobs=args.torch_jobs,
        ),
        log_dir=Path(args.log_dir),
        stale_after_seconds=args.stale_after_seconds,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Claim and run pending model search jobs.")
    parser.add_argument("--max-running-jobs", type=int, default=1)
    parser.add_argument("--internal-threads", type=int, default=4)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--xgboost-process-workers", type=int, default=1)
    parser.add_argument("--torch-jobs", type=int, default=1)
    parser.add_argument("--resource-profile", default="local_safe")
    parser.add_argument("--log-dir", default=str(Path("runtime") / "model-search-jobs"))
    parser.add_argument("--stale-after-seconds", type=int, default=3600)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-until-empty", action="store_true", help="Keep claiming jobs until the queue is empty.")
    mode.add_argument("--loop", action="store_true", help="Keep running in foreground and poll when no job is pending.")
    parser.add_argument("--poll-seconds", type=_positive_float, default=DEFAULT_POLL_SECONDS)
    return parser.parse_args()


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
