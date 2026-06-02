from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.model_search_adaptive_parallelism import (  # noqa: E402
    AUTO_CPU_PER_JOB,
    AUTO_MAX_RUNNING_JOBS,
    AUTO_MEMORY_PER_JOB_MB,
    DEFAULT_CPU_HEADROOM_CORES,
    DEFAULT_CPU_HIGH_PERCENT,
    DEFAULT_CPU_LOW_PERCENT,
    DEFAULT_CPU_SAMPLE_SECONDS,
    DEFAULT_MIN_AVAILABLE_MEMORY_MB,
    DEFAULT_MIN_RUNNING_JOBS,
    AdaptiveParallelismConfig,
)
from app.services.model_search_job_runner import DEFAULT_LOG_DIR, ModelSearchWorkerConfig, run_one_model_search_job  # noqa: E402
from app.services.model_search_resource import ModelSearchResourceConfig  # noqa: E402
from app.services.model_search_worker_pool import run_adaptive_worker_pool  # noqa: E402

DEFAULT_POLL_SECONDS = 5.0
RUN_ONCE_MAX_RUNNING_JOBS = 1


def main() -> int:
    args = _parse_args()
    config = _worker_config(args)
    if args.loop:
        return _run_forever(config, args, compact=args.compact)
    if args.run_until_empty:
        return _run_until_empty(config, args, compact=args.compact)
    return _run_once(config, compact=args.compact)


def _run_once(config: ModelSearchWorkerConfig, *, compact: bool = False) -> int:
    _print_report(run_one_model_search_job(config), compact=compact)
    return 0


def _run_until_empty(
    config: ModelSearchWorkerConfig,
    args: argparse.Namespace,
    *,
    compact: bool = False,
) -> int:
    if args.adaptive_parallelism:
        return _run_adaptive(config, args, run_until_empty=True, compact=compact)
    while True:
        report = run_one_model_search_job(config)
        _print_report(report, compact=compact)
        if _is_idle(report):
            return 0


def _run_forever(
    config: ModelSearchWorkerConfig,
    args: argparse.Namespace,
    *,
    compact: bool = False,
) -> int:
    if args.adaptive_parallelism:
        return _run_adaptive(config, args, run_until_empty=False, compact=compact)
    while True:
        report = run_one_model_search_job(config)
        _print_report(report, compact=compact)
        if _is_idle(report):
            time.sleep(args.poll_seconds)


def _run_adaptive(
    config: ModelSearchWorkerConfig,
    args: argparse.Namespace,
    *,
    run_until_empty: bool,
    compact: bool,
) -> int:
    reports = run_adaptive_worker_pool(
        config,
        _adaptive_config(args),
        poll_seconds=args.poll_seconds,
        run_until_empty=run_until_empty,
    )
    for report in reports:
        _print_report(report, compact=compact)
    return 0


def _is_idle(report: dict) -> bool:
    return report.get("status") == "idle" and report.get("reason") == "no_pending_job"


def _print_report(report: dict, *, compact: bool) -> None:
    payload = _compact_report(report) if compact else report
    _write_json_report(payload, sys.stdout)


def _write_json_report(payload: dict[str, Any], stream: TextIO) -> None:
    json.dump(payload, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
    stream.flush()


def _compact_report(report: dict) -> dict:
    job = report.get("job") or {}
    result = report.get("result") or {}
    batch = result.get("jobBatch") or {}
    adaptive = report.get("adaptiveParallelism") or {}
    return {
        "status": report.get("status"),
        "jobId": job.get("job_id"),
        "symbol": job.get("symbol"),
        "duration": job.get("duration"),
        "modelFamily": job.get("model_family"),
        "jobStatus": job.get("status"),
        "resultStatus": result.get("status"),
        "reason": report.get("reason") or result.get("reason") or result.get("validationFailureReason"),
        "jobBatch": batch or None,
        "adaptiveParallelism": adaptive or None,
        "logPath": job.get("log_path"),
    }


def _worker_config(args: argparse.Namespace) -> ModelSearchWorkerConfig:
    return ModelSearchWorkerConfig(
        max_running_jobs=_worker_max_running_jobs(args),
        resource=ModelSearchResourceConfig(
            resource_profile=args.resource_profile,
            internal_threads=args.internal_threads,
            parallel_workers=args.parallel_workers,
            xgboost_process_workers=args.xgboost_process_workers,
            torch_jobs=args.torch_jobs,
        ),
        log_dir=Path(args.log_dir),
        stale_after_seconds=args.stale_after_seconds,
        candidates_per_job=args.candidates_per_job,
    )


def _worker_max_running_jobs(args: argparse.Namespace) -> int:
    if not args.adaptive_parallelism:
        return args.max_running_jobs or RUN_ONCE_MAX_RUNNING_JOBS
    if args.run_until_empty or args.loop:
        return RUN_ONCE_MAX_RUNNING_JOBS
    return args.max_running_jobs or RUN_ONCE_MAX_RUNNING_JOBS


def _adaptive_config(args: argparse.Namespace) -> AdaptiveParallelismConfig:
    return AdaptiveParallelismConfig(
        min_jobs=args.adaptive_min_jobs,
        max_jobs=args.max_running_jobs,
        cpu_per_job=args.cpu_per_job,
        memory_per_job_mb=args.memory_per_job_mb,
        cpu_headroom_cores=args.cpu_headroom_cores,
        cpu_low_percent=args.cpu_low_percent,
        cpu_high_percent=args.cpu_high_percent,
        min_available_memory_mb=args.min_available_memory_mb,
        cpu_sample_seconds=args.cpu_sample_seconds,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Claim and run pending model search jobs.")
    parser.add_argument("--max-running-jobs", type=_non_negative_int, default=AUTO_MAX_RUNNING_JOBS)
    parser.add_argument("--internal-threads", type=int, default=1)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--xgboost-process-workers", type=int, default=1)
    parser.add_argument("--torch-jobs", type=int, default=1)
    parser.add_argument("--resource-profile", default="local_safe")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--stale-after-seconds", type=int, default=3600)
    parser.add_argument("--candidates-per-job", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-until-empty", action="store_true", help="Keep claiming jobs until the queue is empty.")
    mode.add_argument("--loop", action="store_true", help="Keep running in foreground and poll when no job is pending.")
    parser.add_argument("--poll-seconds", type=_positive_float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--adaptive-parallelism", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-min-jobs", type=_positive_int, default=DEFAULT_MIN_RUNNING_JOBS)
    parser.add_argument("--cpu-per-job", type=_non_negative_int, default=AUTO_CPU_PER_JOB)
    parser.add_argument("--memory-per-job-mb", type=_non_negative_int, default=AUTO_MEMORY_PER_JOB_MB)
    parser.add_argument("--cpu-headroom-cores", type=_non_negative_int, default=DEFAULT_CPU_HEADROOM_CORES)
    parser.add_argument("--cpu-low-percent", type=_percent, default=DEFAULT_CPU_LOW_PERCENT)
    parser.add_argument("--cpu-high-percent", type=_percent, default=DEFAULT_CPU_HIGH_PERCENT)
    parser.add_argument("--cpu-sample-seconds", type=_positive_float, default=DEFAULT_CPU_SAMPLE_SECONDS)
    parser.add_argument("--min-available-memory-mb", type=_non_negative_int, default=DEFAULT_MIN_AVAILABLE_MEMORY_MB)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if args.cpu_low_percent >= args.cpu_high_percent:
        parser.error("--cpu-low-percent must be lower than --cpu-high-percent")
    return args


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


def _percent(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or parsed > 100:
        raise argparse.ArgumentTypeError("value must be within 0..100")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
