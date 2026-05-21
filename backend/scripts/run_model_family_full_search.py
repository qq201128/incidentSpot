from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from app.services.model_family_candidate_search_service import (  # noqa: E402
    ModelCandidateSearchConfig,
    XGBOOST_PROCESS_WORKERS_ENV,
    run_model_candidate_search,
)
from app.services.model_family_config import MODEL_FAMILIES  # noqa: E402
from app.services.model_family_search_rules import model_family_training_rules  # noqa: E402
from app.services.model_family_status_service import model_family_status  # noqa: E402


DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_DURATIONS = ("10m", "60m")
DEFAULT_LOG_DIR = Path("runtime") / "model-family-full-search"
DEFAULT_TARGET_WORKERS = 2
JSONL_LOCK = Lock()


@dataclass(frozen=True)
class SearchTarget:
    family: str
    duration: str


def main() -> int:
    args = _parse_args()
    _configure_worker_overrides(args)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    jsonl_path = log_dir / f"{run_id}.jsonl"
    summary_path = log_dir / f"{run_id}_summary.json"
    targets = _targets_from_args(args)
    results = _run_targets(targets, args, jsonl_path)
    summary = _summary_payload(args, run_id, results)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"runId": run_id, "summaryPath": str(summary_path), "results": results}, ensure_ascii=False))
    return 1 if any(item["executionStatus"] == "error" for item in results) else 0


def _run_targets(targets: list[SearchTarget], args, jsonl_path: Path) -> list[dict[str, Any]]:
    workers = min(max(int(args.target_workers), 1), len(targets))
    if workers <= 1:
        return [_run_target(target, args, jsonl_path) for target in targets]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_target, target, args, jsonl_path) for target in targets]
        for future in as_completed(futures):
            results.append(future.result())
    return _ordered_results(targets, results)


def _run_target(target: SearchTarget, args, jsonl_path: Path) -> dict[str, Any]:
    started = _utc_now()
    _append_jsonl(jsonl_path, {"event": "start", "family": target.family, "duration": target.duration, "at": started})
    try:
        result = run_model_candidate_search(
            ModelCandidateSearchConfig(target.family, args.symbol, target.duration, args.profile, args.parallel_workers)
        )
        payload = _result_payload(target, args.symbol, started, result, "completed")
    except Exception as exc:
        payload = _error_payload(target, args.symbol, started, exc)
    _append_jsonl(jsonl_path, {"event": "finish", **payload})
    return payload


def _result_payload(target: SearchTarget, symbol: str, started: str, result: dict[str, Any], execution_status: str) -> dict[str, Any]:
    status = model_family_status(target.family, symbol, target.duration)
    progress = status.get("candidateSearchProgress") or {}
    reports = result.get("reports") or []
    return {
        "executionStatus": execution_status,
        "family": target.family,
        "duration": target.duration,
        "startedAt": started,
        "finishedAt": _utc_now(),
        "searchStatus": result.get("status"),
        "reportsReturned": len(reports),
        "progress": _progress_payload(progress),
        "modelStatus": status.get("status"),
        "shadowPredictionReady": status.get("shadowPredictionReady"),
        "blockedReason": status.get("shadowPredictionBlockedReason"),
        "rules": model_family_training_rules(target.family),
    }


def _error_payload(target: SearchTarget, symbol: str, started: str, exc: Exception) -> dict[str, Any]:
    payload = _result_payload(target, symbol, started, {"status": "error", "reports": []}, "error")
    return {**payload, "errorType": type(exc).__name__, "error": str(exc)}


def _progress_payload(progress: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": progress.get("status"),
        "completed": progress.get("completed"),
        "total": progress.get("total"),
        "searchSpaceTotal": progress.get("searchSpaceTotal"),
        "counts": progress.get("counts"),
    }


def _summary_payload(args, run_id: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runId": run_id,
        "symbol": args.symbol,
        "profile": args.profile,
        "durations": args.durations,
        "families": args.families,
        "parallelWorkers": args.parallel_workers,
        "targetWorkers": args.target_workers,
        "xgboostProcessWorkers": args.xgboost_process_workers,
        "targetWinRate": ">70%",
        "generatedAt": _utc_now(),
        "results": results,
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--durations", nargs="+", default=list(DEFAULT_DURATIONS))
    parser.add_argument("--families", nargs="+", default=list(MODEL_FAMILIES))
    parser.add_argument("--profile", default="full")
    parser.add_argument("--parallel-workers", type=int, default=10)
    parser.add_argument("--target-workers", type=int, default=DEFAULT_TARGET_WORKERS)
    parser.add_argument("--xgboost-process-workers", type=int)
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    return parser.parse_args()


def _configure_worker_overrides(args) -> None:
    if args.xgboost_process_workers is None:
        return
    if args.xgboost_process_workers <= 0:
        raise ValueError("--xgboost-process-workers must be positive")
    os.environ[XGBOOST_PROCESS_WORKERS_ENV] = str(args.xgboost_process_workers)


def _ordered_results(targets: list[SearchTarget], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {(target.family, target.duration): idx for idx, target in enumerate(targets)}
    return sorted(results, key=lambda item: order[(str(item["family"]), str(item["duration"]))])


def _targets_from_args(args) -> list[SearchTarget]:
    seen = set()
    targets = []
    for duration in args.durations:
        for family in args.families:
            target = SearchTarget(family, duration)
            key = (target.family, target.duration)
            if key not in seen:
                seen.add(key)
                targets.append(target)
    return targets


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with JSONL_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
