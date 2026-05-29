from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from app.services.model_family_candidate_search_service import (  # noqa: E402
    ModelCandidateSearchConfig,
    run_model_candidate_search,
)
from app.services.model_family_candidate_executor import XGBOOST_PROCESS_WORKERS_ENV  # noqa: E402
from app.services.model_family_candidates import read_model_candidate_progress  # noqa: E402
from app.services.model_family_config import MODEL_FAMILIES  # noqa: E402
from app.services.model_family_search_rules import model_family_training_rules  # noqa: E402
from app.services.model_family_status_service import model_family_status  # noqa: E402
from app.services.runtime_symbols import configured_runtime_symbols, parse_symbol_csv  # noqa: E402


DEFAULT_DURATIONS = ("10m", "60m")
DEFAULT_LOG_DIR = Path("runtime") / "model-family-full-search"
DEFAULT_TARGET_WORKERS = 2
DEFAULT_PROGRESS_INTERVAL_SECONDS = 15
JSONL_LOCK = Lock()
PROGRESS_LOCK = Lock()


@dataclass(frozen=True)
class SearchTarget:
    family: str
    duration: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.family, self.duration)


class RunProgressMonitor:
    def __init__(
        self,
        *,
        symbol: str,
        targets: list[SearchTarget],
        run_id: str,
        jsonl_path: Path,
        interval_seconds: int,
    ) -> None:
        self.symbol = symbol
        self.targets = targets
        self.run_id = run_id
        self.jsonl_path = jsonl_path
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running: set[tuple[str, str]] = set()
        self._finished: dict[tuple[str, str], dict[str, Any]] = {}

    def start(self) -> None:
        if self.interval_seconds <= 0:
            return
        self._thread = threading.Thread(target=self._loop, name="model-family-progress", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self.interval_seconds > 0:
            self._print_snapshot(final=True)

    def mark_start(self, target: SearchTarget) -> None:
        with PROGRESS_LOCK:
            self._running.add(target.key)
        _log_progress(f"START {target.family}/{target.duration}")

    def mark_finish(self, target: SearchTarget, payload: dict[str, Any]) -> None:
        with PROGRESS_LOCK:
            self._running.discard(target.key)
            self._finished[target.key] = payload
        status = payload.get("searchStatus") or payload.get("executionStatus") or "done"
        _log_progress(f"DONE  {target.family}/{target.duration} -> {status}")

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._print_snapshot()

    def _print_snapshot(self, *, final: bool = False) -> None:
        with PROGRESS_LOCK:
            running = set(self._running)
            finished = dict(self._finished)
        rows = _progress_rows(self.symbol, self.targets, running, finished)
        done_count = len(self._finished)
        total_count = len(self.targets)
        title = "FINAL" if final else "PROGRESS"
        _log_progress("")
        _log_progress(
            f"[{title}] run={self.run_id} symbol={self.symbol} "
            f"targets={done_count}/{total_count} log={self.jsonl_path}"
        )
        _log_progress(_progress_table(rows))
        if not final:
            _log_progress("")


def main() -> int:
    args = _parse_args()
    _configure_worker_overrides(args)
    symbols = _selected_symbols(args)
    if len(symbols) > 1:
        results = [_run_for_symbol(args, symbol) for symbol in symbols]
        print(json.dumps({"symbols": list(symbols), "results": results}, ensure_ascii=False))
        return 1 if any(item["exitCode"] for item in results) else 0
    return _run_for_symbol(args, symbols[0])["exitCode"]


def _run_for_symbol(args, symbol: str) -> dict[str, Any]:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    jsonl_path = log_dir / f"{run_id}_{symbol}.jsonl"
    summary_path = log_dir / f"{run_id}_{symbol}_summary.json"
    targets = _targets_from_args(args)
    monitor = RunProgressMonitor(
        symbol=symbol,
        targets=targets,
        run_id=run_id,
        jsonl_path=jsonl_path,
        interval_seconds=0 if args.no_progress else args.progress_interval,
    )
    _log_progress(
        f"Model family full search started: run={run_id} symbol={symbol} "
        f"targets={len(targets)} profile={args.profile} resetHistory={args.reset_history}"
    )
    _log_progress(f"Progress log: {jsonl_path}")
    if args.no_progress:
        _log_progress("Live progress disabled (--no-progress).")
    elif args.progress_interval > 0:
        _log_progress(f"Live progress every {args.progress_interval}s (also on each target start/finish).")
    monitor.start()
    try:
        results = _run_targets(targets, args, symbol, jsonl_path, monitor)
    finally:
        monitor.stop()
    summary = _summary_payload(args, symbol, run_id, results)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {"runId": run_id, "symbol": symbol, "summaryPath": str(summary_path), "results": results}
    if len(_selected_symbols(args)) == 1:
        print(json.dumps(payload, ensure_ascii=False))
    return {**payload, "exitCode": 1 if any(item["executionStatus"] == "error" for item in results) else 0}


def _run_targets(
    targets: list[SearchTarget],
    args,
    symbol: str,
    jsonl_path: Path,
    monitor: RunProgressMonitor,
) -> list[dict[str, Any]]:
    workers = min(max(int(args.target_workers), 1), len(targets))
    if workers <= 1:
        results = []
        for target in targets:
            results.append(_run_target(target, args, symbol, jsonl_path, monitor))
        return results
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_target, target, args, symbol, jsonl_path, monitor): target for target in targets
        }
        for future in as_completed(futures):
            results.append(future.result())
    return _ordered_results(targets, results)


def _run_target(
    target: SearchTarget,
    args,
    symbol: str,
    jsonl_path: Path,
    monitor: RunProgressMonitor,
) -> dict[str, Any]:
    started = _utc_now()
    monitor.mark_start(target)
    _append_jsonl(jsonl_path, {"event": "start", "family": target.family, "duration": target.duration, "at": started})
    try:
        result = run_model_candidate_search(
            ModelCandidateSearchConfig(
                target.family,
                symbol,
                target.duration,
                args.profile,
                args.parallel_workers,
                reset_history=args.reset_history,
            )
        )
        payload = _result_payload(target, symbol, started, result, "completed")
    except Exception as exc:
        payload = _error_payload(target, symbol, started, exc)
    _append_jsonl(jsonl_path, {"event": "finish", **payload})
    monitor.mark_finish(target, payload)
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


def _summary_payload(args, symbol: str, run_id: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runId": run_id,
        "symbol": symbol,
        "profile": args.profile,
        "durations": args.durations,
        "families": args.families,
        "parallelWorkers": args.parallel_workers,
        "targetWorkers": args.target_workers,
        "resetHistory": args.reset_history,
        "xgboostProcessWorkers": args.xgboost_process_workers,
        "targetWinRate": ">70%",
        "generatedAt": _utc_now(),
        "results": results,
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--durations", nargs="+", default=list(DEFAULT_DURATIONS))
    parser.add_argument("--families", nargs="+", default=list(MODEL_FAMILIES))
    parser.add_argument("--profile", default="full")
    parser.add_argument("--parallel-workers", type=int, default=10)
    parser.add_argument("--target-workers", type=int, default=DEFAULT_TARGET_WORKERS)
    parser.add_argument("--xgboost-process-workers", type=int)
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument(
        "--reset-history",
        action="store_true",
        help="Ignore prior candidate search history and retrain from scratch.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL_SECONDS,
        help="Print live progress every N seconds (default: 15).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable periodic live progress output.",
    )
    return parser.parse_args()


def _selected_symbols(args) -> tuple[str, ...]:
    if args.symbols and args.symbol:
        raise ValueError("use either --symbol or --symbols, not both")
    if args.symbols:
        return parse_symbol_csv(args.symbols)
    if args.symbol:
        return parse_symbol_csv(args.symbol)
    return configured_runtime_symbols()


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


def _log_progress(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def _progress_rows(
    symbol: str,
    targets: list[SearchTarget],
    running: set[tuple[str, str]],
    finished: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, str]]:
    rows = []
    for target in targets:
        key = target.key
        if key in finished:
            payload = finished[key]
            progress = payload.get("progress") or {}
            rows.append(
                {
                    "state": "DONE",
                    "target": f"{target.family}/{target.duration}",
                    "candidate": _candidate_progress_text(progress),
                    "result": str(payload.get("searchStatus") or payload.get("executionStatus") or "-"),
                }
            )
            continue
        progress = read_model_candidate_progress(target.family, symbol, target.duration)
        state = "RUN" if key in running else "WAIT"
        rows.append(
            {
                "state": state,
                "target": f"{target.family}/{target.duration}",
                "candidate": _candidate_progress_text(progress),
                "result": str(progress.get("status") or "-"),
            }
        )
    return rows


def _candidate_progress_text(progress: dict[str, Any]) -> str:
    completed = progress.get("completed")
    total = progress.get("total")
    if completed is None or total is None:
        return "-"
    percent = progress.get("percent")
    if percent is None and int(total) > 0:
        percent = round(float(completed) / float(total) * 100.0, 1)
    percent_text = "" if percent is None else f" {percent}%"
    return f"{completed}/{total}{percent_text}"


def _progress_table(rows: list[dict[str, str]]) -> str:
    headers = ("STATE", "TARGET", "CANDIDATES", "STATUS")
    widths = [
        max(len(headers[0]), *(len(row["state"]) for row in rows), 4),
        max(len(headers[1]), *(len(row["target"]) for row in rows), 6),
        max(len(headers[2]), *(len(row["candidate"]) for row in rows), 9),
        max(len(headers[3]), *(len(row["result"]) for row in rows), 6),
    ]

    def _line(cells: tuple[str, str, str, str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))

    lines = [_line(headers), _line(("-" * width for width in widths))]
    lines.extend(_line((row["state"], row["target"], row["candidate"], row["result"])) for row in rows)
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
