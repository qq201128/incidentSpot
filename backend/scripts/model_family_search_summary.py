from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.model_family_candidates import read_model_candidate_library  # noqa: E402
from app.services.model_family_config import MODEL_FAMILIES  # noqa: E402
from app.services.model_family_search_rules import model_family_training_rules  # noqa: E402
from app.services.model_family_status_service import model_family_status  # noqa: E402
from app.services.runtime_symbols import configured_runtime_symbols, parse_symbol_csv  # noqa: E402


DEFAULT_DURATIONS = ("10m", "60m")
DEFAULT_WATCH_INTERVAL_SECONDS = 15


def main() -> int:
    args = _parse_args()
    if args.watch:
        return _watch_loop(args)
    symbols = _selected_symbols(args)
    payload = {
        "symbols": [
            {
                "symbol": symbol,
                "rows": [_row(symbol, duration, family) for duration in args.durations for family in args.families],
            }
            for symbol in symbols
        ],
        "generatedAt": _utc_now(),
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.compact:
        for item in payload["symbols"]:
            print(item["symbol"], file=sys.stderr if args.json else sys.stdout)
            print(_compact_table(item["rows"]), file=sys.stderr if args.json else sys.stdout)
    if args.json or not args.compact:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _watch_loop(args) -> int:
    try:
        while True:
            for symbol in _selected_symbols(args):
                rows = [_row(symbol, duration, family) for duration in args.durations for family in args.families]
                _print_watch_header(symbol, rows)
                print(_compact_table(rows))
            print(f"refresh every {args.interval}s | Ctrl+C to stop", flush=True)
            time.sleep(max(int(args.interval), 1))
    except KeyboardInterrupt:
        print("\nwatch stopped.", file=sys.stderr)
        return 0


def _print_watch_header(symbol: str, rows: list[dict[str, Any]]) -> None:
    done = sum(1 for row in rows if str(row.get("progressStatus") or "") in {"completed", "exhausted", "failed"})
    running = sum(1 for row in rows if str(row.get("progressStatus") or "") == "running")
    ready = sum(1 for row in rows if row.get("shadowPredictionReady"))
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{stamp}] {symbol} | done={done}/{len(rows)} running={running} shadowReady={ready}")


def _compact_table(rows: list[dict[str, Any]]) -> str:
    headers = ("DUR", "FAMILY", "PROGRESS", "PSTATUS", "MODEL", "READY")
    body = []
    for row in rows:
        completed = row.get("completed")
        total = row.get("total")
        progress = "-" if completed is None or total is None else f"{completed}/{total}"
        body.append(
            (
                str(row.get("duration") or "-"),
                str(row.get("family") or "-"),
                progress,
                str(row.get("progressStatus") or "-"),
                str(row.get("modelStatus") or "-"),
                "yes" if row.get("shadowPredictionReady") else "no",
            )
        )
    widths = [
        max(len(headers[i]), *(len(item[i]) for item in body), 3)
        for i in range(len(headers))
    ]

    def _line(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))

    lines = [_line(headers), _line(tuple("-" * width for width in widths))]
    lines.extend(_line(item) for item in body)
    return "\n".join(lines)


def _row(symbol: str, duration: str, family: str) -> dict[str, Any]:
    rules = model_family_training_rules(family)
    try:
        status = model_family_status(family, symbol, duration)
        progress = status.get("candidateSearchProgress") or {}
        library = read_model_candidate_library(family, symbol, duration)
        records = library.get("records") or []
    except Exception as exc:
        return _status_error_row(symbol, duration, family, rules, exc)
    return {
        "duration": duration,
        "family": family,
        "modelStatus": status.get("status"),
        "shadowPredictionReady": status.get("shadowPredictionReady"),
        "blockedReason": status.get("shadowPredictionBlockedReason"),
        "progressStatus": progress.get("status"),
        "completed": progress.get("completed"),
        "total": progress.get("total"),
        "searchSpaceTotal": rules.get("searchSpaceTotal"),
        "libraryTotal": len(records),
        "counts": progress.get("counts"),
        "best": _best(records),
    }


def _status_error_row(
    symbol: str,
    duration: str,
    family: str,
    rules: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    return {
        "duration": duration,
        "family": family,
        "modelStatus": "status_failed",
        "shadowPredictionReady": False,
        "blockedReason": str(exc),
        "progressStatus": "status_failed",
        "completed": None,
        "total": rules.get("searchSpaceTotal"),
        "searchSpaceTotal": rules.get("searchSpaceTotal"),
        "libraryTotal": None,
        "counts": None,
        "best": None,
        "error": str(exc),
        "exceptionType": type(exc).__name__,
        "symbol": symbol,
    }


def _best(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in records if row.get("validation") and row.get("test")]
    if not valid:
        return None
    record = max(valid, key=_score)
    validation = record.get("validation") or {}
    test = record.get("test") or {}
    return {
        "status": record.get("status"),
        "searchKey": record.get("searchKey"),
        "validationWinRate": validation.get("winRate"),
        "testWinRate": test.get("winRate"),
        "minWinRate": min(float(validation.get("winRate") or 0.0), float(test.get("winRate") or 0.0)),
        "modelVersion": record.get("modelVersion"),
    }


def _score(record: dict[str, Any]) -> tuple[float, float, int]:
    validation = record.get("validation") or {}
    test = record.get("test") or {}
    return (
        min(float(validation.get("winRate") or 0.0), float(test.get("winRate") or 0.0)),
        min(float(validation.get("profitFactor") or 0.0), float(test.get("profitFactor") or 0.0)),
        int(test.get("sampleCount") or 0),
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--durations", nargs="+", default=list(DEFAULT_DURATIONS))
    parser.add_argument("--families", nargs="+", default=list(MODEL_FAMILIES))
    parser.add_argument("--output")
    parser.add_argument("--compact", action="store_true", help="Print a compact progress table.")
    parser.add_argument("--json", action="store_true", help="With --compact, also print full JSON.")
    parser.add_argument("--watch", action="store_true", help="Poll and refresh progress until Ctrl+C.")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_WATCH_INTERVAL_SECONDS,
        help="Refresh interval for --watch (seconds).",
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
