from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from app.services.model_family_candidates import read_model_candidate_library  # noqa: E402
from app.services.model_family_config import MODEL_FAMILIES  # noqa: E402
from app.services.model_family_search_rules import model_family_training_rules  # noqa: E402
from app.services.model_family_status_service import model_family_status  # noqa: E402


DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_DURATIONS = ("10m", "60m")


def main() -> int:
    args = _parse_args()
    rows = [_row(args.symbol, duration, family) for duration in args.durations for family in args.families]
    payload = {"symbol": args.symbol, "generatedAt": _utc_now(), "rows": rows}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _row(symbol: str, duration: str, family: str) -> dict[str, Any]:
    status = model_family_status(family, symbol, duration)
    progress = status.get("candidateSearchProgress") or {}
    library = read_model_candidate_library(family, symbol, duration)
    records = library.get("records") or []
    rules = model_family_training_rules(family)
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
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--durations", nargs="+", default=list(DEFAULT_DURATIONS))
    parser.add_argument("--families", nargs="+", default=list(MODEL_FAMILIES))
    parser.add_argument("--output")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
