from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.model_family_config import MODEL_FAMILIES  # noqa: E402
from app.services.model_search_status_service import model_search_status_with_lifecycle  # noqa: E402
from app.services.runtime_symbols import parse_symbol_csv  # noqa: E402


def main() -> int:
    args = _parse_args()
    payload = model_search_status_with_lifecycle(_filters(args))
    if args.compact:
        print(_compact_status(payload))
    if args.json or not args.compact:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _filters(args: argparse.Namespace) -> dict:
    return {
        "symbols": parse_symbol_csv(args.symbols) if args.symbols else (),
        "durations": tuple(args.durations or ()),
        "families": tuple(args.families or ()),
        "statuses": tuple(args.statuses or ()),
    }


def _compact_status(payload: dict) -> str:
    lines = [f"total={payload['totalJobs']} counts={payload['counts']} latestLog={payload.get('latestLogPath') or '-'}"]
    for symbol in payload["symbols"]:
        lines.append(f"{symbol['symbol']} counts={symbol['counts']}")
        for duration in symbol["durations"]:
            lines.append(f"  {duration['duration']} counts={duration['counts']}")
            for family in duration["families"]:
                lines.append(_family_line(family))
    return "\n".join(lines)


def _family_line(family: dict) -> str:
    latest = family.get("latestJob") or {}
    return (
        f"    {family['modelFamily']} job={latest.get('status')} stage={latest.get('stage')} "
        f"model={family.get('modelStatus')} ready={family.get('shadowPredictionReady')} "
        f"reason={family.get('blockedReason') or latest.get('failure_reason') or latest.get('rejection_reason') or '-'}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show queued/running/completed model search jobs.")
    parser.add_argument("--symbols")
    parser.add_argument("--durations", nargs="+")
    parser.add_argument("--families", nargs="+", choices=MODEL_FAMILIES)
    parser.add_argument("--statuses", nargs="+")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
