#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.factor_combo_feature_backfill import (
    FactorComboSnapshotBackfillConfig,
    backfill_factor_combo_feature_snapshots,
)
from app.services.rule_config import SUPPORTED_RULE_DURATIONS


def main() -> int:
    args = _parse_args()
    payload = backfill_factor_combo_feature_snapshots(
        args.symbol,
        args.duration,
        FactorComboSnapshotBackfillConfig(
            lookback_rows=args.lookback_rows,
            step_rows=args.step_rows,
            min_history_rows=args.min_history_rows,
            ranking_limit=args.ranking_limit,
            candidate_limit=args.candidate_limit,
            min_trades=args.min_trades,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        ),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical factor combo feature snapshots.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--duration", required=True, choices=SUPPORTED_RULE_DURATIONS)
    parser.add_argument("--lookback-rows", type=int, default=1800)
    parser.add_argument("--step-rows", type=int, default=240)
    parser.add_argument("--min-history-rows", type=int, default=900)
    parser.add_argument("--ranking-limit", type=int, default=12)
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--min-trades", type=int, default=80)
    parser.add_argument("--threshold-min", type=float, default=0.8)
    parser.add_argument("--threshold-max", type=float, default=1.6)
    parser.add_argument("--threshold-step", type=float, default=0.4)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
