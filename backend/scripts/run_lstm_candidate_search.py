from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.experiment_profiles import normalize_experiment_profile
from app.services.lstm_candidate_search_notification import notify_lstm_candidate_search_finished
from app.services.lstm_candidate_retry import LstmCandidateRetryConfig, run_lstm_candidate_retry
from app.services.lstm_candidate_search import LstmCandidateSearchConfig
from app.services.runtime_symbols import configured_runtime_symbols


def main() -> None:
    args = _parse_args()
    config = LstmCandidateRetryConfig(
        symbols=_symbols(args.symbols),
        durations=_csv_strings(args.durations),
        profile=normalize_experiment_profile(args.profile),
        search=_search_config(args),
    )
    report = run_lstm_candidate_retry(config)
    notification = notify_lstm_candidate_search_finished(report)
    report["notification"] = notification
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def _search_config(args: argparse.Namespace) -> LstmCandidateSearchConfig:
    defaults = LstmCandidateSearchConfig()
    return LstmCandidateSearchConfig(
        feature_windows=_csv_ints(args.feature_windows) or defaults.feature_windows,
        min_move_bps_values=_csv_floats(args.min_move_bps) or defaults.min_move_bps_values,
        epoch_values=_csv_ints(args.epochs) or defaults.epoch_values,
        seeds=_csv_ints(args.seeds) or defaults.seeds,
        candidates_per_duration=args.candidates_per_duration or defaults.candidates_per_duration,
        parallel_workers=args.parallel_workers or defaults.parallel_workers,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LSTM candidate search and record every candidate.")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--durations", default="10m,30m,60m")
    parser.add_argument("--profile", default="full", choices=("fast", "full"))
    parser.add_argument("--feature-windows", default=None)
    parser.add_argument("--min-move-bps", default=None)
    parser.add_argument("--epochs", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--candidates-per-duration", type=int, default=None)
    parser.add_argument("--parallel-workers", type=int, default=None)
    return parser.parse_args()


def _csv_strings(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("CSV argument must include at least one value")
    return values


def _symbols(raw: str | None) -> tuple[str, ...]:
    return _csv_strings(raw) if raw is not None else configured_runtime_symbols()


def _csv_ints(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    return tuple(int(part) for part in _csv_strings(raw))


def _csv_floats(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    return tuple(float(part) for part in _csv_strings(raw))


if __name__ == "__main__":
    main()
