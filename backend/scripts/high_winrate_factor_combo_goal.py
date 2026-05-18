#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config.env_file import load_backend_env_file
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_performance_metrics import BACKTEST_MIN_PERIODS
from app.services.high_winrate_combo_cache_service import save_cached_high_winrate_combo_ranking
from app.services.high_winrate_combo_multi_duration import parse_durations, run_multi_duration_goal
from app.services.high_winrate_strategy_demotion import promote_high_winrate_strategy
from app.services.factor_combination_live_service import rebuild_combination_signal_watchlist
from app.services.factor_combination_signal_cache_service import save_cached_combination_signals
from app.services.factor_mined_library import upsert_good_combinations
from app.services import high_winrate_combo_goal_search as goal_search
from app.services.high_winrate_combo_goal_validation import validate_goal_combo_hits
from app.services.high_winrate_combo_goal_payloads import (
    ONLINE_RESEARCH_SOURCES,
    combo_display_name as _combo_display_name,
    library_payload as _library_payload,
    member_payload as _member_payload,
    paper_signal as _paper_signal,
    ranking_row as _ranking_row,
    report_payload as _report_payload,
    search_payload as _search_payload,
    target_payload as _target_payload,
)
from app.services.high_winrate_combo_goal_search import (
    COMBO_SIZES,
    EXCLUDED_COLUMNS,
    GoalSearchConfig,
    NEXT_ENTRY_HORIZON_BARS,
    SEARCH_CANDIDATE_LIMIT,
    SIGNAL_THRESHOLDS,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
    TARGET_COUNT,
    TARGET_WIN_RATE,
    ZSCORE_CLIP,
    ComboHit,
    OrientedScore,
    RankedSearch,
    ScoreSearch,
)

REPORT_PATH = BACKEND_ROOT / "reports" / "factor_backtests" / "high_winrate_factor_combo_goal.json"
LIBRARY_PATH = BACKEND_ROOT / "models" / "factor_learning" / "high_winrate_factor_combo_goal_library.json"
DEFAULT_PRIMARY_DURATIONS = ("10m", "60m")
DEFAULT_PARALLEL_WORKERS = 2

_best_combo_hit = goal_search.best_combo_hit
_combo_hit = goal_search.combo_hit
_combo_hit_result = goal_search.combo_hit_result
_expanding_zscore = goal_search.expanding_zscore
_numeric_factor_columns = goal_search.numeric_factor_columns
_orientation = goal_search.orientation_for_series
_oriented_score_search = goal_search.oriented_score_search
_oriented_scores = goal_search.oriented_scores
_profit_factor = goal_search.profit_factor
_ranked_hit_search = goal_search.ranked_hit_search
_ranked_hits = goal_search.ranked_hits
_search_candidate_names = goal_search.search_candidate_names
_search_frame = goal_search.search_frame
_selected_hits = goal_search.selected_hits
_usable_pair_count = goal_search.usable_pair_count


def __getattr__(name: str) -> Any:
    if name == "TARGET_MIN_TRADES":
        return goal_search.TARGET_MIN_TRADES
    raise AttributeError(name)


def run_goal(
    symbol: str,
    duration: str,
    target_count: int,
    output: Path,
    library: Path,
    search_config: GoalSearchConfig | None = None,
) -> dict[str, Any]:
    cfg = goal_search.validated_search_config(search_config)
    load_backend_env_file()
    frame = load_factor_frame(symbol, duration)
    search_frame = _search_frame(frame, duration)
    score_search = _oriented_score_search(search_frame)
    ranked_search = _ranked_hit_search(search_frame, score_search.scores, cfg)
    validation = validate_goal_combo_hits(search_frame, ranked_search.hits, duration)
    selected = _selected_hits(validation.passed, target_count)
    payload = _report_payload(
        symbol,
        duration,
        target_count,
        search_frame,
        score_search,
        ranked_search,
        selected,
        validation.payload,
        cfg,
    )
    if not payload["ranking"]:
        _write_json(output, payload)
        _write_json(library, _library_payload(payload))
        return payload
    payload["promotion"] = upsert_good_combinations(payload)
    save_cached_high_winrate_combo_ranking(payload)
    promote_high_winrate_strategy(symbol, duration)
    save_cached_combination_signals(rebuild_combination_signal_watchlist(symbol))
    _write_json(output, payload)
    _write_json(library, _library_payload(payload))
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find high-win-rate factor combos and paper-live signals.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--duration", help="Run one duration instead of the primary multi-duration set")
    parser.add_argument("--durations", default=",".join(DEFAULT_PRIMARY_DURATIONS))
    parser.add_argument("--target-count", type=int, default=TARGET_COUNT)
    parser.add_argument("--min-trades", type=int, default=BACKTEST_MIN_PERIODS)
    parser.add_argument("--candidate-limit", type=int, default=SEARCH_CANDIDATE_LIMIT)
    parser.add_argument("--threshold-min", type=float, default=THRESHOLD_MIN)
    parser.add_argument("--threshold-max", type=float, default=THRESHOLD_MAX)
    parser.add_argument("--threshold-step", type=float, default=THRESHOLD_STEP)
    parser.add_argument("--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--library", type=Path, default=LIBRARY_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    search_config = _search_config_from_args(args)
    durations = [] if args.duration else parse_durations(args.durations)
    if durations:
        report = run_multi_duration_goal(
            args.symbol,
            durations,
            args.target_count,
            args.output,
            args.library,
            lambda symbol, duration, target_count, output, library: run_goal(
                symbol,
                duration,
                target_count,
                output,
                library,
                search_config,
            ),
            parallel_workers=args.parallel_workers,
        )
        print(json.dumps(_multi_duration_stdout(args, report), ensure_ascii=False, indent=2))
        return
    duration = args.duration or DEFAULT_PRIMARY_DURATIONS[0]
    report = run_goal(args.symbol, duration, args.target_count, args.output, args.library, search_config)
    print(json.dumps(_single_duration_stdout(args, report), ensure_ascii=False, indent=2))


def _search_config_from_args(args: argparse.Namespace) -> GoalSearchConfig:
    thresholds = goal_search.signal_thresholds(args.threshold_min, args.threshold_max, args.threshold_step)
    return goal_search.validated_search_config(
        GoalSearchConfig(
            candidate_limit=args.candidate_limit,
            signal_thresholds=thresholds,
            min_trades=args.min_trades,
        )
    )


def _multi_duration_stdout(args: argparse.Namespace, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "output": str(args.output),
        "library": str(args.library),
        "promotion": report["promotion"],
        "bestRanking": report["bestRanking"],
    }


def _single_duration_stdout(args: argparse.Namespace, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "output": str(args.output),
        "library": str(args.library),
        "promotion": report.get("promotion"),
        "ranking": report["ranking"],
    }


if __name__ == "__main__":
    main()
