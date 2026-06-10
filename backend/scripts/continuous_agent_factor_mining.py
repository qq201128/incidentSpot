#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config.env_file import load_backend_env_file  # noqa: E402
from app.db.session import init_db  # noqa: E402
from app.services.factor_learning_refresh_tasks import (  # noqa: E402
    mark_factor_learning_refresh_completed,
    mark_factor_learning_refresh_failed,
    mark_factor_learning_refresh_queued,
    mark_factor_learning_refresh_running,
)
from app.services.factor_learning_service import (  # noqa: E402
    mark_factor_learning_agent_failed,
    mark_factor_learning_agent_pending,
    mark_factor_learning_agent_running,
    refresh_factor_learning_memory,
    run_factor_learning_llm_agent,
)
from app.services.market_context_ingest_service import (  # noqa: E402
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_CONTEXT_PERIOD,
    ingest_market_context_data,
)
from app.services.runtime_symbols import configured_runtime_symbols, parse_symbol_csv  # noqa: E402
from app.services.rule_config import SUPPORTED_RULE_DURATIONS  # noqa: E402

DEFAULT_POLL_SECONDS = 900.0
DEFAULT_DURATIONS = ("10m",)
DEFAULT_FACTOR_LOOKBACK_DAYS = 30


@dataclass(frozen=True)
class AgentMiningConfig:
    symbols: tuple[str, ...]
    durations: tuple[str, ...]
    factor_lookback_days: int | None
    context_period: str
    context_limit: int
    poll_seconds: float
    ingest_market_context: bool
    compact: bool


def main() -> int:
    load_backend_env_file()
    args = _parse_args()
    config = _config_from_args(args)
    init_db()
    if args.loop:
        return _run_forever(config)
    _print_report(run_agent_mining_cycle(config), compact=config.compact)
    return 0


def run_agent_mining_cycle(config: AgentMiningConfig) -> dict[str, Any]:
    context_reports = _market_context_reports(config)
    target_reports = [
        _run_symbol_duration_agent(
            symbol,
            duration,
            factor_lookback_days=config.factor_lookback_days,
        )
        for symbol in config.symbols
        for duration in config.durations
    ]
    return {
        "status": "completed",
        "symbols": list(config.symbols),
        "durations": list(config.durations),
        "factorLookbackDays": config.factor_lookback_days,
        "marketContext": context_reports,
        "targets": target_reports,
    }


def _run_forever(config: AgentMiningConfig) -> int:
    while True:
        _print_report(run_agent_mining_cycle(config), compact=config.compact)
        time.sleep(config.poll_seconds)


def _market_context_reports(config: AgentMiningConfig) -> list[dict[str, Any]]:
    if not config.ingest_market_context:
        return []
    return [
        ingest_market_context_data(
            symbol,
            period=config.context_period,
            limit=config.context_limit,
            durations=config.durations,
        )
        for symbol in config.symbols
    ]


def _run_symbol_duration_agent(
    symbol: str,
    duration: str,
    *,
    factor_lookback_days: int | None,
) -> dict[str, Any]:
    stage = "queue_refresh"
    try:
        queued = mark_factor_learning_refresh_queued(symbol, duration, run_agent=True)
        mark_factor_learning_agent_pending(queued)
        stage = "refresh_memory"
        mark_factor_learning_refresh_running(symbol, duration, run_agent=True)
        memory = refresh_factor_learning_memory(
            symbol,
            duration,
            run_llm_agent=False,
            factor_lookback_days=factor_lookback_days,
        )
        completed = mark_factor_learning_refresh_completed(memory, run_agent=True)
        stage = "llm_agent"
        mark_factor_learning_agent_running(completed)
        memory = run_factor_learning_llm_agent(symbol, duration, factor_lookback_days=factor_lookback_days)
        return _target_report(memory, factor_lookback_days=factor_lookback_days)
    except Exception as exc:
        error = f"{stage}: {exc}"
        mark_factor_learning_refresh_failed(symbol, duration, error, run_agent=True)
        mark_factor_learning_agent_failed(symbol, duration, error)
        raise RuntimeError(
            f"continuous agent factor mining failed symbol={symbol} duration={duration} stage={stage}: {exc}"
        ) from exc


def _target_report(memory: dict[str, Any], *, factor_lookback_days: int | None) -> dict[str, Any]:
    agent = memory.get("llmAgent") or {}
    evaluation = memory.get("agentCandidateEvaluation") or {}
    library = memory.get("agentMinedFactorLibrary") or {}
    return {
        "status": "completed",
        "symbol": memory.get("symbol"),
        "duration": memory.get("duration"),
        "factorLookbackDays": factor_lookback_days,
        "memoryPath": memory.get("memoryPath"),
        "agentStatus": agent.get("status"),
        "agentModel": agent.get("model"),
        "completionId": agent.get("completionId"),
        "usage": agent.get("usage") or {},
        "generatedCount": evaluation.get("generatedCount"),
        "promotedCount": evaluation.get("promotedCount"),
        "failedCount": evaluation.get("failedCount"),
        "libraryTotal": library.get("total"),
        "simulationEligibleTotal": library.get("simulationEligibleTotal"),
    }


def _print_report(report: dict[str, Any], *, compact: bool) -> None:
    payload = _compact_report(report) if compact else report
    _write_json_report(payload, sys.stdout)


def _write_json_report(payload: dict[str, Any], stream: TextIO) -> None:
    json.dump(payload, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
    stream.flush()


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "symbols": report.get("symbols"),
        "durations": report.get("durations"),
        "factorLookbackDays": report.get("factorLookbackDays"),
        "marketContextCount": len(report.get("marketContext") or []),
        "targets": [
            _compact_target(target)
            for target in report.get("targets") or []
            if isinstance(target, dict)
        ],
    }


def _compact_target(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": target.get("symbol"),
        "duration": target.get("duration"),
        "factorLookbackDays": target.get("factorLookbackDays"),
        "agentStatus": target.get("agentStatus"),
        "generatedCount": target.get("generatedCount"),
        "promotedCount": target.get("promotedCount"),
        "failedCount": target.get("failedCount"),
        "libraryTotal": target.get("libraryTotal"),
        "simulationEligibleTotal": target.get("simulationEligibleTotal"),
    }


def _config_from_args(args: argparse.Namespace) -> AgentMiningConfig:
    return AgentMiningConfig(
        symbols=_selected_symbols(args),
        durations=_selected_durations(args.durations),
        factor_lookback_days=_selected_factor_lookback_days(args),
        context_period=args.context_period,
        context_limit=args.context_limit,
        poll_seconds=args.poll_seconds,
        ingest_market_context=not args.skip_market_context,
        compact=args.compact,
    )


def _selected_symbols(args: argparse.Namespace) -> tuple[str, ...]:
    if args.symbols and args.symbol:
        raise ValueError("use either positional symbol or --symbols, not both")
    if args.symbols:
        return parse_symbol_csv(args.symbols)
    if args.symbol:
        return parse_symbol_csv(args.symbol)
    return configured_runtime_symbols()


def _selected_durations(raw: str) -> tuple[str, ...]:
    durations = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    if not durations:
        raise ValueError("--durations must include at least one duration")
    unsupported = sorted(set(durations) - set(SUPPORTED_RULE_DURATIONS))
    if unsupported:
        raise ValueError(f"unsupported durations: {unsupported}")
    return durations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously run real networked LLM factor mining.")
    parser.add_argument("symbol", nargs="?")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--durations", default=",".join(DEFAULT_DURATIONS))
    parser.add_argument("--factor-lookback-days", type=_positive_int, default=DEFAULT_FACTOR_LOOKBACK_DAYS)
    parser.add_argument("--full-factor-history", action="store_true")
    parser.add_argument("--context-period", default=DEFAULT_CONTEXT_PERIOD)
    parser.add_argument("--context-limit", type=_positive_int, default=DEFAULT_CONTEXT_LIMIT)
    parser.add_argument("--skip-market-context", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--poll-seconds", type=_positive_float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def _selected_factor_lookback_days(args: argparse.Namespace) -> int | None:
    if args.full_factor_history:
        return None
    return args.factor_lookback_days


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


if __name__ == "__main__":
    raise SystemExit(main())
