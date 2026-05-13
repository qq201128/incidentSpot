from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Any

import pandas as pd

from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS, run_factor_backtest_on_frame
from app.services.factor_combo_scoring import combination_score
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_mined_candidates import build_mined_candidates
from app.services.factor_performance_metrics import add_contribution_scores
from app.services.factor_registry import FactorCategory, FactorDefinition, FactorDirection, factor_payload, list_factors
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

COMBINATION_METHOD = "expanding_oriented_zscore_mean_v1"
COMBO_SOURCE_FILE = "factor_combination_service.py"
DEFAULT_BASE_FACTOR_LIMIT = 16
DEFAULT_RESULT_LIMIT = 200
MIN_COMBO_SIZE = 2
DEFAULT_MAX_COMBO_SIZE = 3
DEFAULT_COMBO_SIZES = (MIN_COMBO_SIZE, DEFAULT_MAX_COMBO_SIZE)


@dataclass(frozen=True)
class CombinationSearchConfig:
    base_factor_limit: int = DEFAULT_BASE_FACTOR_LIMIT
    combo_sizes: tuple[int, ...] = DEFAULT_COMBO_SIZES
    result_limit: int = DEFAULT_RESULT_LIMIT
    method: str = COMBINATION_METHOD


@dataclass(frozen=True)
class _BaseCandidate:
    factor: FactorDefinition
    metrics: dict[str, Any]
    orientation: int


def run_factor_combination_ranking(
    symbol: str,
    duration: str,
    config: CombinationSearchConfig | None = None,
) -> dict[str, Any]:
    cfg = _validated_config(config or CombinationSearchConfig())
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    frame = load_factor_frame(symbol)
    return run_factor_combination_ranking_on_frame(
        frame,
        symbol=symbol,
        duration=duration,
        config=cfg,
    )


def run_factor_combination_ranking_on_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    duration: str,
    config: CombinationSearchConfig,
) -> dict[str, Any]:
    cfg = _validated_config(config)
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    mined = build_mined_candidates(frame, symbol=symbol.upper(), duration=duration)
    base, base_failures = _base_candidates(mined.frame, symbol.upper(), duration)
    base.extend(_mined_base_candidates(mined.candidates))
    selected = sorted(base, key=_base_rank_key, reverse=True)[: cfg.base_factor_limit]
    ranking, tested_count, combo_failures = _rank_combinations(mined.frame, selected, symbol.upper(), duration, cfg)
    add_contribution_scores(ranking)
    failures = [*base_failures, *mined.failures, *combo_failures]
    return _ranking_report(symbol, duration, cfg, selected, ranking, tested_count, failures, mined.source_count)


def _base_candidates(
    frame: pd.DataFrame,
    symbol: str,
    duration: str,
) -> tuple[list[_BaseCandidate], list[dict[str, Any]]]:
    candidates: list[_BaseCandidate] = []
    failures: list[dict[str, Any]] = []
    for factor in list_factors():
        if factor.name not in frame.columns:
            continue
        try:
            metrics = run_factor_backtest_on_frame(factor, frame, symbol=symbol, duration=duration)
            if _usable_base_metrics(metrics):
                candidates.append(_BaseCandidate(factor, metrics, _factor_orientation(factor, metrics)))
        except Exception as exc:
            failures.append({"factorName": factor.name, "stage": "single_factor", "error": str(exc)})
    return candidates, failures


def _rank_combinations(
    frame: pd.DataFrame,
    candidates: list[_BaseCandidate],
    symbol: str,
    duration: str,
    config: CombinationSearchConfig,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    ranking: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for size in config.combo_sizes:
        for members in combinations(candidates, size):
            _append_combination_result(ranking, failures, frame, members, symbol, duration)
    ranking.sort(key=_combo_rank_key, reverse=True)
    tested_count = len(ranking)
    return ranking[: config.result_limit], tested_count, failures


def _append_combination_result(
    ranking: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    frame: pd.DataFrame,
    members: tuple[_BaseCandidate, ...],
    symbol: str,
    duration: str,
) -> None:
    factor_def = _combination_definition(members, duration)
    try:
        combo_frame = _combo_backtest_frame(frame, members, factor_def.name)
        result = run_factor_backtest_on_frame(factor_def, combo_frame, symbol=symbol, duration=duration)
        ranking.append(_enriched_combo_result(result, members))
    except Exception as exc:
        failures.append({"factorName": factor_def.name, "stage": "combination", "error": str(exc)})


def _combo_backtest_frame(
    frame: pd.DataFrame,
    members: tuple[_BaseCandidate, ...],
    combo_name: str,
) -> pd.DataFrame:
    out = frame[["close"]].copy()
    if "open_time" in frame.columns:
        out["open_time"] = frame["open_time"]
    out[combo_name] = combination_score(frame, _member_payloads(members))
    return out


def _combination_definition(
    members: tuple[_BaseCandidate, ...],
    duration: str,
) -> FactorDefinition:
    names = [member.factor.name for member in members]
    return FactorDefinition(
        name="combo__" + "__".join(names),
        category=FactorCategory.PERFORMANCE,
        description="组合因子：" + " + ".join(member.factor.description for member in members),
        formula=f"{COMBINATION_METHOD}(" + ", ".join(names) + ")",
        source_file=COMBO_SOURCE_FILE,
        timeframes=(duration,),
        direction=FactorDirection.HIGHER_BETTER,
        parameters={"members": names, "method": COMBINATION_METHOD},
    )


def _enriched_combo_result(result: dict[str, Any], members: tuple[_BaseCandidate, ...]) -> dict[str, Any]:
    payload = {
        **result,
        "comboSize": len(members),
        "method": COMBINATION_METHOD,
        "members": _member_payloads(members),
    }
    payload["factorDisplayName"] = _combo_display_name(payload["members"])
    payload["description"] = payload["factorDisplayName"]
    return payload


def _member_payloads(members: tuple[_BaseCandidate, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": member.factor.name,
            "displayName": member.factor.description,
            "category": member.factor.category.value,
            "orientation": member.orientation,
            "singleWinRate": member.metrics.get("winRate"),
            "singleIr": member.metrics.get("ir"),
            "singleSharpe": member.metrics.get("sharpe"),
        }
        for member in members
    ]


def _mined_base_candidates(candidates: tuple[Any, ...]) -> list[_BaseCandidate]:
    return [_BaseCandidate(item.factor, item.metrics, item.orientation) for item in candidates]


def _ranking_report(
    symbol: str,
    duration: str,
    config: CombinationSearchConfig,
    selected: list[_BaseCandidate],
    ranking: list[dict[str, Any]],
    tested_count: int,
    failures: list[dict[str, Any]],
    mined_source_count: int,
) -> dict[str, Any]:
    return {
        "symbol": symbol.upper(),
        "duration": duration,
        "ranking": ranking,
        "total": len(ranking),
        "searchConfig": _config_payload(config),
        "baseFactors": [_base_payload(item) for item in selected],
        "baseFactorCount": len(selected),
        "minedFactorSourceCount": mined_source_count,
        "minedFactorUsedCount": _mined_factor_count(selected),
        "testedCombinationCount": tested_count,
        "failureCount": len(failures),
        "failures": failures[:50],
    }


def _base_payload(candidate: _BaseCandidate) -> dict[str, Any]:
    return {
        **factor_payload(candidate.factor),
        "orientation": candidate.orientation,
        "singleWinRate": candidate.metrics.get("winRate"),
        "singleIr": candidate.metrics.get("ir"),
        "singleSharpe": candidate.metrics.get("sharpe"),
    }


def _mined_factor_count(selected: list[_BaseCandidate]) -> int:
    return sum(1 for item in selected if item.factor.source_file == "mined_factor_library.json")


def _factor_orientation(factor: FactorDefinition, metrics: dict[str, Any]) -> int:
    if factor.direction == FactorDirection.LOWER_BETTER:
        return -1
    if factor.direction == FactorDirection.HIGHER_BETTER:
        return 1
    return 1 if _metric_sign(metrics) >= 0 else -1


def _metric_sign(metrics: dict[str, Any]) -> float:
    for key in ("icMean", "longShortReturn", "ir"):
        value = _finite_float(metrics.get(key))
        if value is not None and value != 0:
            return value
    return 1.0


def _usable_base_metrics(metrics: dict[str, Any]) -> bool:
    return int(metrics.get("totalPeriods") or 0) >= BACKTEST_MIN_PERIODS and metrics.get("winRate") is not None

def _base_rank_key(candidate: _BaseCandidate) -> tuple[float, float, float, int]:
    row = candidate.metrics
    return (_num(row.get("winRate")), _num(row.get("sharpe")), _abs_num(row.get("ir")), int(row.get("totalPeriods") or 0))

def _combo_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (_num(row.get("winRate")), _num(row.get("profitFactor")), _num(row.get("sharpe")), _abs_num(row.get("ir")))

def _config_payload(config: CombinationSearchConfig) -> dict[str, Any]:
    return {
        "baseFactorLimit": config.base_factor_limit,
        "comboSizes": list(config.combo_sizes),
        "resultLimit": config.result_limit,
        "method": config.method,
        "minPeriods": BACKTEST_MIN_PERIODS,
    }

def _validated_config(config: CombinationSearchConfig) -> CombinationSearchConfig:
    sizes = tuple(sorted(set(int(size) for size in config.combo_sizes)))
    if config.method != COMBINATION_METHOD:
        raise ValueError(f"unsupported combination method: {config.method}")
    if not sizes or config.base_factor_limit < max(sizes):
        raise ValueError("combo sizes must be non-empty and base_factor_limit must be >= largest combo size")
    if config.result_limit <= 0 or any(size < MIN_COMBO_SIZE for size in sizes):
        raise ValueError("result_limit must be > 0 and combo sizes must be >= 2")
    return CombinationSearchConfig(int(config.base_factor_limit), sizes, int(config.result_limit), config.method)

def _combo_display_name(members: list[dict[str, Any]]) -> str:
    return "组合：" + " + ".join(str(member["displayName"]) for member in members)

def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None

def _num(value: Any) -> float:
    number = _finite_float(value)
    return number if number is not None else float("-inf")

def _abs_num(value: Any) -> float:
    number = _finite_float(value)
    return abs(number) if number is not None else 0.0
