from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services.factor_learning_common import (
    DEFAULT_MIN_CONFIRMATIONS,
    REDUNDANCY_CORR_MIN,
    SUCCESS_IR_MIN,
    SUCCESS_PROFIT_FACTOR_MIN,
    SUCCESS_SHARPE_MIN,
    SUCCESS_WIN_RATE_MIN,
    TOP_FACTOR_LIMIT,
    TOP_PATTERN_LIMIT,
    edge_score,
    finite,
    formula_tokens,
    num,
    round_metric,
)


def factor_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in report.get("baseFactors") or []:
        normalized = _normal_factor_row(row)
        if normalized:
            rows[normalized["name"]] = normalized
    for combo in report.get("ranking") or []:
        for member in combo.get("members") or []:
            normalized = _normal_member_row(member)
            if normalized and normalized["name"] not in rows:
                rows[normalized["name"]] = normalized
    return list(rows.values())


def success_patterns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not _is_success_row(row):
            continue
        for key, label in _row_pattern_keys(row):
            grouped.setdefault(key, []).append({"label": label, "score": edge_score(row), "row": row})
    patterns = [_success_payload(key, items) for key, items in grouped.items()]
    patterns.sort(key=lambda item: (item["score"], item["support"]), reverse=True)
    return patterns[:TOP_PATTERN_LIMIT]


def forbidden_regions(frame: pd.DataFrame, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = [row["name"] for row in rows if row["name"] in frame.columns][:TOP_FACTOR_LIMIT]
    if len(names) < 2:
        return []
    corr = frame[names].apply(pd.to_numeric, errors="coerce").corr(method="spearman").abs()
    regions = _correlation_regions(corr, names)
    regions.sort(key=lambda item: (item["avgAbsCorrelation"], item["support"]), reverse=True)
    return regions[:TOP_PATTERN_LIMIT]


def candidate_loss_columns(rows: list[dict[str, Any]], frame: pd.DataFrame) -> list[str]:
    names = []
    for row in rows:
        name = row["name"]
        if name in frame.columns and name not in names:
            names.append(name)
    return names[:TOP_FACTOR_LIMIT]


def factor_weights(rows: list[dict[str, Any]], loss_patterns: list[dict[str, Any]]) -> dict[str, float]:
    loss_features = {str(pattern["feature"]) for pattern in loss_patterns}
    scored = [(_weight_score(row, loss_features), row["name"]) for row in rows]
    positive = [(score, name) for score, name in scored if score > 0]
    total = sum(score for score, _name in positive)
    if total <= 0:
        return {}
    return {name: round_metric(score / total, 6) for score, name in sorted(positive, reverse=True)}


def filter_config(loss_memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "minConfirmations": DEFAULT_MIN_CONFIRMATIONS,
        "lossPatternMaxMatches": 0 if loss_memory["patterns"] else None,
        "minHistoricalWinRate": 0.50,
    }


def _normal_factor_row(row: dict[str, Any]) -> dict[str, Any] | None:
    name = row.get("factorName") or row.get("name")
    if not name:
        return None
    return {
        "name": str(name),
        "category": str(row.get("category") or "unknown"),
        "formula": str(row.get("formula") or name),
        "sourceFile": str(row.get("sourceFile") or ""),
        "winRate": finite(row.get("winRate") or row.get("singleWinRate")),
        "ir": finite(row.get("ir") or row.get("singleIr")),
        "sharpe": finite(row.get("sharpe") or row.get("singleSharpe")),
        "profitFactor": finite(row.get("profitFactor")),
    }


def _normal_member_row(row: dict[str, Any]) -> dict[str, Any] | None:
    name = row.get("name")
    if not name:
        return None
    return {
        "name": str(name),
        "category": str(row.get("category") or "unknown"),
        "formula": str(name),
        "sourceFile": "",
        "winRate": finite(row.get("singleWinRate")),
        "ir": finite(row.get("singleIr")),
        "sharpe": finite(row.get("singleSharpe")),
        "profitFactor": None,
    }


def _is_success_row(row: dict[str, Any]) -> bool:
    return (
        num(row.get("winRate")) >= SUCCESS_WIN_RATE_MIN
        or num(row.get("profitFactor")) >= SUCCESS_PROFIT_FACTOR_MIN
        or num(row.get("sharpe")) >= SUCCESS_SHARPE_MIN
        or abs(num(row.get("ir"))) >= SUCCESS_IR_MIN
    )


def _row_pattern_keys(row: dict[str, Any]) -> list[tuple[str, str]]:
    keys = [(f"category:{row['category']}", f"category={row['category']}")]
    keys.extend((f"operator:{token}", f"operator={token}") for token in formula_tokens(row["formula"]))
    return keys


def _success_payload(key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    factors = [item["row"]["name"] for item in rows]
    scores = [item["score"] for item in rows]
    return {
        "pattern": key,
        "label": rows[0]["label"],
        "support": len(rows),
        "score": round_metric(sum(scores) / max(len(scores), 1), 4),
        "factors": factors[:8],
    }


def _correlation_regions(corr: pd.DataFrame, names: list[str]) -> list[dict[str, Any]]:
    visited: set[str] = set()
    regions = []
    for name in names:
        if name in visited:
            continue
        peers = _correlated_peers(corr, name)
        if len(peers) < 2:
            continue
        visited.update(peers)
        regions.append(_correlation_region_payload(corr, name, peers))
    return regions


def _correlated_peers(corr: pd.DataFrame, name: str) -> list[str]:
    values = corr[name].drop(labels=[name], errors="ignore")
    peers = [name, *values[values >= REDUNDANCY_CORR_MIN].index.tolist()]
    return sorted(set(peers))


def _correlation_region_payload(corr: pd.DataFrame, seed: str, peers: list[str]) -> dict[str, Any]:
    sub = corr.loc[peers, peers].where(~np.eye(len(peers), dtype=bool))
    avg_corr = float(sub.stack().mean()) if len(peers) > 1 else 0.0
    return {
        "region": f"correlation_cluster:{seed}",
        "reason": "redundant_factor_neighborhood",
        "support": len(peers),
        "avgAbsCorrelation": round_metric(avg_corr, 4),
        "members": peers[:10],
    }


def _weight_score(row: dict[str, Any], loss_features: set[str]) -> float:
    score = edge_score(row)
    if row["name"] in loss_features:
        score *= 0.5
    return max(score, 0.0)
