from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services.factor_combo_scoring import oriented_zscore

RIDGE_L2 = 0.05
CORRELATION_PENALTY = 0.001
MIN_WEIGHT = 0.0


def learned_member_payloads(frame: pd.DataFrame, members: list[dict[str, Any]], train_end: int) -> list[dict[str, Any]]:
    zscores = _member_zscores(frame, members)
    weights = _non_negative_weights(zscores.iloc[:train_end], frame["fwd_ret"].iloc[:train_end])
    return [{**member, "weight": round(float(weight), 6)} for member, weight in zip(members, weights)]


def weighted_score(frame: pd.DataFrame, members: list[dict[str, Any]]) -> pd.Series:
    scores = [_weighted_member_score(frame, member) for member in members]
    weights = np.asarray([float(member.get("weight") or 0.0) for member in members], dtype=float)
    if float(weights.sum()) <= 0:
        weights = np.ones(len(members), dtype=float) / len(members)
    stacked = pd.concat(scores, axis=1)
    return stacked.dot(weights / weights.sum()).replace([np.inf, -np.inf], np.nan)


def _member_zscores(frame: pd.DataFrame, members: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [_weighted_member_score(frame, member) for member in members]
    return pd.concat(columns, axis=1)


def _weighted_member_score(frame: pd.DataFrame, member: dict[str, Any]) -> pd.Series:
    name = str(member["name"])
    if name not in frame.columns:
        raise ValueError(f"combination score missing factor column: {name}")
    return oriented_zscore(frame[name], int(member.get("orientation") or 1)).rename(name)


def _non_negative_weights(zscores: pd.DataFrame, target: pd.Series) -> np.ndarray:
    aligned = pd.concat([zscores, target.rename("target")], axis=1).dropna()
    if aligned.empty:
        return np.ones(zscores.shape[1], dtype=float) / zscores.shape[1]
    x = aligned.iloc[:, :-1].to_numpy(dtype=float)
    y = aligned["target"].to_numpy(dtype=float)
    raw = np.maximum(_ridge_solution(x, y) - _correlation_penalty(x), MIN_WEIGHT)
    return raw / raw.sum() if float(raw.sum()) > 0 else np.ones(x.shape[1], dtype=float) / x.shape[1]


def _ridge_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    gram = x.T @ x + np.eye(x.shape[1]) * RIDGE_L2
    return np.linalg.solve(gram, x.T @ y)


def _correlation_penalty(x: np.ndarray) -> np.ndarray:
    if x.shape[1] <= 1:
        return np.zeros(x.shape[1], dtype=float)
    corr = np.nan_to_num(np.corrcoef(x, rowvar=False), nan=0.0)
    avg_abs = (np.abs(corr).sum(axis=1) - 1.0) / (x.shape[1] - 1)
    return avg_abs * CORRELATION_PENALTY
