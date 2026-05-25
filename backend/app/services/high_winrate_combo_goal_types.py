from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class OrientedScore:
    score: pd.Series
    orientation: int


@dataclass(frozen=True)
class ComboHit:
    members: tuple[str, ...]
    orientations: tuple[int, ...]
    threshold: float
    win_rate: float
    profit_factor: float
    trades: int
    avg_return: float
    score: pd.Series


@dataclass(frozen=True)
class ScoreSearch:
    scores: dict[str, OrientedScore]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RankedSearch:
    hits: list[ComboHit]
    diagnostics: dict[str, Any]
