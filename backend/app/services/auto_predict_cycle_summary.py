from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.auto_predict_loop_status import record_auto_predict_cycle_progress
from app.services.auto_trade_types import AutoTradeSettings


@dataclass(frozen=True)
class PredictionCycleTargets:
    target_count: int
    due: tuple[AutoTradeSettings, ...]
    collection: tuple[AutoTradeSettings, ...]
    skipped: tuple[dict[str, Any], ...]

    @property
    def active(self) -> tuple[AutoTradeSettings, ...]:
        return tuple(_merged_targets(self.due, self.collection))


def prediction_cycle_targets(
    targets: list[AutoTradeSettings],
    due_targets: list[AutoTradeSettings],
    collection_targets: list[AutoTradeSettings],
    *,
    skipped_targets: list[dict[str, Any]],
) -> PredictionCycleTargets:
    return PredictionCycleTargets(
        target_count=len(targets),
        due=tuple(due_targets),
        collection=tuple(collection_targets),
        skipped=tuple(skipped_targets),
    )


def prediction_cycle_summary(cycle: PredictionCycleTargets) -> dict[str, Any]:
    return {
        "readyDueCount": len(cycle.due),
        "collectionTargetCount": len(cycle.collection),
        "activeTargetCount": len(cycle.active),
        "skippedTargetCount": len(cycle.skipped),
        "skippedTargets": list(cycle.skipped),
    }


def record_prediction_progress(cycle: PredictionCycleTargets, phase: str) -> None:
    record_auto_predict_cycle_progress(
        cycle.target_count,
        cycle_details={**prediction_cycle_summary(cycle), "phase": phase},
    )


def _merged_targets(
    first: tuple[AutoTradeSettings, ...],
    second: tuple[AutoTradeSettings, ...],
) -> list[AutoTradeSettings]:
    merged = {}
    for settings in [*first, *second]:
        merged[(settings.strategy_key, settings.symbol.upper(), settings.duration)] = settings
    return list(merged.values())
