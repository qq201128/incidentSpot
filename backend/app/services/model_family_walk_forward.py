from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np

from app.services.lstm_feature_builder import LstmDataset
from app.services.model_family_candidate_executor import CandidateTrainingResult
from app.services.model_family_config import ModelFamilyTrainingConfig
from app.services.model_family_training_service import train_model_family

FOLD_COUNT = 3
MIN_FOLD_SAMPLE_COUNT = 30


DatasetBuilder = Callable[[ModelFamilyTrainingConfig], LstmDataset]


def run_walk_forward_stage(finalists: list[CandidateTrainingResult], dataset_builder: DatasetBuilder) -> tuple[list[CandidateTrainingResult], dict[str, Any]]:
    evaluated = [_evaluate_candidate(item, dataset_builder) for item in finalists]
    advanced = [item for item in evaluated if item.report["walkForwardStage"]["status"] == "passed"]
    return advanced, _stage_payload(evaluated, advanced)


def _evaluate_candidate(item: CandidateTrainingResult, dataset_builder: DatasetBuilder) -> CandidateTrainingResult:
    dataset = dataset_builder(item.config)
    folds = [_train_fold(item.config, dataset, fold) for fold in _fold_ranges(len(dataset.x))]
    failed = [fold for fold in folds if fold["status"] != "passed"]
    stage = {
        "status": "passed" if not failed else "failed",
        "reason": None if not failed else failed[0]["reason"],
        "folds": folds,
        "foldCount": len(folds),
        "testEvaluationPolicy": "walk_forward_fold_test_not_global_final_test",
    }
    return CandidateTrainingResult(item.config, {**item.report, "searchStage": "walk_forward", "walkForwardStage": stage})


def _train_fold(config: ModelFamilyTrainingConfig, dataset: LstmDataset, fold: dict[str, int]) -> dict[str, Any]:
    fold_dataset = _slice_dataset(dataset, fold["start"], fold["end"])
    total = len(fold_dataset.x)
    train_ratio = (fold["trainEnd"] - fold["start"]) / total
    val_ratio = (fold["validationEnd"] - fold["trainEnd"]) / total
    fold_config = replace(config, train_ratio=train_ratio, val_ratio=val_ratio)
    report = train_model_family(
        fold_config,
        dataset_builder=lambda _cfg: fold_dataset,
        publish_shadow_active=False,
        publish_trade_active=False,
        write_attempt=False,
        persist_artifacts=False,
        evaluate_test=True,
    )
    return _fold_payload(fold, report)


def _fold_ranges(total: int) -> list[dict[str, int]]:
    fold = max(MIN_FOLD_SAMPLE_COUNT, total // (FOLD_COUNT + 2))
    train = max(fold * 2, MIN_FOLD_SAMPLE_COUNT * 2)
    required = train + fold * 2
    if total < required:
        raise ValueError(f"insufficient samples for model family walk-forward: {total} < {required}")
    starts = np.linspace(0, total - required, FOLD_COUNT, dtype=int)
    return [_fold_range(int(start), train, fold) for start in starts]


def _fold_range(start: int, train: int, holdout: int) -> dict[str, int]:
    train_end = start + train
    validation_end = train_end + holdout
    end = validation_end + holdout
    return {"start": start, "trainEnd": train_end, "validationEnd": validation_end, "end": end}


def _slice_dataset(dataset: LstmDataset, start: int, end: int) -> LstmDataset:
    return LstmDataset(
        x=dataset.x[start:end],
        y=dataset.y[start:end],
        future_returns=dataset.future_returns[start:end],
        entry_open_times=dataset.entry_open_times[start:end],
        feature_columns=list(dataset.feature_columns),
        feature_frame=dataset.feature_frame.iloc[start:end].copy(),
        combo_snapshot=list(dataset.combo_snapshot),
        learning_context=dataset.learning_context,
    )


def _fold_payload(fold: dict[str, int], report: dict[str, Any]) -> dict[str, Any]:
    reason = report.get("validationFailureReason")
    return {
        **fold,
        "status": "passed" if report.get("validationGate", {}).get("status") == "passed" else "failed",
        "reason": reason,
        "sampleCounts": report.get("sampleCounts"),
        "validation": report.get("validation"),
        "test": report.get("test"),
        "validationGate": report.get("validationGate"),
    }


def _stage_payload(evaluated: list[CandidateTrainingResult], advanced: list[CandidateTrainingResult]) -> dict[str, Any]:
    return {
        "stage": "walk_forward",
        "evaluated": len(evaluated),
        "advanced": len(advanced),
        "candidateKeys": _search_keys(evaluated),
        "advancedKeys": _search_keys(advanced),
        "candidates": [item.report["walkForwardStage"] for item in evaluated],
    }


def _search_keys(results: list[CandidateTrainingResult]) -> list[str]:
    return [str(item.report.get("searchKey")) for item in results if item.report.get("searchKey")]
