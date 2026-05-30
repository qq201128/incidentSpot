from __future__ import annotations

import pandas as pd
import pytest

from app.services.lstm_config import LstmTrainingConfig
from app.services.lstm_feature_builder import build_lstm_training_dataset
from app.services.model_family_config import ModelFamilyTrainingConfig
from app.services import lstm_feature_builder
from app.services.sim_feedback_features import (
    SIM_FEEDBACK_PREFIX,
    attach_sim_feedback_features,
    normalize_sim_feedback_prediction_family,
    sim_feedback_feature_names,
)


def test_normalize_sim_feedback_prediction_family_accepts_factor_combo() -> None:
    assert normalize_sim_feedback_prediction_family("factor_combo") == "factor_combo"
    assert normalize_sim_feedback_prediction_family("FACTOR") == "factor"
    assert normalize_sim_feedback_prediction_family("lstm") == "lstm"


def test_attach_sim_feedback_counts_factor_combo_predictions() -> None:
    frame = _labeled_frame([3000])
    predictions = [
        {
            "open_time": 2000,
            "actual_return": 0.02,
            "prediction_correct": 1,
            "confidence": 0.9,
            "model_family": "factor_combo",
            "strategy_key": "factor_combo_goal_top1",
        },
    ]

    enriched = attach_sim_feedback_features(
        frame,
        "BTCUSDT",
        "10m",
        model_family="factor_combo",
        predictions_loader=lambda *_args: predictions,
    )

    row = enriched.loc[enriched["entry_open_time"] == 3000].iloc[0]
    assert row[f"{SIM_FEEDBACK_PREFIX}settled_count"] == 1.0
    assert row[f"{SIM_FEEDBACK_PREFIX}family_factor_combo_settled_count"] == 1.0
    assert row[f"{SIM_FEEDBACK_PREFIX}family_factor_combo_win_rate"] == pytest.approx(1.0)


def test_sim_feedback_feature_names_include_family_columns() -> None:
    global_names = sim_feedback_feature_names()
    family_names = sim_feedback_feature_names("xgboost")

    assert f"{SIM_FEEDBACK_PREFIX}win_rate" in global_names
    assert f"{SIM_FEEDBACK_PREFIX}family_xgboost_win_rate" in family_names
    assert len(family_names) == len(global_names) + 4


def test_attach_sim_feedback_is_causal() -> None:
    frame = _labeled_frame([1000, 2000, 3000, 4000])
    predictions = [
        {"open_time": 2000, "actual_return": 0.01, "prediction_correct": 1, "confidence": 0.8},
        {"open_time": 3000, "actual_return": -0.01, "prediction_correct": 0, "confidence": 0.7},
    ]

    enriched = attach_sim_feedback_features(
        frame,
        "BTCUSDT",
        "10m",
        predictions_loader=lambda *_args: predictions,
    )

    row_at_2000 = enriched.loc[enriched["entry_open_time"] == 2000].iloc[0]
    row_at_3000 = enriched.loc[enriched["entry_open_time"] == 3000].iloc[0]
    row_at_4000 = enriched.loc[enriched["entry_open_time"] == 4000].iloc[0]

    assert row_at_2000[f"{SIM_FEEDBACK_PREFIX}settled_count"] == 0.0
    assert row_at_3000[f"{SIM_FEEDBACK_PREFIX}settled_count"] == 1.0
    assert row_at_3000[f"{SIM_FEEDBACK_PREFIX}win_rate"] == pytest.approx(1.0)
    assert row_at_4000[f"{SIM_FEEDBACK_PREFIX}settled_count"] == 2.0
    assert row_at_4000[f"{SIM_FEEDBACK_PREFIX}win_rate"] == pytest.approx(0.5)
    assert row_at_4000[f"{SIM_FEEDBACK_PREFIX}loss_streak"] == 1.0
    assert enriched.attrs["simFeedbackMetadata"]["settledCount"] == 2


def test_attach_sim_feedback_adds_family_specific_columns() -> None:
    frame = _labeled_frame([1000, 3000])
    predictions = [
        {
            "open_time": 1000,
            "actual_return": 0.01,
            "prediction_correct": 1,
            "confidence": 0.8,
            "model_family": "xgboost",
            "signal_key": "factor_xgboost_shadow_10m",
        },
        {
            "open_time": 2000,
            "actual_return": -0.01,
            "prediction_correct": 0,
            "confidence": 0.6,
            "model_family": "lstm",
            "signal_key": "factor_lstm_shadow_10m",
        },
    ]

    enriched = attach_sim_feedback_features(
        frame,
        "BTCUSDT",
        "10m",
        model_family="xgboost",
        predictions_loader=lambda *_args: predictions,
    )

    row = enriched.loc[enriched["entry_open_time"] == 3000].iloc[0]
    assert row[f"{SIM_FEEDBACK_PREFIX}settled_count"] == 2.0
    assert row[f"{SIM_FEEDBACK_PREFIX}family_xgboost_settled_count"] == 1.0
    assert row[f"{SIM_FEEDBACK_PREFIX}family_xgboost_win_rate"] == pytest.approx(1.0)


def test_attach_sim_feedback_loader_exception_is_not_swallowed() -> None:
    frame = _labeled_frame([1000])

    def broken_loader(*_args):
        raise RuntimeError("database read failed")

    with pytest.raises(RuntimeError, match="database read failed"):
        attach_sim_feedback_features(frame, "BTCUSDT", "10m", predictions_loader=broken_loader)


def test_attach_sim_feedback_empty_samples_marks_neutral_metadata() -> None:
    frame = _labeled_frame([1000, 2000])

    enriched = attach_sim_feedback_features(
        frame,
        "BTCUSDT",
        "10m",
        predictions_loader=lambda *_args: [],
    )

    metadata = enriched.attrs["simFeedbackMetadata"]
    assert metadata["settledCount"] == 0
    assert metadata["neutralFeaturesUsed"] is True
    assert enriched[f"{SIM_FEEDBACK_PREFIX}win_rate"].tolist() == [0.5, 0.5]


def test_build_lstm_training_dataset_includes_sim_feedback_columns(monkeypatch) -> None:
    frame = _training_frame()
    monkeypatch.setattr(lstm_feature_builder, "load_factor_learning_memory_for", lambda *_args: None)
    monkeypatch.setattr(lstm_feature_builder, "build_lstm_market_feature_frame", lambda *_args, **_kwargs: frame)
    monkeypatch.setattr(lstm_feature_builder, "resolve_lstm_combo_ranking", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        lstm_feature_builder,
        "attach_sim_feedback_features",
        lambda labeled, *_args, **_kwargs: labeled.assign(
            **{name: 0.0 for name in sim_feedback_feature_names("lstm")}
        ),
    )
    monkeypatch.setattr(lstm_feature_builder, "load_factor_combo_feature_snapshots", _combo_feature_snapshots)

    dataset = build_lstm_training_dataset(
        ModelFamilyTrainingConfig(
            family="lstm",
            symbol="BTCUSDT",
            duration="10m",
            feature_window=8,
            horizon_minutes=10,
            min_samples=20,
            epochs=1,
        ),
        frame_loader=lambda *_args: frame,
    )

    assert f"{SIM_FEEDBACK_PREFIX}win_rate" in dataset.feature_columns
    assert f"{SIM_FEEDBACK_PREFIX}family_lstm_win_rate" in dataset.feature_columns


def _labeled_frame(entry_open_times: list[int]) -> pd.DataFrame:
    rows = []
    for index, entry_open_time in enumerate(entry_open_times):
        rows.append(
            {
                "open_time": entry_open_time - 600_000,
                "entry_open_time": entry_open_time,
                "close": 100.0 + index,
                "ret_1": 0.01,
                "label_up": 1.0,
                "future_return": 0.01,
            }
        )
    return pd.DataFrame(rows)


def _training_frame() -> pd.DataFrame:
    rows = []
    for index in range(40):
        rows.append(
            {
                "open_time": index * 600_000,
                "entry_open_time": (index + 1) * 600_000,
                "open": 100.0 + index * 0.1,
                "high": 100.5 + index * 0.1,
                "low": 99.5 + index * 0.1,
                "close": 100.0 + index * 0.1,
                "volume": 1000.0 + index,
                "ret_1": 0.001,
                "vol_std_20": 0.02,
                "factor_learning_top_weight_sum": 0.5,
                "lstm_regime_trend_score": 0.1,
            }
        )
    return pd.DataFrame(rows)


def _combo_feature_snapshots(*_args) -> list[dict]:
    return [
        {
            "entryOpenTime": (index + 1) * 600_000,
            "ranking": [
                {
                    "factorName": "goal_combo__alpha",
                    "direction": "up",
                    "factorScore": 88.0,
                    "winRate": 0.7,
                    "profitFactor": 2.0,
                    "ir": 1.0,
                    "totalPeriods": 80,
                    "members": [{"name": "factor_a", "orientation": 1}],
                }
            ],
        }
        for index in range(40)
    ]
