from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.factor_learning_common import round_metric

MIN_ACTIVE_SAMPLES = 5
DIVERSITY_SCALE = 12
LOSS_PATTERN_SCALE = 5
WARNING_ACCURACY = 0.45
SAMPLE_WEIGHT = 0.38
ACCURACY_WEIGHT = 0.32
DIVERSITY_WEIGHT = 0.18
LOSS_PRESSURE_WEIGHT = 0.12
DURATION_BASE_WEIGHT = 0.65
DURATION_ACCURACY_WEIGHT = 0.35
UNKNOWN_DURATION_WEIGHT = 0.8
DEFAULT_DURATION_WEIGHTS = {"10m": 1.0, "30m": 0.92, "60m": 0.84, "1d": 0.72}


@dataclass(frozen=True)
class AlgorithmProfile:
    key: str
    label: str
    family: str
    sample_floor: int
    duration_bias: dict[str, float]


ALGORITHM_PROFILES = (
    AlgorithmProfile("lstm", "LSTM", "sequence", 30, {"10m": 1.0, "30m": 0.96, "60m": 0.90, "1d": 0.82}),
    AlgorithmProfile("gru", "GRU", "sequence", 24, {"10m": 1.0, "30m": 0.98, "60m": 0.90, "1d": 0.80}),
    AlgorithmProfile("cnn", "CNN", "pattern", 20, {"10m": 0.96, "30m": 1.0, "60m": 0.92, "1d": 0.80}),
    AlgorithmProfile("transformer", "Transformer", "attention", 40, {"10m": 0.96, "30m": 1.0, "60m": 0.96, "1d": 0.88}),
    AlgorithmProfile("random_forest", "随机森林", "tree", 10, {"10m": 0.94, "30m": 1.0, "60m": 1.0, "1d": 0.90}),
    AlgorithmProfile("extra_trees", "ExtraTrees", "tree", 10, {"10m": 0.94, "30m": 1.0, "60m": 1.0, "1d": 0.90}),
    AlgorithmProfile("xgboost", "XGBoost", "tree", 14, {"10m": 0.96, "30m": 1.0, "60m": 0.96, "1d": 0.88}),
    AlgorithmProfile("lightgbm", "LightGBM", "tree", 14, {"10m": 0.96, "30m": 1.0, "60m": 0.96, "1d": 0.88}),
    AlgorithmProfile("catboost", "CatBoost", "tree", 14, {"10m": 0.96, "30m": 1.0, "60m": 0.96, "1d": 0.88}),
    AlgorithmProfile("logistic_elasticnet", "LogisticElasticNet", "linear", 12, {"10m": 0.90, "30m": 0.96, "60m": 1.0, "1d": 0.92}),
    AlgorithmProfile("svm", "SVM", "margin", 16, {"10m": 0.90, "30m": 0.96, "60m": 1.0, "1d": 0.92}),
    AlgorithmProfile("reinforcement", "强化学习策略", "policy", 35, {"10m": 0.96, "30m": 1.0, "60m": 0.94, "1d": 0.86}),
    AlgorithmProfile("bayesian", "贝叶斯", "probabilistic", 8, {"10m": 1.0, "30m": 0.96, "60m": 0.92, "1d": 0.84}),
    AlgorithmProfile("knn", "KNN", "neighbor", 8, {"10m": 0.92, "30m": 1.0, "60m": 0.94, "1d": 0.82}),
    AlgorithmProfile("ensemble", "综合", "ensemble", 5, {"10m": 1.0, "30m": 1.0, "60m": 1.0, "1d": 0.94}),
)


def adaptive_learning_summary(
    rows: list[dict[str, Any]],
    settled_predictions: list[dict[str, Any]],
    *,
    duration: str,
    loss_patterns: list[dict[str, Any]],
    monitoring_report: dict[str, Any] | None,
    lstm_shadow: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = _learning_metrics(rows, settled_predictions, loss_patterns, monitoring_report)
    algorithms = _algorithm_payloads(metrics, duration, lstm_shadow)
    active = [item for item in algorithms if item["status"] == "active"]
    return {
        "status": _summary_status(metrics),
        "overallAccuracy": metrics["accuracy"],
        "sampleCount": metrics["sampleCount"],
        "activeAlgorithmCount": len(active),
        "algorithmCount": len(algorithms),
        "algorithms": algorithms,
        "insight": _insight(metrics, duration, active),
        "durationPreference": _duration_preference(duration, metrics),
    }


def _learning_metrics(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    loss_patterns: list[dict[str, Any]],
    monitoring: dict[str, Any] | None,
) -> dict[str, Any]:
    accuracy = _prediction_accuracy(predictions)
    if accuracy is None:
        accuracy = _monitoring_accuracy(monitoring)
    return {
        "accuracy": accuracy,
        "sampleCount": len(predictions),
        "factorCount": len(rows),
        "lossPatternCount": len(loss_patterns),
        "qualityPassRate": _monitoring_metric(monitoring, "qualityPassRate"),
    }


def _algorithm_payloads(
    metrics: dict[str, Any],
    duration: str,
    lstm_shadow: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    scored = [_algorithm_score(profile, metrics, duration, lstm_shadow) for profile in ALGORITHM_PROFILES]
    active_total = sum(item["rawScore"] for item in scored if item["status"] == "active")
    return [_normalized_algorithm(item, active_total) for item in scored]


def _algorithm_score(
    profile: AlgorithmProfile,
    metrics: dict[str, Any],
    duration: str,
    lstm_shadow: dict[str, Any] | None,
) -> dict[str, Any]:
    if profile.key == "lstm" and lstm_shadow:
        return _lstm_algorithm_score(profile, lstm_shadow, duration)
    sample_ratio = min(metrics["sampleCount"] / profile.sample_floor, 1.0)
    active = metrics["sampleCount"] >= max(MIN_ACTIVE_SAMPLES, profile.sample_floor // 2)
    raw = _raw_score(profile, metrics, duration, sample_ratio) if active else 0.0
    return {
        "key": profile.key,
        "label": profile.label,
        "family": profile.family,
        "sampleFloor": profile.sample_floor,
        "status": "active" if active else "waiting_for_samples",
        "rawScore": raw,
    }


def _lstm_algorithm_score(
    profile: AlgorithmProfile,
    lstm_shadow: dict[str, Any],
    duration: str,
) -> dict[str, Any]:
    sample_count = int(lstm_shadow.get("sampleCount") or 0)
    win_rate = lstm_shadow.get("winRate")
    sample_ratio = min(sample_count / profile.sample_floor, 1.0)
    active = sample_count >= max(MIN_ACTIVE_SAMPLES, profile.sample_floor // 2)
    accuracy = float(win_rate) if win_rate is not None else 0.5
    duration_bias = profile.duration_bias.get(duration, DEFAULT_DURATION_WEIGHTS.get(duration, UNKNOWN_DURATION_WEIGHT))
    raw = (SAMPLE_WEIGHT * sample_ratio + ACCURACY_WEIGHT * accuracy) * duration_bias if active else 0.0
    return {
        "key": profile.key,
        "label": profile.label,
        "family": profile.family,
        "sampleFloor": profile.sample_floor,
        "status": "active" if active else "waiting_for_lstm_shadow_samples",
        "rawScore": raw,
    }


def _raw_score(
    profile: AlgorithmProfile,
    metrics: dict[str, Any],
    duration: str,
    sample_ratio: float,
) -> float:
    accuracy = metrics["accuracy"] if metrics["accuracy"] is not None else 0.5
    diversity = min(metrics["factorCount"] / DIVERSITY_SCALE, 1.0)
    loss_pressure = min(metrics["lossPatternCount"] / LOSS_PATTERN_SCALE, 1.0)
    duration_bias = profile.duration_bias.get(duration, DEFAULT_DURATION_WEIGHTS.get(duration, UNKNOWN_DURATION_WEIGHT))
    return (
        SAMPLE_WEIGHT * sample_ratio
        + ACCURACY_WEIGHT * accuracy
        + DIVERSITY_WEIGHT * diversity
        + LOSS_PRESSURE_WEIGHT * (1.0 - loss_pressure)
    ) * duration_bias


def _normalized_algorithm(item: dict[str, Any], active_total: float) -> dict[str, Any]:
    weight = item["rawScore"] / active_total if active_total > 0 and item["status"] == "active" else 0.0
    return {**item, "weight": round_metric(weight, 4), "score": round_metric(item["rawScore"], 4)}


def _summary_status(metrics: dict[str, Any]) -> str:
    if metrics["sampleCount"] < MIN_ACTIVE_SAMPLES:
        return "insufficient_data"
    if metrics["accuracy"] is not None and metrics["accuracy"] < WARNING_ACCURACY:
        return "warning"
    return "learning"


def _insight(metrics: dict[str, Any], duration: str, active: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(active, key=lambda item: item["weight"], default=None)
    return {
        "bestAlgorithm": best["label"] if best else None,
        "message": _insight_message(metrics, duration, best),
    }


def _insight_message(metrics: dict[str, Any], duration: str, best: dict[str, Any] | None) -> str:
    if metrics["sampleCount"] < MIN_ACTIVE_SAMPLES:
        return "等待更多已结算样本后再启用算法权重。"
    if best is None:
        return "暂无可用算法权重。"
    accuracy = metrics["accuracy"]
    acc_text = f"{round_metric(accuracy * 100, 1)}%" if accuracy is not None else "未知"
    return f"{duration} 当前样本准确率 {acc_text}，优先使用 {best['label']}。"


def _duration_preference(duration: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    base = DEFAULT_DURATION_WEIGHTS.get(duration, UNKNOWN_DURATION_WEIGHT)
    accuracy = metrics["accuracy"] if metrics["accuracy"] is not None else 0.0
    score = base * (DURATION_BASE_WEIGHT + DURATION_ACCURACY_WEIGHT * accuracy)
    return [{"duration": duration, "score": round_metric(score, 4), "sampleCount": metrics["sampleCount"]}]


def _prediction_accuracy(predictions: list[dict[str, Any]]) -> float | None:
    if not predictions:
        return None
    wins = sum(1 for item in predictions if bool(item.get("prediction_correct")))
    return round_metric(wins / len(predictions), 4)


def _monitoring_accuracy(monitoring: dict[str, Any] | None) -> float | None:
    return _monitoring_metric(monitoring, "predictionSuccessRate")


def _monitoring_metric(monitoring: dict[str, Any] | None, key: str) -> float | None:
    metrics = monitoring.get("metrics") if isinstance(monitoring, dict) else None
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    return float(value) if value is not None else None
