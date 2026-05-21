from __future__ import annotations

from typing import Any

import numpy as np

from app.services.lstm_lifecycle import (
    LSTM_STATUS_INITIAL_BASELINE,
    LSTM_STATUS_VALIDATION_FAILED,
    candidate_status,
    promotion_reason,
)


def initial_baseline_report(report: dict[str, Any], publish_initial_baseline: bool) -> dict[str, Any]:
    if not publish_initial_baseline or report["status"] != LSTM_STATUS_VALIDATION_FAILED:
        return report
    status = LSTM_STATUS_INITIAL_BASELINE
    return {
        **report,
        "status": status,
        "candidateStatus": candidate_status(status),
        "promotionReason": promotion_reason(status, report.get("validationGate") or {}),
    }


def return_stats(returns: np.ndarray) -> dict[str, float]:
    up = returns[returns > 0]
    down = returns[returns <= 0]
    return {"mean": _mean(returns), "upMean": _mean(up), "downMean": _mean(down)}


def _mean(values: np.ndarray) -> float:
    return 0.0 if len(values) == 0 else float(np.mean(values))
