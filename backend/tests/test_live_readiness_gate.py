from __future__ import annotations

from app.services.high_winrate_strategy_metrics import high_winrate_metrics
from app.services.high_winrate_strategy_status_store import status_payload
from app.services.live_readiness_gate import live_readiness_gate


def test_live_readiness_gate_accepts_stable_metrics() -> None:
    metrics = _stable_metrics()

    gate = live_readiness_gate(metrics, "paper_stable")

    assert gate["eligible"] is True
    assert gate["status"] == "eligible"
    assert gate["reason"] == "passed"
    assert gate["reasons"] == []
    assert gate["manualEnableRequired"] is True
    assert gate["realTradingEnabled"] is False
    assert gate["thresholds"]["minTotalEventPnlU"] == 0.0
    assert gate["decision"]["status"] == "paper_stable"


def test_live_readiness_gate_blocks_when_total_event_pnl_is_not_positive() -> None:
    metrics = _stable_metrics()
    metrics["totalEventPnlU"] = 0.0

    gate = live_readiness_gate(metrics, "paper_stable")

    assert gate["eligible"] is False
    assert gate["reason"] == "paper_live_total_pnl_below_target"
    assert "paper_live_total_pnl_below_target" in gate["reasons"]


def test_live_readiness_gate_blocks_when_event_pnl_is_missing() -> None:
    metrics = _stable_metrics()
    metrics["totalEventPnlU"] = None

    gate = live_readiness_gate(metrics, "paper_stable")

    assert gate["eligible"] is False
    assert gate["reason"] == "paper_live_total_pnl_missing"
    assert gate["thresholds"]["requiresTotalEventPnlU"] is True


def test_live_readiness_gate_blocks_when_status_is_not_stable() -> None:
    metrics = _stable_metrics()

    gate = live_readiness_gate(metrics, "paper_collecting")

    assert gate["eligible"] is False
    assert gate["reason"] == "paper_live_status_not_stable"
    assert gate["reasons"] == ["paper_live_status_not_stable"]
    assert gate["decision"]["reason"] == "stable_paper_live_target_met"


def test_live_readiness_gate_blocks_policy_override_requests() -> None:
    metrics = _stable_metrics()

    gate = live_readiness_gate(metrics, "paper_stable", real_trading_enabled=True)

    assert gate["eligible"] is False
    assert gate["reason"] == "real_trading_disabled_by_project_policy"
    assert gate["realTradingEnabled"] is False


def test_status_payload_exposes_live_readiness() -> None:
    metrics = _stable_metrics()

    payload = status_payload("paper_stable", "stable_paper_live_target_met", metrics)

    assert payload["liveReadiness"]["eligible"] is True
    assert payload["liveReadiness"]["paperLiveStatus"] == "paper_stable"
    assert payload["realTradingEnabled"] is False


def test_status_payload_blocks_prediction_only_metrics_before_live_review() -> None:
    metrics = _stable_prediction_only_metrics()

    payload = status_payload("paper_stable", "stable_paper_live_target_met", metrics)

    assert payload["liveReadiness"]["eligible"] is False
    assert payload["liveReadiness"]["reason"] == "paper_live_total_pnl_missing"
    assert payload["realTradingEnabled"] is False


def _stable_metrics() -> dict:
    values = ([0.8] * 6 + [-1.0] * 4) * 3
    rows = [{"actual_return": value, "event_pnl": value} for value in values]
    return high_winrate_metrics(rows)


def _stable_prediction_only_metrics() -> dict:
    values = ([0.8] * 6 + [-1.0] * 4) * 3
    rows = [{"actual_return": value, "prediction_correct": int(value > 0)} for value in values]
    return high_winrate_metrics(rows)
