from __future__ import annotations

from app.services.prediction_policy import trade_policy_payload


def rule_prediction_response(result: dict) -> dict:
    return {
        "symbol": result["symbol"],
        "signalKey": result.get("signal_key") or result.get("strategy_key"),
        "strategyKey": result.get("strategy_key"),
        "duration": result["duration"],
        "direction": result["direction"],
        "probabilityUp": result["probability_up"],
        "confidence": result["confidence"],
        "certaintyLabel": result["certainty_label"],
        "threshold": result["threshold"],
        "tradeQualityScore": result.get("trade_quality_score"),
        "tradeQualityPassed": result.get("trade_quality_passed"),
        "tradeQualityGate": result.get("trade_quality_gate"),
        "highWinrateGate": result.get("high_winrate_gate"),
        "highWinrateGatePassed": result.get("high_winrate_gate_passed"),
        "highWinrateGateValue": result.get("high_winrate_gate_value"),
        "signalSource": result.get("signal_source"),
        "ruleScore": result.get("rule_score"),
        "ruleReasons": result.get("rule_reasons"),
        "orderbook": result.get("orderbook"),
        "timeframeVotes": result.get("timeframe_votes"),
        **trade_policy_payload(result["duration"], strategy_key=result.get("strategy_key")),
    }
