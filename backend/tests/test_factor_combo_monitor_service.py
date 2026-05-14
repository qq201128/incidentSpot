from __future__ import annotations

from app.services import factor_combo_monitor_service as monitor
from app.services.factor_combo_simulation_keys import factor_combo_simulation_strategy_keys

LOW_SAMPLE_COUNT = 10


def test_monitor_reads_all_top_three_simulation_strategy_keys(monkeypatch) -> None:
    captured = []

    class Rows:
        def fetchall(self) -> list[dict]:
            return [_loss_row(index) for index in range(LOW_SAMPLE_COUNT)]

    class Conn:
        def execute(self, sql: str, params: tuple) -> Rows:
            captured.append((sql, params))
            return Rows()

        def close(self) -> None:
            return None

    monkeypatch.setattr(monitor, "get_conn", lambda: Conn())

    report = monitor.factor_combo_monitor_report("BTCUSDT", "10m")

    assert captured[0][1][:3] == factor_combo_simulation_strategy_keys()
    assert report["status"] == "warning"
    assert report["metrics"]["predictionSuccessRate"] == 0.0
    assert report["solutions"][0]["text"]
    assert report["solutions"][0]["action"] == "refresh_learning"
    assert report["solutions"][0]["requiresConfirmation"] is True


def _loss_row(index: int) -> dict:
    return {
        "open_time": index,
        "direction": "up",
        "confidence": 0.55,
        "trade_quality_score": 0.55,
        "trade_quality_passed": 1,
        "actual_return": -0.01,
        "prediction_correct": 0,
        "high_winrate_rule": f"combo_{index}",
        "strategy_key": factor_combo_simulation_strategy_keys()[index % 3],
    }
