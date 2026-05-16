from __future__ import annotations

from app.services import lstm_combo_ranking


def test_resolve_lstm_combo_ranking_prefers_primary_when_usable(monkeypatch) -> None:
    primary = _ranking("combo_primary")
    high = _ranking("combo_high")

    monkeypatch.setattr(lstm_combo_ranking, "get_cached_combination_ranking", lambda *_args: primary)
    monkeypatch.setattr(lstm_combo_ranking, "get_cached_high_winrate_combo_ranking", lambda *_args: high)

    result = lstm_combo_ranking.resolve_lstm_combo_ranking("BTCUSDT", "10m")

    assert result["ranking"][0]["factorName"] == "combo_primary"
    assert result["lstmComboRankingSource"] == lstm_combo_ranking.LSTM_COMBO_SOURCE_PRIMARY


def test_resolve_lstm_combo_ranking_falls_back_to_high_winrate_when_primary_empty(monkeypatch) -> None:
    primary = {**_ranking("combo_primary"), "ranking": []}
    high = _ranking("combo_high")

    monkeypatch.setattr(lstm_combo_ranking, "get_cached_combination_ranking", lambda *_args: primary)
    monkeypatch.setattr(lstm_combo_ranking, "get_cached_high_winrate_combo_ranking", lambda *_args: high)

    result = lstm_combo_ranking.resolve_lstm_combo_ranking("BTCUSDT", "10m")

    assert result["ranking"][0]["factorName"] == "combo_high"
    assert result["lstmComboRankingSource"] == lstm_combo_ranking.LSTM_COMBO_SOURCE_HIGH_WINRATE
    assert result["lstmComboRankingReason"] == "primary_empty;high_winrate_ready"


def _ranking(factor_name: str) -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "ranking": [{"factorName": factor_name, "members": [{"name": "factor_a"}]}],
    }
