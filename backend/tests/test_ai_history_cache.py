from __future__ import annotations

import pytest

from app.services import ai_history_cache
from app.services.ai_history_cache import AiHistoryWarmupError


def test_warm_ai_history_cache_raises_symbol_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.services.event_ai_history.query_ai_history_meta",
        lambda _conn, symbol: (_ for _ in ()).throw(RuntimeError("meta failed")) if symbol == "ETHUSDT" else {},
    )
    monkeypatch.setattr("app.services.event_ai_history.query_ai_history_success", lambda *_args, **_kwargs: {})

    with pytest.raises(AiHistoryWarmupError) as raised:
        ai_history_cache.warm_ai_history_cache(object(), symbols=("BTCUSDT", "ETHUSDT"))

    assert str(raised.value) == "AI history cache warm-up failed for: ETHUSDT"
    assert raised.value.details == {
        "failures": [{"symbol": "ETHUSDT", "error": "meta failed", "exceptionType": "RuntimeError"}]
    }
