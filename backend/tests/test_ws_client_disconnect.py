from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from uvicorn.protocols.utils import ClientDisconnected

from app.services.index_kline_fallback import send_index_rest_fallback
from app.services.ws_client_disconnect import CLIENT_WS_GONE_EXC


def test_client_disconnected_is_oserror_subclass() -> None:
    assert issubclass(ClientDisconnected, OSError)


def test_client_ws_gone_includes_client_disconnected() -> None:
    assert ClientDisconnected in CLIENT_WS_GONE_EXC


def test_index_rest_fallback_reraises_client_disconnected() -> None:
    async def run() -> None:
        client_ws = AsyncMock()
        client_ws.send_json.side_effect = ClientDisconnected()
        row = {
            "openTime": 1,
            "open": "1",
            "high": "1",
            "low": "1",
            "close": "1",
            "volume": "0",
            "closeTime": 2,
            "isClosed": False,
        }
        with patch(
            "app.services.index_kline_fallback.fetch_index_price_klines",
            return_value=[row],
        ):
            with pytest.raises(ClientDisconnected):
                await send_index_rest_fallback(client_ws, "BTCUSDT", "10m", "test")

    asyncio.run(run())
