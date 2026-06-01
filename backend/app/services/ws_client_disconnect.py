from __future__ import annotations

from fastapi import WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

try:
    from uvicorn.protocols.utils import ClientDisconnected as UvicornClientDisconnected
except ImportError:
    UvicornClientDisconnected = None


class ClientDisconnected(OSError):
    """ASGI transport closed before a WebSocket send completed."""


CLIENT_WS_GONE_EXC = tuple(
    exc_type
    for exc_type in (
        WebSocketDisconnect,
        ConnectionClosed,
        ClientDisconnected,
        UvicornClientDisconnected,
    )
    if exc_type is not None
)
