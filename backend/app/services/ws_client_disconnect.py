from __future__ import annotations

from fastapi import WebSocketDisconnect
from uvicorn.protocols.utils import ClientDisconnected
from websockets.exceptions import ConnectionClosed

CLIENT_WS_GONE_EXC = (WebSocketDisconnect, ConnectionClosed, ClientDisconnected)
