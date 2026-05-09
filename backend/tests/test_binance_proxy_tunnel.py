from __future__ import annotations

import socket
import threading

from app.services import binance_proxy_tunnel as proxy_tunnel
from app.services.binance_upstream_connect import upstream_websocket_connect_kwargs

STREAM_URL = "wss://fstream.binance.com/market/ws/btcusdt@markPrice@1s"
TARGET_HOST = "fstream.binance.com"
TARGET_PORT = 443
HTTP_OK = b"HTTP/1.1 200 Connection Established\r\n\r\n"
LOCALHOST = "127.0.0.1"
RECV_CHUNK_BYTES = 1024
WINDOWS_HTTPS_PROXY_PORT = 7891
PROXY_ENV_NAMES = (
    "BINANCE_WS_PROXY",
    "BINANCE_FSTREAM_PROXY",
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
)


def test_upstream_connect_uses_http_proxy_tunnel(monkeypatch) -> None:
    _clear_proxy_env(monkeypatch)
    requests: list[str] = []
    listener = _listen_local()
    thread = threading.Thread(
        target=_serve_http_connect,
        args=(listener, requests),
        daemon=True,
    )
    thread.start()
    port = listener.getsockname()[1]
    monkeypatch.setenv("BINANCE_WS_PROXY", f"http://{LOCALHOST}:{port}")

    kwargs = upstream_websocket_connect_kwargs(STREAM_URL)
    sock = kwargs["sock"]
    try:
        assert kwargs["server_hostname"] == TARGET_HOST
        assert isinstance(sock, socket.socket)
    finally:
        sock.close()
        listener.close()
        thread.join(timeout=1)

    assert requests
    assert requests[0].startswith(f"CONNECT {TARGET_HOST}:{TARGET_PORT} HTTP/1.1")
    assert f"Host: {TARGET_HOST}:{TARGET_PORT}" in requests[0]


def test_direct_proxy_value_disables_proxy(monkeypatch) -> None:
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("BINANCE_WS_PROXY", "direct")

    assert proxy_tunnel.proxy_config_for_stream() is None


def test_windows_proxy_server_prefers_https_entry() -> None:
    value = proxy_tunnel._select_windows_proxy_entry(
        "http=127.0.0.1:7890;https=127.0.0.1:7891;socks=127.0.0.1:7892"
    )
    config = proxy_tunnel._parse_proxy_config(value, "http")

    assert config.scheme == "http"
    assert config.host == LOCALHOST
    assert config.port == WINDOWS_HTTPS_PROXY_PORT


def _clear_proxy_env(monkeypatch) -> None:
    for name in PROXY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


def _listen_local() -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind((LOCALHOST, 0))
    listener.listen(1)
    return listener


def _serve_http_connect(listener: socket.socket, requests: list[str]) -> None:
    conn, _ = listener.accept()
    with conn:
        data = b""
        while b"\r\n\r\n" not in data:
            data += conn.recv(RECV_CHUNK_BYTES)
        requests.append(data.decode("ascii"))
        conn.sendall(HTTP_OK)
        conn.recv(1)
