from __future__ import annotations

import base64
import os
import socket
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

_DEFAULT_PROXY_TIMEOUT_SECONDS = 12.0
_HTTP_PROXY_MIN_OK_STATUS = 200
_HTTP_PROXY_MAX_OK_STATUS = 299
_MAX_HTTP_PROXY_HEADER_BYTES = 8192
_HTTP_HEADER_CHUNK_BYTES = 1
_STATUS_LINE_MIN_PARTS = 2
_PORT_BYTE_LENGTH = 2
_BYTE_ORDER = "big"
_SOCKS_VERSION = 5
_SOCKS_CONNECT_COMMAND = 1
_SOCKS_RESERVED = 0
_SOCKS_IPV4_ATYP = 1
_SOCKS_DOMAIN_ATYP = 3
_SOCKS_IPV6_ATYP = 4
_SOCKS_RESPONSE_PREFIX_BYTES = 4
_SOCKS_SUCCESS = 0
_SOCKS_NO_AUTH = 0
_SOCKS_USER_PASS_AUTH = 2
_SOCKS_NO_ACCEPTABLE_METHODS = 255
_SOCKS_AUTH_VERSION = 1
_SOCKS_AUTH_SUCCESS = 0
_SOCKS_DOMAIN_LENGTH_MAX = 255
_SOCKS_DOMAIN_LENGTH_BYTES = 1
_IPV4_BYTES = 4
_IPV6_BYTES = 16
_PROXY_ENV_NAMES = (
    "BINANCE_WS_PROXY",
    "BINANCE_FSTREAM_PROXY",
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
)
_DIRECT_PROXY_VALUES = {"direct", "none", "off", "false", "0"}
_FALSEY_VALUES = {"0", "false", "no", "off"}
_WINDOWS_PROXY_REGISTRY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"


@dataclass(frozen=True)
class ProxyConfig:
    scheme: str
    host: str
    port: int
    username: str | None = None
    password: str | None = None


def proxy_config_for_stream() -> ProxyConfig | None:
    explicit = _explicit_proxy_value()
    if explicit is not None:
        if _is_direct_proxy_value(explicit):
            return None
        return _parse_proxy_config(explicit, "http")
    windows_value = _windows_proxy_value()
    if windows_value is None:
        return None
    return _parse_proxy_config(windows_value, "http")

def open_proxy_tunnel(
    proxy: ProxyConfig,
    target_host: str,
    target_port: int,
) -> socket.socket:
    if proxy.scheme == "http":
        return _open_http_proxy_tunnel(proxy, target_host, target_port)
    return _open_socks5_proxy_tunnel(proxy, target_host, target_port)

def _explicit_proxy_value() -> str | None:
    for name in _PROXY_ENV_NAMES:
        value = os.environ.get(name) or os.environ.get(name.lower())
        if value is not None and value.strip():
            return value.strip()
    return None

def _is_direct_proxy_value(value: str) -> bool:
    return value.strip().lower() in _DIRECT_PROXY_VALUES

def _windows_proxy_value() -> str | None:
    if os.name != "nt" or not _use_windows_proxy():
        return None
    server = _read_windows_proxy_server()
    if not server:
        return None
    return _select_windows_proxy_entry(server)

def _use_windows_proxy() -> bool:
    raw = os.environ.get("BINANCE_WS_USE_WINDOWS_PROXY")
    return raw is None or raw.strip().lower() not in _FALSEY_VALUES

def _read_windows_proxy_server() -> str | None:
    try:
        import winreg
    except ImportError:
        return None
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _WINDOWS_PROXY_REGISTRY_PATH) as key:
        enabled = _registry_value(winreg, key, "ProxyEnable")
        server = _registry_value(winreg, key, "ProxyServer")
    if int(enabled or 0) != 1 or not server:
        return None
    return str(server).strip()

def _registry_value(winreg_module: object, key: object, name: str) -> object | None:
    try:
        return winreg_module.QueryValueEx(key, name)[0]
    except OSError:
        return None

def _select_windows_proxy_entry(server: str) -> str | None:
    entries = [part.strip() for part in server.split(";") if part.strip()]
    default = None
    keyed: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            default = entry
            continue
        key, value = entry.split("=", maxsplit=1)
        keyed[key.strip().lower()] = value.strip()
    value = keyed.get("https") or keyed.get("http") or keyed.get("socks") or default
    if value is None:
        return None
    scheme = "socks5" if value == keyed.get("socks") else "http"
    return _ensure_proxy_scheme(value, scheme)

def _parse_proxy_config(raw: str, default_scheme: str) -> ProxyConfig:
    parsed = urlparse(_ensure_proxy_scheme(raw, default_scheme))
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "socks5"}:
        raise ValueError(f"unsupported Binance WS proxy scheme: {scheme}")
    if not parsed.hostname or not parsed.port:
        raise ValueError(f"invalid Binance WS proxy: {raw}")
    return ProxyConfig(
        scheme=scheme,
        host=parsed.hostname,
        port=parsed.port,
        username=unquote(parsed.username) if parsed.username else None,
        password=unquote(parsed.password) if parsed.password else None,
    )


def _ensure_proxy_scheme(value: str, default_scheme: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme:
        return value
    return f"{default_scheme}://{value}"


def _open_http_proxy_tunnel(
    proxy: ProxyConfig,
    target_host: str,
    target_port: int,
) -> socket.socket:
    sock = socket.create_connection(
        (proxy.host, proxy.port),
        timeout=_DEFAULT_PROXY_TIMEOUT_SECONDS,
    )
    try:
        _send_http_connect(sock, proxy, target_host, target_port)
        _verify_http_connect_response(sock)
        return sock
    except Exception:
        sock.close()
        raise


def _send_http_connect(
    sock: socket.socket,
    proxy: ProxyConfig,
    target_host: str,
    target_port: int,
) -> None:
    target = f"{target_host}:{target_port}"
    headers = [
        f"CONNECT {target} HTTP/1.1",
        f"Host: {target}",
        "Proxy-Connection: Keep-Alive",
    ]
    if proxy.username:
        credential = f"{proxy.username}:{proxy.password or ''}".encode()
        token = base64.b64encode(credential).decode()
        headers.append(f"Proxy-Authorization: Basic {token}")
    sock.sendall(("\r\n".join(headers) + "\r\n\r\n").encode("ascii"))


def _verify_http_connect_response(sock: socket.socket) -> None:
    header = _read_http_proxy_header(sock)
    status_line = header.split(b"\r\n", maxsplit=1)[0].decode("iso-8859-1")
    parts = status_line.split()
    if len(parts) < _STATUS_LINE_MIN_PARTS:
        raise ConnectionError(f"invalid proxy CONNECT response: {status_line}")
    status = int(parts[1])
    if not _HTTP_PROXY_MIN_OK_STATUS <= status <= _HTTP_PROXY_MAX_OK_STATUS:
        raise ConnectionError(f"proxy CONNECT failed: {status_line}")


def _read_http_proxy_header(sock: socket.socket) -> bytes:
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(_HTTP_HEADER_CHUNK_BYTES)
        if not chunk:
            raise ConnectionError("proxy closed before CONNECT response")
        data += chunk
        if len(data) > _MAX_HTTP_PROXY_HEADER_BYTES:
            raise ConnectionError("proxy CONNECT response header too large")
    return data


def _open_socks5_proxy_tunnel(
    proxy: ProxyConfig,
    target_host: str,
    target_port: int,
) -> socket.socket:
    sock = socket.create_connection(
        (proxy.host, proxy.port),
        timeout=_DEFAULT_PROXY_TIMEOUT_SECONDS,
    )
    try:
        _socks5_negotiate_auth(sock, proxy)
        _socks5_connect(sock, target_host, target_port)
        return sock
    except Exception:
        sock.close()
        raise


def _socks5_negotiate_auth(sock: socket.socket, proxy: ProxyConfig) -> None:
    methods = [_SOCKS_NO_AUTH]
    if proxy.username:
        methods.append(_SOCKS_USER_PASS_AUTH)
    sock.sendall(bytes([_SOCKS_VERSION, len(methods), *methods]))
    version, method = _recv_exact(sock, _PORT_BYTE_LENGTH)
    if version != _SOCKS_VERSION or method == _SOCKS_NO_ACCEPTABLE_METHODS:
        raise ConnectionError("SOCKS5 proxy rejected authentication methods")
    if method == _SOCKS_USER_PASS_AUTH:
        _socks5_authenticate(sock, proxy)


def _socks5_authenticate(sock: socket.socket, proxy: ProxyConfig) -> None:
    username = (proxy.username or "").encode()
    password = (proxy.password or "").encode()
    if len(username) > _SOCKS_DOMAIN_LENGTH_MAX or len(password) > _SOCKS_DOMAIN_LENGTH_MAX:
        raise ValueError("SOCKS5 proxy credentials are too long")
    payload = bytes([_SOCKS_AUTH_VERSION, len(username)])
    sock.sendall(payload + username + bytes([len(password)]) + password)
    version, status = _recv_exact(sock, _PORT_BYTE_LENGTH)
    if version != _SOCKS_AUTH_VERSION or status != _SOCKS_AUTH_SUCCESS:
        raise ConnectionError("SOCKS5 proxy authentication failed")


def _socks5_connect(sock: socket.socket, target_host: str, target_port: int) -> None:
    host_bytes = target_host.encode("idna")
    if len(host_bytes) > _SOCKS_DOMAIN_LENGTH_MAX:
        raise ValueError(f"SOCKS5 target hostname is too long: {target_host}")
    request = bytes([
        _SOCKS_VERSION,
        _SOCKS_CONNECT_COMMAND,
        _SOCKS_RESERVED,
        _SOCKS_DOMAIN_ATYP,
        len(host_bytes),
    ])
    port_bytes = target_port.to_bytes(_PORT_BYTE_LENGTH, _BYTE_ORDER)
    sock.sendall(request + host_bytes + port_bytes)
    version, status, _, atyp = _recv_exact(sock, _SOCKS_RESPONSE_PREFIX_BYTES)
    if version != _SOCKS_VERSION or status != _SOCKS_SUCCESS:
        raise ConnectionError(f"SOCKS5 CONNECT failed with status {status}")
    _discard_socks5_bind_address(sock, atyp)


def _discard_socks5_bind_address(sock: socket.socket, atyp: int) -> None:
    if atyp == _SOCKS_IPV4_ATYP:
        _recv_exact(sock, _IPV4_BYTES)
    elif atyp == _SOCKS_IPV6_ATYP:
        _recv_exact(sock, _IPV6_BYTES)
    elif atyp == _SOCKS_DOMAIN_ATYP:
        length = _recv_exact(sock, _SOCKS_DOMAIN_LENGTH_BYTES)[0]
        _recv_exact(sock, length)
    else:
        raise ConnectionError(f"SOCKS5 proxy returned unsupported address type {atyp}")
    _recv_exact(sock, _PORT_BYTE_LENGTH)


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionError("proxy closed while reading handshake")
        data += chunk
    return data
