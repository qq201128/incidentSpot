"""Connection helpers for Binance futures WebSocket upstreams."""

from __future__ import annotations

import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from app.services.binance_proxy_tunnel import (
    open_proxy_tunnel,
    proxy_config_for_stream,
)

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[str, float]] = {}
_DEFAULT_CACHE_TTL = 300.0
_DEFAULT_WS_PORT = 443


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def resolve_hostname_a_via_doh(hostname: str, timeout: float = 6.0) -> str | None:
    url = f"https://cloudflare-dns.com/dns-query?name={hostname}&type=A"
    req = urllib.request.Request(url, headers={"Accept": "application/dns-json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("DoH resolve failed for %s: %s", hostname, exc)
        return None
    answers = [a for a in payload.get("Answer", []) if a.get("type") == 1 and a.get("data")]
    if not answers:
        logger.warning("DoH returned no A records for %s", hostname)
        return None
    return str(random.choice(answers)["data"]).strip()


def upstream_websocket_connect_kwargs(stream_url: str) -> dict[str, object]:
    """Extra kwargs for ``websockets.legacy.client.connect``."""
    parsed = urlparse(stream_url)
    hostname = parsed.hostname
    if not hostname:
        return {}
    port = parsed.port or _DEFAULT_WS_PORT

    proxy = proxy_config_for_stream()
    if proxy:
        sock = open_proxy_tunnel(proxy, hostname, port)
        logger.info(
            "Binance stream proxy tunnel: %s:%s via %s://%s:%s",
            hostname,
            port,
            proxy.scheme,
            proxy.host,
            proxy.port,
        )
        return {"sock": sock, "server_hostname": hostname}

    manual_ip = os.environ.get("BINANCE_FSTREAM_CONNECT_IP", "").strip()
    if manual_ip:
        port = int(os.environ.get("BINANCE_FSTREAM_CONNECT_PORT", str(_DEFAULT_WS_PORT)))
        logger.info("Binance stream TCP override: %s -> %s:%s", hostname, manual_ip, port)
        return {"host": manual_ip, "port": port, "server_hostname": hostname}

    if not _env_truthy("BINANCE_STREAM_CONNECT_VIA_DOH"):
        return {}

    now = time.monotonic()
    cached = _CACHE.get(hostname)
    if cached and now < cached[1]:
        ip = cached[0]
    else:
        ip = resolve_hostname_a_via_doh(hostname)
        if not ip:
            return {}
        ttl = float(os.environ.get("BINANCE_STREAM_DOH_CACHE_TTL", str(_DEFAULT_CACHE_TTL)))
        _CACHE[hostname] = (ip, now + ttl)
        logger.info("Binance stream DoH: %s -> %s (cache %.0fs)", hostname, ip, ttl)

    return {"host": ip, "port": port, "server_hostname": hostname}
