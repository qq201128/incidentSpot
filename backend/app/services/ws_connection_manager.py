"""
WebSocket 连接管理器 - 自动重连、心跳检测、连接池管理
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from websockets.legacy.client import WebSocketClientProtocol, connect as upstream_ws_connect
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


@dataclass
class ConnectionConfig:
    """WebSocket连接配置"""
    url: str
    ping_interval: float = 20.0
    ping_timeout: float = 20.0
    open_timeout: float = 20.0
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    max_reconnect_attempts: int = 0  # 0表示无限重试
    extra_headers: dict[str, str] = field(default_factory=dict)
    connect_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionStats:
    """连接统计"""
    connection_id: str
    state: ConnectionState
    connected_at: datetime | None = None
    last_message_at: datetime | None = None
    last_ping_at: datetime | None = None
    reconnect_count: int = 0
    message_count: int = 0
    error_count: int = 0


class WebSocketConnection:
    """
    单个WebSocket连接的管理器

    功能：
    - 自动重连（指数退避）
    - 心跳检测
    - 消息队列
    - 连接状态跟踪
    """

    def __init__(
        self,
        config: ConnectionConfig,
        on_message: Callable[[dict], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ):
        self.config = config
        self.on_message = on_message
        self.on_error = on_error

        self.connection_id = f"ws_{id(self)}"
        self._ws: WebSocketClientProtocol | None = None
        self._state = ConnectionState.DISCONNECTED
        self._retry_delay = config.initial_retry_delay
        self._reconnect_count = 0
        self._message_count = 0
        self._error_count = 0
        self._connected_at: datetime | None = None
        self._last_message_at: datetime | None = None
        self._last_ping_at: datetime | None = None
        self._stop_event = asyncio.Event()

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED and self._ws is not None

    @property
    def stats(self) -> ConnectionStats:
        return ConnectionStats(
            connection_id=self.connection_id,
            state=self._state,
            connected_at=self._connected_at,
            last_message_at=self._last_message_at,
            last_ping_at=self._last_ping_at,
            reconnect_count=self._reconnect_count,
            message_count=self._message_count,
            error_count=self._error_count,
        )

    async def connect(self) -> None:
        """建立连接"""
        if self._state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
            logger.warning(f"{self.connection_id} already connecting or connected")
            return

        self._state = ConnectionState.CONNECTING
        try:
            self._ws = await upstream_ws_connect(
                self.config.url,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                open_timeout=self.config.open_timeout,
                extra_headers=self.config.extra_headers,
                **self.config.connect_kwargs,
            )
            self._state = ConnectionState.CONNECTED
            self._connected_at = datetime.utcnow()
            self._retry_delay = self.config.initial_retry_delay
            logger.info(f"{self.connection_id} connected to {self.config.url}")

        except Exception as exc:
            self._state = ConnectionState.DISCONNECTED
            self._error_count += 1
            logger.error(f"{self.connection_id} connection failed: {exc}")
            raise

    async def disconnect(self) -> None:
        """主动断开连接"""
        self._state = ConnectionState.CLOSED
        if self._ws:
            try:
                await self._ws.close()
            except Exception as exc:
                logger.warning(f"{self.connection_id} close error: {exc}")
            finally:
                self._ws = None
        logger.info(f"{self.connection_id} disconnected")

    async def send(self, data: str | bytes | dict) -> None:
        """发送消息"""
        if not self.is_connected or not self._ws:
            raise RuntimeError(f"{self.connection_id} not connected")

        try:
            if isinstance(data, dict):
                import json
                data = json.dumps(data)
            await self._ws.send(data)
        except Exception as exc:
            self._error_count += 1
            logger.error(f"{self.connection_id} send failed: {exc}")
            raise

    async def receive(self) -> str | bytes:
        """接收消息"""
        if not self.is_connected or not self._ws:
            raise RuntimeError(f"{self.connection_id} not connected")

        try:
            message = await self._ws.recv()
            self._message_count += 1
            self._last_message_at = datetime.utcnow()
            return message
        except ConnectionClosed:
            self._state = ConnectionState.DISCONNECTED
            raise
        except Exception as exc:
            self._error_count += 1
            logger.error(f"{self.connection_id} receive failed: {exc}")
            raise

    async def ping(self) -> bool:
        """发送心跳"""
        if not self.is_connected or not self._ws:
            return False

        try:
            pong_waiter = await self._ws.ping()
            await asyncio.wait_for(pong_waiter, timeout=self.config.ping_timeout)
            self._last_ping_at = datetime.utcnow()
            return True
        except asyncio.TimeoutError:
            logger.warning(f"{self.connection_id} ping timeout")
            return False
        except Exception as exc:
            logger.warning(f"{self.connection_id} ping failed: {exc}")
            return False

    async def run_with_reconnect(self) -> None:
        """
        运行连接，带自动重连

        持续监听消息，断线后自动重连（指数退避）
        """
        while not self._stop_event.is_set():
            try:
                await self.connect()
                await self._message_loop()

            except ConnectionClosed as exc:
                logger.warning(f"{self.connection_id} connection closed: {exc}")
                await self._handle_reconnect()

            except asyncio.TimeoutError:
                logger.warning(f"{self.connection_id} connection timeout")
                await self._handle_reconnect()

            except Exception as exc:
                logger.exception(f"{self.connection_id} unexpected error: {exc}")
                if self.on_error:
                    try:
                        self.on_error(exc)
                    except Exception:
                        pass
                await self._handle_reconnect()

            finally:
                if self._ws:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                    self._ws = None

    async def _message_loop(self) -> None:
        """消息接收循环"""
        if not self._ws:
            return

        async for message in self._ws:
            self._message_count += 1
            self._last_message_at = datetime.utcnow()

            if self.on_message:
                try:
                    # 尝试解析为JSON
                    if isinstance(message, (str, bytes)):
                        import json
                        data = json.loads(message)
                    else:
                        data = message
                    self.on_message(data)
                except Exception as exc:
                    logger.warning(f"{self.connection_id} message handler error: {exc}")

    async def _handle_reconnect(self) -> None:
        """处理重连逻辑"""
        if self._state == ConnectionState.CLOSED:
            return

        # 检查重连次数限制
        if 0 < self.config.max_reconnect_attempts <= self._reconnect_count:
            logger.error(
                f"{self.connection_id} max reconnect attempts reached "
                f"({self.config.max_reconnect_attempts})"
            )
            self._state = ConnectionState.CLOSED
            return

        self._state = ConnectionState.RECONNECTING
        self._reconnect_count += 1

        logger.info(
            f"{self.connection_id} reconnecting in {self._retry_delay}s "
            f"(attempt {self._reconnect_count})"
        )

        await asyncio.sleep(self._retry_delay)

        # 指数退避
        self._retry_delay = min(
            self._retry_delay * 2,
            self.config.max_retry_delay
        )

    async def stop(self) -> None:
        """停止连接"""
        self._stop_event.set()
        await self.disconnect()


class WebSocketManager:
    """
    WebSocket连接管理器

    管理多个WebSocket连接，提供连接池、统一接口
    """

    def __init__(self):
        self._connections: dict[str, WebSocketConnection] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def register(
        self,
        key: str,
        config: ConnectionConfig,
        on_message: Callable[[dict], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> WebSocketConnection:
        """
        注册新连接

        Args:
            key: 连接唯一标识
            config: 连接配置
            on_message: 消息回调
            on_error: 错误回调

        Returns:
            WebSocketConnection
        """
        if key in self._connections:
            logger.warning(f"Connection {key} already registered")
            return self._connections[key]

        conn = WebSocketConnection(config, on_message, on_error)
        self._connections[key] = conn
        logger.info(f"Registered connection: {key} -> {config.url}")
        return conn

    async def start(self, key: str) -> None:
        """启动连接（带自动重连）"""
        conn = self._connections.get(key)
        if not conn:
            raise ValueError(f"Connection {key} not registered")

        if key in self._tasks:
            logger.warning(f"Connection {key} already started")
            return

        task = asyncio.create_task(conn.run_with_reconnect())
        self._tasks[key] = task
        logger.info(f"Started connection: {key}")

    async def stop(self, key: str) -> None:
        """停止连接"""
        conn = self._connections.get(key)
        if conn:
            await conn.stop()

        task = self._tasks.pop(key, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        logger.info(f"Stopped connection: {key}")

    async def stop_all(self) -> None:
        """停止所有连接"""
        keys = list(self._connections.keys())
        for key in keys:
            await self.stop(key)

    def get_connection(self, key: str) -> WebSocketConnection | None:
        """获取连接"""
        return self._connections.get(key)

    def get_stats(self, key: str) -> ConnectionStats | None:
        """获取连接统计"""
        conn = self._connections.get(key)
        return conn.stats if conn else None

    def get_all_stats(self) -> dict[str, ConnectionStats]:
        """获取所有连接统计"""
        return {key: conn.stats for key, conn in self._connections.items()}
