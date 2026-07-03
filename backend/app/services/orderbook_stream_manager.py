"""
订单簿WebSocket流服务 - 增量推送
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import websockets
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class OrderbookStreamManager:
    """
    订单簿流管理器

    功能：
    - 订阅Binance订单簿增量更新
    - 过滤只推送前N档
    - 支持多客户端订阅
    """

    def __init__(self):
        self.connections: dict[str, set[WebSocket]] = {}
        self.binance_ws: dict[str, websockets.WebSocketClientProtocol] = {}
        self.running = False

    async def subscribe(
        self,
        client_ws: WebSocket,
        symbol: str,
        depth: int = 5
    ):
        """
        订阅订单簿更新

        Args:
            client_ws: 客户端WebSocket
            symbol: 交易对
            depth: 深度档位数
        """
        symbol_lower = symbol.lower()

        # 添加客户端连接
        if symbol_lower not in self.connections:
            self.connections[symbol_lower] = set()
        self.connections[symbol_lower].add(client_ws)

        # 如果是第一个订阅该标的的客户端，启动上游连接
        if symbol_lower not in self.binance_ws:
            await self._start_binance_stream(symbol_lower)

        # 发送初始快照
        try:
            snapshot = await self._get_snapshot(symbol_lower, depth)
            await client_ws.send_json({
                "type": "snapshot",
                "data": snapshot
            })
        except Exception as e:
            logger.error(f"Failed to send snapshot: {e}")

    async def unsubscribe(self, client_ws: WebSocket, symbol: str):
        """取消订阅"""
        symbol_lower = symbol.lower()

        if symbol_lower in self.connections:
            self.connections[symbol_lower].discard(client_ws)

            # 如果没有客户端订阅了，关闭上游连接
            if not self.connections[symbol_lower]:
                await self._stop_binance_stream(symbol_lower)
                del self.connections[symbol_lower]

    async def _start_binance_stream(self, symbol: str):
        """启动Binance订单簿流"""
        stream_name = f"{symbol}@depth@100ms"
        uri = f"wss://fstream.binance.com/ws/{stream_name}"

        try:
            ws = await websockets.connect(uri)
            self.binance_ws[symbol] = ws

            # 启动接收任务
            asyncio.create_task(self._receive_binance_updates(symbol, ws))

            logger.info(f"Started Binance orderbook stream: {symbol}")
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")

    async def _stop_binance_stream(self, symbol: str):
        """停止Binance订单簿流"""
        if symbol in self.binance_ws:
            ws = self.binance_ws[symbol]
            await ws.close()
            del self.binance_ws[symbol]
            logger.info(f"Stopped Binance orderbook stream: {symbol}")

    async def _receive_binance_updates(
        self,
        symbol: str,
        ws: websockets.WebSocketClientProtocol
    ):
        """接收Binance更新并转发给客户端"""
        try:
            async for message in ws:
                data = json.loads(message)

                # 处理深度更新
                if "e" in data and data["e"] == "depthUpdate":
                    update = self._parse_depth_update(data)

                    # 广播给所有订阅的客户端
                    await self._broadcast_update(symbol, update)
        except Exception as e:
            logger.error(f"Binance stream error for {symbol}: {e}")

            # 重连
            await asyncio.sleep(1)
            if symbol in self.connections and self.connections[symbol]:
                await self._start_binance_stream(symbol)

    def _parse_depth_update(self, data: dict) -> dict:
        """解析Binance深度更新"""
        return {
            "bids": [[float(p), float(q)] for p, q in data.get("b", [])],
            "asks": [[float(p), float(q)] for p, q in data.get("a", [])],
            "timestamp": data.get("T"),
            "updateId": data.get("u"),
        }

    async def _broadcast_update(self, symbol: str, update: dict):
        """广播更新给所有客户端"""
        if symbol not in self.connections:
            return

        dead_connections = set()

        for client_ws in self.connections[symbol]:
            try:
                await client_ws.send_json({
                    "type": "update",
                    "data": update
                })
            except Exception:
                dead_connections.add(client_ws)

        # 清理断开的连接
        for dead_ws in dead_connections:
            self.connections[symbol].discard(dead_ws)

    async def _get_snapshot(self, symbol: str, depth: int) -> dict:
        """获取订单簿快照"""
        from app.services.binance_client import get_orderbook_depth

        # 调用现有的获取深度接口
        full_depth = await get_orderbook_depth(symbol.upper(), depth)

        return {
            "bids": full_depth.get("bids", [])[:depth],
            "asks": full_depth.get("asks", [])[:depth],
            "bestBid": full_depth.get("bestBid"),
            "bestAsk": full_depth.get("bestAsk"),
            "spread": full_depth.get("spread"),
            "timestamp": full_depth.get("timestamp"),
        }


# 全局实例
_manager = OrderbookStreamManager()


def get_orderbook_stream_manager() -> OrderbookStreamManager:
    """获取全局管理器实例"""
    return _manager
