from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.auto_trade_types import AutoTradeSettings
from app.services.wxpusher_app_client import WxPusherAppClient, wxpusher_app_configured


def notify_live_order_success(
    settings: AutoTradeSettings,
    prediction: dict[str, Any],
    result: dict[str, Any],
    *,
    entry_price: float,
    order_time: datetime | None = None,
) -> dict[str, Any]:
    if not settings.live_trading_enabled:
        return {"sent": False, "reason": "live_trading_disabled"}
    if not wxpusher_app_configured():
        return {"sent": False, "reason": "wxpusher_app_not_configured"}
    message = _success_message(settings, prediction, result, entry_price, _time_text(order_time))
    response = WxPusherAppClient().send_markdown(summary=message["summary"], content=message["content"])
    return {"sent": True, "provider": "wxpusher_app", "response": response}


def notify_live_order_failure(
    settings: AutoTradeSettings,
    prediction: dict[str, Any],
    exc: Exception,
    *,
    entry_price: float | None = None,
    order_time: datetime | None = None,
) -> dict[str, Any]:
    if not settings.live_trading_enabled:
        return {"sent": False, "reason": "live_trading_disabled"}
    if not wxpusher_app_configured():
        return {"sent": False, "reason": "wxpusher_app_not_configured"}
    message = _failure_message(settings, prediction, exc, entry_price, _time_text(order_time))
    response = WxPusherAppClient().send_markdown(summary=message["summary"], content=message["content"])
    return {"sent": True, "provider": "wxpusher_app", "response": response}


def _success_message(
    settings: AutoTradeSettings,
    prediction: dict[str, Any],
    result: dict[str, Any],
    entry_price: float,
    order_time: str,
) -> dict[str, str]:
    summary = f"实盘下单成功：{settings.symbol} {settings.duration}"
    lines = [
        f"# {summary}",
        "",
        f"- 下单时间：{order_time}",
        f"- 下单金额：{settings.qty}",
        f"- 开仓价：{entry_price}",
        f"- 币种：{settings.symbol}",
        f"- 周期：{settings.duration}",
        f"- 候选：{_candidate_label(settings, prediction)}",
        f"- 策略Key：{settings.strategy_key}",
        f"- 方向：{_direction_label(prediction)}",
        f"- 事件ID：{result.get('eventId')}",
        f"- 订单ID：{result.get('orderId')}",
        f"- 外部订单ID：{result.get('externalOrderId') or '无'}",
        f"- 外部状态：{result.get('externalStatus') or 'unknown'}",
    ]
    return {"summary": summary, "content": "\n".join(lines)}


def _failure_message(
    settings: AutoTradeSettings,
    prediction: dict[str, Any],
    exc: Exception,
    entry_price: float | None,
    order_time: str,
) -> dict[str, str]:
    summary = f"实盘下单失败：{settings.symbol} {settings.duration}"
    lines = [
        f"# {summary}",
        "",
        f"- 下单时间：{order_time}",
        f"- 下单金额：{settings.qty}",
        f"- 开仓价：{entry_price if entry_price is not None else '未取得'}",
        f"- 币种：{settings.symbol}",
        f"- 周期：{settings.duration}",
        f"- 候选：{_candidate_label(settings, prediction)}",
        f"- 策略Key：{settings.strategy_key}",
        f"- 方向：{_direction_label(prediction)}",
        f"- 失败类型：{type(exc).__name__}",
        f"- 失败内容：{exc}",
    ]
    return {"summary": summary, "content": "\n".join(lines)}


def _candidate_label(settings: AutoTradeSettings, prediction: dict[str, Any]) -> str:
    for key in ("high_winrate_rule", "model_version", "signal_key"):
        value = str(prediction.get(key) or "").strip()
        if value:
            return value
    return settings.strategy_key


def _direction_label(prediction: dict[str, Any]) -> str:
    direction = str(prediction.get("direction") or "").upper()
    if direction == "UP":
        return "BUY / UP"
    if direction == "DOWN":
        return "SELL / DOWN"
    return direction or "unknown"


def _time_text(value: datetime | None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat()
