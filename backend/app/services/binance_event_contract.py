from __future__ import annotations

import os
from decimal import Decimal, InvalidOperation
from typing import Any

import requests
from requests import RequestException

PLACE_ORDER_URL = "https://www.binance.com/bapi/futures/v2/private/future/event-contract/place-order"
SUCCESS_CODE = "000000"
REQUEST_TIMEOUT = (3, 8)
TIME_INCREMENT_BY_INTERVAL = {
    "10m": "TEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "60m": "ONE_HOUR",
    "1d": "ONE_DAY",
}
DIRECTION_BY_SIDE = {"BUY": "LONG", "SELL": "SHORT"}


class BinanceEventConfigError(RuntimeError):
    pass


class BinanceEventOrderError(RuntimeError):
    pass


def place_event_contract_order(
    *,
    symbol: str,
    event_interval: str,
    side: str,
    amount: float,
    payout_ratio: float,
) -> dict[str, Any]:
    headers, cookies = _load_auth()
    payload = build_event_contract_order_payload(
        symbol=symbol,
        event_interval=event_interval,
        side=side,
        amount=amount,
        payout_ratio=payout_ratio,
    )
    response_data = _post_order(headers, cookies, payload)
    _assert_success(response_data)
    return {
        "request": payload,
        "response": response_data,
        "externalOrderId": _extract_order_id(response_data),
        "externalStatus": "PLACED",
    }


def build_event_contract_order_payload(
    *,
    symbol: str,
    event_interval: str,
    side: str,
    amount: float,
    payout_ratio: float,
) -> dict[str, str]:
    return _build_payload(
        symbol=symbol,
        event_interval=event_interval,
        side=side,
        amount=amount,
        payout_ratio=payout_ratio,
    )


def _load_auth() -> tuple[dict[str, str], dict[str, str]]:
    csrf_token = os.getenv("BINANCE_EVENT_CSRF_TOKEN", "").strip()
    p20t = os.getenv("BINANCE_EVENT_P20T", "").strip()
    if not csrf_token or not p20t:
        raise BinanceEventConfigError(
            "missing BINANCE_EVENT_CSRF_TOKEN or BINANCE_EVENT_P20T environment variable"
        )
    return {
        "content-type": "application/json",
        "clienttype": "web",
        "csrftoken": csrf_token,
    }, {"p20t": p20t}


def _build_payload(
    *,
    symbol: str,
    event_interval: str,
    side: str,
    amount: float,
    payout_ratio: float,
) -> dict[str, str]:
    if event_interval not in TIME_INCREMENT_BY_INTERVAL:
        raise BinanceEventOrderError(f"unsupported event interval: {event_interval}")
    if side not in DIRECTION_BY_SIDE:
        raise BinanceEventOrderError(f"unsupported order side: {side}")
    return {
        "orderAmount": _format_amount(amount),
        "timeIncrements": TIME_INCREMENT_BY_INTERVAL[event_interval],
        "symbolName": symbol.upper(),
        "payoutRatio": _format_ratio(payout_ratio),
        "direction": DIRECTION_BY_SIDE[side],
    }


def _post_order(
    headers: dict[str, str],
    cookies: dict[str, str],
    payload: dict[str, str],
) -> dict[str, Any]:
    try:
        response = requests.post(
            PLACE_ORDER_URL,
            headers=headers,
            cookies=cookies,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except ValueError as exc:
        raise BinanceEventOrderError("binance event order returned non-json response") from exc
    except RequestException as exc:
        raise BinanceEventOrderError(f"binance event order request failed: {exc}") from exc


def _assert_success(response_data: dict[str, Any]) -> None:
    success = response_data.get("success")
    code = str(response_data.get("code", ""))
    if success is True or code == SUCCESS_CODE:
        return
    raise BinanceEventOrderError(f"binance event order rejected: {_message(response_data)}")


def _message(response_data: dict[str, Any]) -> str:
    for key in ("message", "msg"):
        value = response_data.get(key)
        if value:
            return str(value)
    return str({"code": response_data.get("code"), "success": response_data.get("success")})


def _extract_order_id(response_data: dict[str, Any]) -> str | None:
    data = response_data.get("data")
    if not isinstance(data, dict):
        return None
    for key in ("orderId", "orderNo", "id"):
        value = data.get(key)
        if value is not None:
            return str(value)
    return None


def _format_amount(value: float) -> str:
    amount = _positive_decimal(value)
    if amount == amount.to_integral_value():
        return format(amount.quantize(Decimal("1")), "f")
    return format(amount.normalize(), "f")


def _format_ratio(value: float) -> str:
    return f"{_positive_decimal(value):.2f}"


def _positive_decimal(value: float) -> Decimal:
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise BinanceEventOrderError(f"invalid numeric value: {value}") from exc
    if amount <= 0:
        raise BinanceEventOrderError(f"numeric value must be greater than 0: {value}")
    return amount
