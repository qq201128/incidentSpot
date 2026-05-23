from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.db.session import get_conn
from app.services.prediction_cache_service import get_latest_prediction
from app.services.kline_timing import current_rule_entry_open_time_for_duration
from app.services.prediction_policy import trade_policy_payload
from app.services.rule_config import SUPPORTED_RULE_DURATIONS
from app.services.rule_signal_service import predict_rule_direction
from app.services.strategy_registry import DEFAULT_STRATEGY_KEY
from app.services.binance_service import (
    fetch_agg_trades_display,
    fetch_index_price_klines,
    fetch_klines,
    fetch_orderbook_depth_display,
    fetch_premium_index,
)

router = APIRouter(prefix="/api", tags=["market"])
ALLOWED_INTERVALS = {"10m", "30m", "60m", "1h", "4h", "1d"}
BINANCE_KLINE_LIMIT = 1000
PREDICTION_MIN_REFRESH_LIMIT = 400
DISPLAY_KLINE_REQUEST_OPTIONS = {"max_attempts": 2, "timeout": (3, 6)}

def _upsert_klines(symbol: str, interval: str, rows: list[dict]) -> None:
    conn = get_conn()
    for item in rows:
        conn.execute(
            """
            INSERT INTO klines(symbol, interval, open_time, open, high, low, close, volume, close_time)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, interval, open_time) DO UPDATE SET
              open=excluded.open,
              high=excluded.high,
              low=excluded.low,
              close=excluded.close,
              volume=excluded.volume,
              close_time=excluded.close_time
            """,
            (
                symbol.upper(),
                interval,
                item["openTime"],
                item["open"],
                item["high"],
                item["low"],
                item["close"],
                item["volume"],
                item["closeTime"],
            ),
        )
    conn.commit()
    conn.close()

@router.get("/last-price")
def get_last_price(symbol: str = Query(..., min_length=6)) -> dict:
    """USD-M 指数价 + 标记价（GET /fapi/v1/premiumIndex），与指数 K、事件价口径一致。"""
    try:
        row = fetch_premium_index(symbol.upper())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"binance premium index failed: {exc}") from exc
    return {
        "symbol": row["symbol"],
        "indexPrice": row["indexPrice"],
        "markPrice": row["markPrice"],
        "lastFundingRate": row["lastFundingRate"],
        "nextFundingTime": row["nextFundingTime"],
        "time": row["time"],
    }

@router.get("/depth")
def get_depth(
    symbol: str = Query(..., min_length=6),
    limit: int = Query(
        20,
        ge=5,
        le=1000,
        description="Levels per side (Binance depth is fetched with the smallest allowed limit ≥ this value; max 1000).",
    ),
) -> dict:
    """USD-M order book depth (REST ``/fapi/v1/depth``), bids/asks truncated for display."""
    try:
        return fetch_orderbook_depth_display(symbol.upper(), levels=limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"order book fetch failed: {exc}") from exc


@router.get("/agg-trades")
def get_agg_trades(
    symbol: str = Query(..., min_length=6),
    limit: int = Query(40, ge=1, le=200, description="Recent aggregate trades (newest first)."),
) -> list[dict]:
    """USD-M compressed trades (REST ``/fapi/v1/aggTrades``) for latest-trades panel."""
    try:
        return fetch_agg_trades_display(symbol.upper(), limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"aggregate trades fetch failed: {exc}") from exc


@router.get("/index-price")
def get_index_price(symbol: str = Query(..., min_length=6)) -> dict:
    """Official USD-M index price + mark price (GET /fapi/v1/premiumIndex)."""
    try:
        row = fetch_premium_index(symbol.upper())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"binance premium index failed: {exc}") from exc
    return {
        "symbol": row["symbol"],
        "indexPrice": row["indexPrice"],
        "markPrice": row["markPrice"],
        "lastFundingRate": row["lastFundingRate"],
        "nextFundingTime": row["nextFundingTime"],
        "time": row["time"],
    }

@router.get("/index-klines")
def get_index_klines(
    symbol: str = Query(..., min_length=6, description="Same as Binance pair, e.g. BTCUSDT"),
    interval: str = Query("30m"),
    limit: int = Query(500, ge=1, le=1500),
) -> list[dict]:
    """Index price OHLCV from GET /fapi/v1/indexPriceKlines (not written to local klines DB)."""
    if interval not in ALLOWED_INTERVALS:
        raise HTTPException(status_code=400, detail=f"unsupported interval: {interval}")
    try:
        return fetch_index_price_klines(
            symbol.upper(),
            interval,
            limit=limit,
            request_options=DISPLAY_KLINE_REQUEST_OPTIONS,
            include_forming=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"failed to fetch index klines: {exc}") from exc

@router.get("/klines")
def get_klines(
    *,
    symbol: str = Query(..., min_length=6),
    interval: str = Query("30m"),
    limit: int = Query(500, ge=1, le=1000),
    live: bool = Query(
        False,
        description="If true, always pull from Binance REST, upsert DB, and return (for WS-down chart refresh).",
    ),
) -> list[dict]:
    if interval not in ALLOWED_INTERVALS:
        raise HTTPException(status_code=400, detail=f"unsupported interval: {interval}")

    if live:
        try:
            remote_rows = fetch_klines(symbol, interval, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"failed to fetch klines: {exc}") from exc
        if remote_rows:
            _upsert_klines(symbol, interval, remote_rows)
        return remote_rows

    conn = get_conn()
    rows = conn.execute(
        """
        SELECT open_time, open, high, low, close, volume, close_time
        FROM klines WHERE symbol = ? AND interval = ?
        ORDER BY open_time DESC LIMIT ?
        """,
        (symbol.upper(), interval, limit),
    ).fetchall()
    conn.close()

    if len(rows) >= min(limit, 100):
        return [
            {
                "openTime": row["open_time"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "closeTime": row["close_time"],
            }
            for row in reversed(rows)
        ]

    remote_rows = fetch_klines(symbol, interval, limit=limit)
    _upsert_klines(symbol, interval, remote_rows)
    return remote_rows

@router.post("/predict")
def predict(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    limit: int = Query(2000, ge=300, le=5000),
    strategyKey: str | None = Query(None),
) -> dict:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"rule engine supports only {sorted(SUPPORTED_RULE_DURATIONS)}",
        )

    sym = symbol.upper()
    _refresh_latest_1m_klines(sym, limit)
    return _predict_rule(sym, duration, strategyKey)

@router.get("/predict/latest")
def latest_prediction(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    signalKey: str | None = Query(None),
    strategyKey: str | None = Query(None),
) -> dict:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"rule engine supports only {sorted(SUPPORTED_RULE_DURATIONS)}",
        )
    try:
        return get_latest_prediction(
            symbol,
            duration,
            signal_key=signalKey,
            strategy_key=strategyKey or DEFAULT_STRATEGY_KEY,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

def _refresh_latest_1m_klines(symbol: str, limit: int) -> None:
    need_fetch = min(BINANCE_KLINE_LIMIT, max(limit, PREDICTION_MIN_REFRESH_LIMIT))
    try:
        fresh = fetch_klines(symbol, "1m", limit=need_fetch)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"failed to refresh 1m klines: {exc}") from exc

    if fresh:
        _upsert_klines(symbol, "1m", fresh)

def _predict_rule(symbol: str, duration: str, strategy_key: str | None) -> dict:
    try:
        result = predict_rule_direction(
            symbol,
            duration,
            entry_open_time=current_rule_entry_open_time_for_duration(duration),
            strategy_key=strategy_key,
        )
        return _rule_prediction_response(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"rule signal failed: {exc}") from exc

@router.post("/predict/10m")
def predict_10m(
    symbol: str = Query(..., min_length=6),
    limit: int = Query(2000, ge=300, le=5000),
    strategyKey: str | None = Query(None),
) -> dict:
    return predict(symbol=symbol, duration="10m", limit=limit, strategyKey=strategyKey)

def _rule_prediction_response(result: dict) -> dict:
    return {
        "symbol": result["symbol"],
        "signalKey": result.get("signal_key") or result.get("strategy_key"),
        "strategyKey": result.get("strategy_key"),
        "duration": result["duration"],
        "direction": result["direction"],
        "probabilityUp": result["probability_up"],
        "confidence": result["confidence"],
        "certaintyLabel": result["certainty_label"],
        "threshold": result["threshold"],
        "tradeQualityScore": result.get("trade_quality_score"),
        "tradeQualityPassed": result.get("trade_quality_passed"),
        "tradeQualityGate": result.get("trade_quality_gate"),
        "highWinrateGate": result.get("high_winrate_gate"),
        "highWinrateGatePassed": result.get("high_winrate_gate_passed"),
        "highWinrateGateValue": result.get("high_winrate_gate_value"),
        "signalSource": result.get("signal_source"),
        "ruleScore": result.get("rule_score"),
        "ruleReasons": result.get("rule_reasons"),
        "orderbook": result.get("orderbook"),
        "timeframeVotes": result.get("timeframe_votes"),
        **trade_policy_payload(result["duration"], strategy_key=result.get("strategy_key")),
    }
