from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.db.session import get_conn
from app.services.factor_combo_position_service import factor_combo_positions_payload

router = APIRouter(prefix="/api/factors/combinations", tags=["factors"])


@router.get("/positions")
def factor_combo_positions(
    symbol: str = Query(..., min_length=6),
    duration: str = Query(...),
    factorName: str | None = Query(None),
    limit: int = Query(80, gt=0, le=300),
) -> dict:
    conn = get_conn()
    try:
        return factor_combo_positions_payload(
            conn,
            symbol=symbol,
            duration=duration,
            factor_name=factorName,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        conn.close()
