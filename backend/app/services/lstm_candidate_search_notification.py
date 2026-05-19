from __future__ import annotations

from typing import Any

from app.services.lstm_candidate_library import lstm_candidate_library_summary
from app.services.wxpusher_app_client import WxPusherAppClient, wxpusher_app_configured


def notify_lstm_candidate_search_finished(report: dict[str, Any]) -> dict[str, Any]:
    if not wxpusher_app_configured():
        return {"sent": False, "reason": "wxpusher_app_not_configured"}
    message = _message_payload(report)
    response = WxPusherAppClient().send_markdown(
        summary=message["summary"],
        content=message["content"],
    )
    return {"sent": True, "provider": "wxpusher_app", "response": response}


def _message_payload(report: dict[str, Any]) -> dict[str, str]:
    symbols = _symbols(report)
    summary = f"LSTM候选训练完成：{','.join(symbols)}"
    lines = [
        f"# {summary}",
        "",
        _plain_summary(report),
        "",
        "## 各周期结果",
        *_duration_lines(report),
        "",
        "说明：只有 trade_active 会替换成可下单模型；shadow_active 只继续做影子模拟。",
    ]
    return {"summary": summary, "content": "\n".join(lines)}


def _plain_summary(report: dict[str, Any]) -> str:
    candidates = _all_candidates(report)
    total = len(candidates)
    trade = _count_status(candidates, {"trade_active", "trained"})
    shadow = _count_status(candidates, {"shadow_active"})
    rejected = _count_status(candidates, {"validation_failed"})
    failed = _count_status(candidates, {"failed", "insufficient_samples"})
    return (
        f"这轮一共新训练了 {total} 个候选。"
        f"可下单模型 {trade} 个，影子可观察模型 {shadow} 个，"
        f"没过验证 {rejected} 个，训练失败或样本不足 {failed} 个。"
    )


def _duration_lines(report: dict[str, Any]) -> list[str]:
    lines = []
    for result in report.get("results") or []:
        symbol = str(result.get("symbol") or "").upper()
        duration = str(result.get("duration") or "")
        summary = lstm_candidate_library_summary(symbol, duration)
        lines.append(_duration_line(symbol, duration, result, summary))
    return lines


def _duration_line(
    symbol: str,
    duration: str,
    result: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    trade = summary.get("bestTradeCandidate")
    shadow = summary.get("bestShadowCandidate")
    latest = summary.get("latest") or {}
    if trade:
        return f"- {symbol} {duration}：找到可下单模型，{_record_text(trade)}。"
    if shadow:
        return f"- {symbol} {duration}：有影子模型但还没达到下单门槛，{_record_text(shadow)}。"
    if result.get("reason") == "candidate_search_exhausted":
        return f"- {symbol} {duration}：参数空间已经跑完，候选库共 {summary.get('total') or 0} 个。"
    return f"- {symbol} {duration}：这轮还没找到可用模型，最近候选 {_record_text(latest)}。"


def _record_text(record: dict[str, Any]) -> str:
    model = record.get("modelVersion") or "无模型版本"
    config = record.get("config") or {}
    validation = record.get("validation") or {}
    test = record.get("test") or {}
    return (
        f"{model}，window={config.get('featureWindow')}，"
        f"move={config.get('minMoveBps')}bp，epochs={config.get('epochs')}，"
        f"validation胜率={_pct(validation.get('winRate'))}，test胜率={_pct(test.get('winRate'))}"
    )


def _all_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        candidate
        for result in report.get("results") or []
        for candidate in result.get("candidates") or []
    ]


def _count_status(candidates: list[dict[str, Any]], statuses: set[str]) -> int:
    return sum(1 for candidate in candidates if str(candidate.get("status") or "") in statuses)


def _symbols(report: dict[str, Any]) -> list[str]:
    values = {
        str(result.get("symbol") or "").upper()
        for result in report.get("results") or []
        if result.get("symbol")
    }
    return sorted(values) or ["UNKNOWN"]


def _pct(value: Any) -> str:
    if value is None:
        return "无"
    return f"{float(value) * 100:.2f}%"
