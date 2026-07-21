# -*- coding: utf-8 -*-
"""
===================================
Report Engine - Jinja2 Report Renderer
===================================

Renders reports from Jinja2 templates. Falls back to caller's logic on template
missing or render error. Template path is relative to project root.
Any expensive data preparation should be injected by the caller via extra_context.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.analyzer import AnalysisResult
from src.config import get_config
from src.market_phase_summary import format_public_market_status_line, format_public_phase_pack_excerpt
from src.report_language import (
    get_localized_stock_name,
    get_report_labels,
    get_signal_level,
    get_chip_unavailable_reason,
    is_chip_structure_unavailable,
    localize_chip_health,
    localize_operation_advice,
    localize_trend_prediction,
    normalize_report_language,
)
from src.schemas.decision_action import (
    display_action_fields_for_result,
    display_decision_type_for_result,
    display_operation_advice_for_result,
    localize_action_label,
)
from src.schemas.strategy_signal import (
    normalize_strategy_signal_payload,
    strategy_signal_definition,
)
from src.utils.data_processing import (
    normalize_model_used,
    signal_attribution_has_content,
    signal_attribution_weight_items,
)

logger = logging.getLogger(__name__)


def _normalize_rendered_output(platform: str, output: str) -> str:
    """Keep mobile notification templates from expanding Jinja control whitespace."""

    if platform != "wechat":
        return output
    normalized = output.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n")).strip()
    return re.sub(r"\n{3,}", "\n\n", normalized)


def _escape_md(text: str) -> str:
    """Escape markdown special chars (*ST etc)."""
    if not text:
        return ""
    return text.replace("*", "\\*").replace("_", "\\_")


def _clean_sniper_value(val: Any) -> str:
    """Format sniper point value for display (strip label prefixes)."""
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val).strip() if val else ""
    if not s or s == "N/A":
        return s or "N/A"
    prefixes = [
        "理想买入点：", "次优买入点：", "止损位：", "目标位：",
        "理想买入点:", "次优买入点:", "止损位:", "目标位:",
        "Ideal Entry:", "Secondary Entry:", "Stop Loss:", "Target:",
    ]
    for prefix in prefixes:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def _compact_items(items: Any, item_limit: int = 64, total_limit: int = 190) -> str:
    """Join report evidence without letting one stock dominate a notification."""

    if not isinstance(items, (list, tuple)):
        items = [items] if items else []
    flattened: List[Any] = []
    pending = list(items)
    while pending:
        item = pending.pop(0)
        if isinstance(item, (list, tuple)):
            pending[0:0] = list(item)
        else:
            flattened.append(item)

    parts: List[str] = []
    for item in flattened:
        if item is None:
            continue
        text = " ".join(str(item).split())
        if not text:
            continue
        if len(text) > item_limit:
            text = text[: item_limit - 1] + "…"
        candidate = "；".join(parts + [text])
        if len(candidate) > total_limit:
            break
        parts.append(text)
    return "；".join(parts)


def _prioritize_external_reasons(items: Any) -> List[Any]:
    if not isinstance(items, list):
        return [items] if items else []
    external_markers = ("基本面", "行业", "市场", "资金", "筹码", "新闻", "事件", "数据限制")
    external = [item for item in items if any(marker in str(item) for marker in external_markers)]
    remaining = [item for item in items if item not in external]
    return external + remaining


def _timing_label(kind: str, value: Any, language: str) -> str:
    key = str(value or "unknown")
    mappings = {
        "phase": {
            "declining": ("下跌过程", "Declining", "하락 구간"),
            "declining_exhaustion": ("下跌衰竭", "Decline exhaustion", "하락 소진"),
            "range_bound": ("横盘震荡", "Range-bound", "박스권"),
            "advancing": ("健康上涨", "Healthy advance", "건전한 상승"),
            "advancing_weakening": ("上涨动能衰减", "Advance weakening", "상승 동력 약화"),
            "advancing_exhaustion": ("上涨极致", "Advance exhaustion", "상승 소진"),
            "high_level_breakdown": ("高位破位", "High-level breakdown", "고점 이탈"),
            "structural_risk": ("结构性风险", "Structural risk", "구조적 위험"),
            "unknown": ("数据不足", "Insufficient data", "데이터 부족"),
        },
        "price_zone": {
            "extreme_low": ("极低位", "Extreme low", "극저위"),
            "low": ("低位", "Low", "저위"),
            "mid": ("中位", "Mid-range", "중위"),
            "high": ("高位", "High", "고위"),
            "extreme_high": ("极高位", "Extreme high", "극고위"),
        },
        "volume_state": {
            "extreme_contraction": ("极致缩量", "Extreme contraction", "극단적 거래량 축소"),
            "contraction": ("缩量", "Volume contraction", "거래량 축소"),
            "normal": ("常态量能", "Normal volume", "보통 거래량"),
            "expansion": ("放量", "Volume expansion", "거래량 확대"),
            "climax": ("高潮量", "Climactic volume", "거래량 클라이맥스"),
            "unknown": ("量能不足", "Volume unavailable", "거래량 부족"),
        },
        "momentum_state": {
            "decelerating_down": ("跌速放缓", "Decline decelerating", "하락세 둔화"),
            "accelerating_down": ("下跌加速", "Decline accelerating", "하락 가속"),
            "weak_down": ("弱势下行", "Weak decline", "약세 하락"),
            "healthy_up": ("上涨健康", "Healthy momentum", "건전한 상승"),
            "weakening_up": ("上涨转弱", "Momentum weakening", "상승세 약화"),
            "exhausted_up": ("上涨衰竭", "Momentum exhausted", "상승 소진"),
            "neutral": ("动能中性", "Neutral momentum", "중립 모멘텀"),
            "unknown": ("动能不足", "Momentum unavailable", "모멘텀 부족"),
        },
        "data_quality": {
            "full": ("完整", "Full", "충분"),
            "partial": ("部分", "Partial", "부분"),
            "insufficient": ("不足", "Insufficient", "부족"),
            "unknown": ("不足", "Unknown", "부족"),
        },
    }
    localized = mappings.get(kind, {}).get(key)
    if not localized:
        return key
    index = {"zh": 0, "en": 1, "ko": 2}.get(language, 0)
    return localized[index]


def _resolve_templates_dir() -> Path:
    """Resolve template directory relative to project root."""
    config = get_config()
    base = Path(__file__).resolve().parent.parent.parent
    templates_dir = Path(config.report_templates_dir)
    if not templates_dir.is_absolute():
        return base / templates_dir
    return templates_dir


def render(
    platform: str,
    results: List[AnalysisResult],
    report_date: Optional[str] = None,
    summary_only: bool = False,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Render report using Jinja2 template.

    Args:
        platform: One of: markdown, wechat, brief
        results: List of AnalysisResult
        report_date: Report date string (default: today)
        summary_only: Whether to output summary only
        extra_context: Additional template context

    Returns:
        Rendered string, or None on error (caller should fallback).
    """
    from datetime import datetime

    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError:
        logger.warning("jinja2 not installed, report renderer disabled")
        return None

    if report_date is None:
        report_date = datetime.now().strftime("%Y-%m-%d")

    templates_dir = _resolve_templates_dir()
    template_name = f"report_{platform}.j2"
    template_path = templates_dir / template_name
    if not template_path.exists():
        logger.debug("Report template not found: %s", template_path)
        return None

    report_language = normalize_report_language(
        (extra_context or {}).get("report_language")
        or next(
            (getattr(result, "report_language", None) for result in results if getattr(result, "report_language", None)),
            None,
        )
        or getattr(get_config(), "report_language", "zh")
    )
    labels = get_report_labels(report_language)

    # Strategy reports are risk-first; legacy reports keep score ordering.
    signal_priority = {"exit": 0, "reduce": 1, "accumulate": 2, "low_buy": 3, "watch": 4, "hold": 5}

    def result_sort_key(result: AnalysisResult) -> tuple[int, float]:
        dashboard = result.dashboard if isinstance(result.dashboard, dict) else {}
        strategy = normalize_strategy_signal_payload(dashboard.get("strategy_signal"), report_language)
        code = strategy.get("signal_code") if strategy else None
        if code in signal_priority:
            return signal_priority[code], -float(getattr(result, "sentiment_score", 50) or 50)
        return 99, -float(getattr(result, "sentiment_score", 50) or 50)

    sorted_results = (
        sorted(results, key=result_sort_key)
        if platform == "wechat"
        else sorted(results, key=lambda item: item.sentiment_score, reverse=True)
    )
    sorted_enriched = []
    for r in sorted_results:
        display_action = display_action_fields_for_result(
            r,
            report_language=report_language,
        )["action"]
        legacy_display_advice = display_operation_advice_for_result(
            r,
            report_language=report_language,
        )
        dashboard = r.dashboard if isinstance(r.dashboard, dict) else {}
        timing_state = dashboard.get("timing_state")
        timing_state = timing_state if isinstance(timing_state, dict) else {}
        strategy_signal = normalize_strategy_signal_payload(
            dashboard.get("strategy_signal"),
            report_language,
        )
        strategy_definition = strategy_signal_definition(
            strategy_signal.get("signal_code") if strategy_signal else None
        )
        if strategy_signal and strategy_definition:
            display_advice = strategy_signal["signal_label"]
            signal_action = strategy_definition.action
            display_bucket = strategy_definition.decision_type
        else:
            display_advice = legacy_display_advice
            signal_action = {
                "buy": "buy",
                "add": "buy",
                "hold": "hold",
                "reduce": "reduce",
                "sell": "sell",
                "watch": "watch",
                "avoid": "hold",
                "alert": "sell",
            }.get(display_action, display_action)
            display_bucket = display_decision_type_for_result(r, report_language=report_language)
        _, se, _ = get_signal_level(signal_action or display_advice, r.sentiment_score, report_language)
        rn = get_localized_stock_name(r.name, r.code, report_language)
        sorted_enriched.append({
            "result": r,
            "signal_text": display_advice,
            "signal_emoji": se,
            "strategy_signal": strategy_signal,
            "timing_state": timing_state,
            "display_decision_type": display_bucket,
            "stock_name": _escape_md(rn),
            "localized_operation_advice": display_advice,
            "localized_trend_prediction": localize_trend_prediction(r.trend_prediction, report_language),
        })

    display_buckets = [entry["display_decision_type"] for entry in sorted_enriched]
    buy_count = sum(1 for bucket in display_buckets if bucket == "buy")
    sell_count = sum(1 for bucket in display_buckets if bucket == "sell")
    hold_count = len(display_buckets) - buy_count - sell_count
    strategy_summary: List[Dict[str, Any]] = []
    if sorted_enriched and all(entry["strategy_signal"] for entry in sorted_enriched):
        strategy_counts: Dict[str, int] = {}
        for entry in sorted_enriched:
            code = entry["strategy_signal"]["signal_code"]
            strategy_counts[code] = strategy_counts.get(code, 0) + 1
        for code in ("exit", "reduce", "accumulate", "low_buy", "watch", "hold"):
            count = strategy_counts.get(code, 0)
            definition = strategy_signal_definition(code)
            if count and definition is not None:
                strategy_summary.append(
                    {
                        "code": code,
                        "label": definition.label_for_language(report_language),
                        "count": count,
                    }
                )
    show_llm_model = bool(getattr(get_config(), "report_show_llm_model", True))
    models_used: List[str] = []
    if show_llm_model:
        for result in results:
            model = normalize_model_used(getattr(result, "model_used", None))
            if model:
                models_used.append(model)
        models_used = list(dict.fromkeys(models_used))

    report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def failed_checks(checklist: List[str]) -> List[str]:
        return [c for c in (checklist or []) if c.startswith("❌") or c.startswith("⚠️")]

    def phase_pack_excerpt(result: AnalysisResult) -> str:
        return format_public_phase_pack_excerpt(
            getattr(result, "market_phase_summary", None),
            getattr(result, "analysis_context_pack_overview", None),
            source=getattr(result, "analysis_visibility_source", None) or "evaluator_snapshot",
            report_language=report_language,
        )

    def market_status_line() -> str:
        for source_results in (results or [], sorted_results):
            for result in source_results:
                line = format_public_market_status_line(
                    getattr(result, "market_phase_summary", None),
                    report_language=report_language,
                )
                if line:
                    return line
        return ""

    context: Dict[str, Any] = {
        "report_date": report_date,
        "report_timestamp": report_timestamp,
        "results": sorted_results,
        "enriched": sorted_enriched,
        "summary_only": summary_only,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "strategy_summary": strategy_summary,
        "labels": labels,
        "report_language": report_language,
        "models_used": models_used,
        "show_llm_model": show_llm_model,
        "market_status_line": market_status_line(),
        "escape_md": _escape_md,
        "clean_sniper": _clean_sniper_value,
        "compact_items": _compact_items,
        "prioritize_external_reasons": _prioritize_external_reasons,
        "timing_label": lambda kind, value: _timing_label(kind, value, report_language),
        "failed_checks": failed_checks,
        "phase_pack_excerpt": phase_pack_excerpt,
        "history_by_code": {},
        "get_chip_unavailable_reason": get_chip_unavailable_reason,
        "is_chip_structure_unavailable": is_chip_structure_unavailable,
        "localize_operation_advice": localize_operation_advice,
        "localize_action_label": localize_action_label,
        "localize_trend_prediction": localize_trend_prediction,
        "localize_chip_health": localize_chip_health,
        "signal_attribution_has_content": signal_attribution_has_content,
        "signal_attribution_weight_items": signal_attribution_weight_items,
    }
    if extra_context:
        safe_extra_context = dict(extra_context)
        safe_extra_context.pop("labels", None)
        safe_extra_context.pop("report_language", None)
        context.update(safe_extra_context)

    try:
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(default=False),
        )
        template = env.get_template(template_name)
        return _normalize_rendered_output(platform, template.render(**context))
    except Exception as e:
        logger.warning("Report render failed for %s: %s", template_name, e)
        return None
