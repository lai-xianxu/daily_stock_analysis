# -*- coding: utf-8 -*-
"""
===================================
Report Engine - Report renderer tests
===================================

Tests for Jinja2 report rendering and fallback behavior.
"""

import sys
import unittest
from copy import deepcopy
from unittest.mock import MagicMock, patch

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    sys.modules["litellm"] = MagicMock()

from src.analyzer import AnalysisResult
from src.services.report_renderer import render


def _make_result(
    code: str = "600519",
    name: str = "贵州茅台",
    sentiment_score: int = 72,
    operation_advice: str = "持有",
    analysis_summary: str = "稳健",
    decision_type: str = "hold",
    dashboard: dict = None,
    report_language: str = "zh",
    model_used: str = None,
) -> AnalysisResult:
    if dashboard is None:
        dashboard = {
            "core_conclusion": {"one_sentence": "持有观望"},
            "intelligence": {"risk_alerts": []},
            "battle_plan": {"sniper_points": {"stop_loss": "110"}},
        }
    return AnalysisResult(
        code=code,
        name=name,
        trend_prediction="看多",
        sentiment_score=sentiment_score,
        operation_advice=operation_advice,
        analysis_summary=analysis_summary,
        decision_type=decision_type,
        dashboard=dashboard,
        report_language=report_language,
        model_used=model_used,
    )


def _make_renderer_config(show_llm_model: bool = True) -> MagicMock:
    config = MagicMock()
    config.report_templates_dir = "templates"
    config.report_language = "zh"
    config.report_show_llm_model = show_llm_model
    return config


def _with_decision_signal_summary(result: AnalysisResult) -> AnalysisResult:
    result.decision_signal_summary = {
        "action": "sell",
        "action_label": "卖出",
        "horizon": "1d",
        "reason": "技术面走弱",
    }
    return result


def _make_strategy_result() -> AnalysisResult:
    result = _make_result(
        sentiment_score=68,
        operation_advice="减仓",
        decision_type="sell",
        dashboard={
            "strategy_signal": {
                "signal_code": "low_buy",
                "signal_label": "错误标签",
                "confidence": "中",
                "summary": "支撑附近抛压减弱，等待转强确认",
                "reasons": [
                    "[基本面] 盈利与现金流未见明显恶化",
                    "[价格位置] 接近可验证支撑区域",
                    "[量价资金] 下跌量能收敛且资金流趋稳",
                ],
                "upgrade_trigger": "形成更高低点并收复短期均线",
                "downgrade_trigger": "有效跌破支撑且基本面风险上升",
                "cycle_phase": "declining",
                "price_zone": "low",
                "reference_points": {
                    "zone_label": "低吸观察区",
                    "zone": "98.00-100.00",
                    "risk_label": "结构失效参考",
                    "risk_line": "96.00",
                },
            },
            "timing_state": {
                "phase": "declining",
                "price_zone": "low",
                "volume_state": "contraction",
                "momentum_state": "decelerating_down",
                "data_quality": "full",
                "suggested_signal": "low_buy",
                "confidence": "中",
                "summary": "下跌过程中抛压收敛",
                "evidence": [
                    "[价格位置] 120日价格分位18.0%，区间=low",
                    "[量能] 5日均量分位20.0%，5/20量比0.72，状态=contraction",
                    "[动能] 最近5日跌速较前5日放缓",
                ],
                "metrics": {
                    "current_price": 100,
                    "price_percentile_120": 18,
                    "volume_percentile_120": 20,
                    "volume_ratio_5_20": 0.72,
                },
                "reference_points": {
                    "zone_label": "低吸观察区",
                    "zone": "98.00-100.00",
                    "risk_label": "结构失效参考",
                    "risk_line": "96.00",
                },
                "limitations": [],
            },
            "core_conclusion": {
                "one_sentence": "旧核心结论仍保留",
                "position_advice": {
                    "no_position": "空仓者等待回踩",
                    "has_position": "持仓者控制三成仓位",
                },
            },
            "intelligence": {
                "latest_news": "近期公告未见重大硬风险",
                "risk_alerts": ["行业需求仍需跟踪"],
            },
            "data_perspective": {
                "trend_status": {"ma_alignment": "震荡", "is_bullish": False, "trend_score": 48},
                "price_position": {"current_price": 100, "support_level": 98, "resistance_level": 110},
                "volume_analysis": {
                    "volume_ratio": 0.72,
                    "volume_status": "缩量",
                    "turnover_rate": 1.2,
                    "volume_meaning": "下跌抛压有所收敛",
                },
            },
            "battle_plan": {
                "sniper_points": {"ideal_buy": "98-100", "stop_loss": "96", "take_profit": "110"},
                "position_strategy": {
                    "suggested_position": "三成仓位",
                    "entry_plan": "首次买入两成",
                    "risk_control": "控制仓位",
                },
                "action_checklist": ["✅ 长期逻辑未明显恶化"],
            },
        },
    )
    result.market_snapshot = {"close": "100", "volume_ratio": "0.72", "source": "test"}
    return result


class TestReportRenderer(unittest.TestCase):
    """Report renderer tests."""

    def test_render_markdown_summary_only(self) -> None:
        """Markdown platform renders with summary_only."""
        r = _make_result()
        out = render("markdown", [r], summary_only=True)
        self.assertIsNotNone(out)
        self.assertIn("决策仪表盘", out)
        self.assertIn("贵州茅台", out)
        self.assertIn("买入", out)
        self.assertIn("🟢买入:1", out)

    def test_render_markdown_preserves_guardrailed_neutral_action(self) -> None:
        r = _make_result(
            dashboard={
                "core_conclusion": {"one_sentence": "等待确认"},
                "decision_stability": {"applied": True, "reason": "等待回踩确认"},
            }
        )

        out = render("markdown", [r], summary_only=True)

        self.assertIsNotNone(out)
        self.assertIn("持有", out)
        self.assertIn("🟡观望:1", out)

    def test_render_markdown_uses_explicit_avoid_and_alert_text(self) -> None:
        avoid = _make_result(
            code="AVOID",
            name="Avoid Corp",
            sentiment_score=90,
            operation_advice="Buy",
            report_language="en",
        )
        avoid.action = "avoid"
        avoid.action_label = "Avoid"
        alert = _make_result(
            code="ALERT",
            name="Alert Corp",
            sentiment_score=85,
            operation_advice="Buy",
            report_language="en",
        )
        alert.action = "alert"
        alert.action_label = "Alert"

        out = render("markdown", [avoid, alert], summary_only=True)

        self.assertIsNotNone(out)
        self.assertIn("🟡 **Avoid Corp(AVOID)**: Avoid | Score 90", out)
        self.assertIn("🔴 **Alert Corp(ALERT)**: Alert | Score 85", out)
        self.assertIn("**Avoid Corp(AVOID)**: Avoid | Score 90", out)
        self.assertIn("**Alert Corp(ALERT)**: Alert | Score 85", out)
        self.assertNotIn("**Avoid Corp(AVOID)**: Buy", out)
        self.assertNotIn("**Alert Corp(ALERT)**: Buy", out)

    def test_render_markdown_full(self) -> None:
        """Markdown platform renders full report."""
        r = _make_result()
        out = render("markdown", [r], summary_only=False)
        self.assertIsNotNone(out)
        self.assertIn("核心结论", out)
        self.assertIn("作战计划", out)
        self.assertNotIn("盘中决策护栏", out)

    def test_all_reports_prioritize_strategy_signal_and_evidence(self) -> None:
        result = _make_strategy_result()

        for platform in ("markdown", "wechat", "brief"):
            with self.subTest(platform=platform):
                out = render(platform, [result], summary_only=False)
                self.assertIsNotNone(out)
                if platform == "wechat":
                    self.assertIn("**周期**", out)
                else:
                    self.assertIn("综合策略判断", out)
                self.assertIn("适合低吸", out)
                self.assertIn("置信度", out)
                self.assertIn("盈利与现金流未见明显恶化", out)
                self.assertIn("形成更高低点并收复短期均线", out)
                self.assertIn("有效跌破支撑且基本面风险上升", out)

    def test_strategy_markdown_precedes_and_preserves_legacy_sections(self) -> None:
        out = render("markdown", [_make_strategy_result()], summary_only=False)

        self.assertIsNotNone(out)
        self.assertLess(out.index("综合策略判断"), out.index("重要信息速览"))
        for legacy_text in ("核心结论", "当日行情", "数据透视", "量比", "作战计划"):
            self.assertIn(legacy_text, out)
        for personalized_text in ("空仓者等待回踩", "持仓者控制三成仓位", "首次买入两成"):
            self.assertNotIn(personalized_text, out)

    def test_strategy_wechat_is_compact_and_uses_six_state_summary(self) -> None:
        out = render("wechat", [_make_strategy_result()], summary_only=False)

        self.assertIsNotNone(out)
        self.assertIn("适合低吸:1", out)
        self.assertIn("盈利与现金流未见明显恶化", out)
        self.assertIn("行业需求仍需跟踪", out)
        self.assertIn("下跌过程", out)
        self.assertIn("低吸观察区", out)
        self.assertNotIn("评分", out)
        self.assertNotIn("理想买入点", out)
        self.assertNotIn("🟢买入:1", out)
        for personalized_text in (
            "空仓者等待回踩",
            "持仓者控制三成仓位",
            "三成仓位",
            "首次买入两成",
        ):
            self.assertNotIn(personalized_text, out)

    def test_strategy_wechat_stays_compact_for_ten_stocks(self) -> None:
        results = []
        for index in range(10):
            result = deepcopy(_make_strategy_result())
            result.code = f"{300000 + index:06d}"
            result.name = f"示例{index + 1}"
            results.append(result)

        out = render("wechat", results, summary_only=False)

        self.assertIsNotNone(out)
        self.assertLess(len(out), 5500)
        self.assertEqual(out.count("**周期**"), 10)
        for personalized_text in ("空仓者", "持仓者", "三成仓位", "首次买入两成"):
            self.assertNotIn(personalized_text, out)

    def test_strategy_wechat_sorts_risk_first_and_uses_action_specific_points(self) -> None:
        low_buy = deepcopy(_make_strategy_result())
        low_buy.name = "低吸样例"
        low_buy.code = "LOW"

        reduce = deepcopy(_make_strategy_result())
        reduce.name = "减仓样例"
        reduce.code = "REDUCE"
        reduce.dashboard["strategy_signal"].update(
            {
                "signal_code": "reduce",
                "signal_label": "适合减仓",
                "cycle_phase": "advancing_weakening",
                "price_zone": "high",
                "reference_points": {
                    "zone_label": "减仓压力区",
                    "zone": "118.00-122.00",
                    "risk_label": "重新转强参考",
                    "risk_line": "123.00",
                },
            }
        )
        reduce.dashboard["timing_state"].update(
            {
                "phase": "advancing_weakening",
                "price_zone": "high",
                "volume_state": "contraction",
                "momentum_state": "weakening_up",
                "suggested_signal": "reduce",
                "reference_points": reduce.dashboard["strategy_signal"]["reference_points"],
            }
        )

        exit_result = deepcopy(reduce)
        exit_result.name = "清仓样例"
        exit_result.code = "EXIT"
        exit_result.dashboard["strategy_signal"].update(
            {
                "signal_code": "exit",
                "signal_label": "适合清仓",
                "cycle_phase": "advancing_exhaustion",
                "reference_points": {
                    "zone_label": "清仓触发区",
                    "zone": "128.00-130.00",
                    "risk_label": "重新转强参考",
                    "risk_line": "131.00",
                },
            }
        )
        exit_result.dashboard["timing_state"].update(
            {
                "phase": "advancing_exhaustion",
                "price_zone": "extreme_high",
                "momentum_state": "exhausted_up",
                "suggested_signal": "exit",
                "reference_points": exit_result.dashboard["strategy_signal"]["reference_points"],
            }
        )

        out = render("wechat", [low_buy, reduce, exit_result], summary_only=False)

        self.assertIsNotNone(out)
        self.assertLess(out.index("清仓样例"), out.index("减仓样例"))
        self.assertLess(out.index("减仓样例"), out.index("低吸样例"))
        self.assertIn("清仓触发区", out)
        self.assertIn("减仓压力区", out)
        self.assertNotIn("理想买入点", out)

    def test_report_without_strategy_signal_keeps_legacy_layout(self) -> None:
        out = render("markdown", [_make_result()], summary_only=False)

        self.assertIsNotNone(out)
        self.assertNotIn("综合策略判断", out)
        self.assertIn("核心结论", out)

    def test_render_markdown_omits_decision_signal_excerpt(self) -> None:
        """Markdown reports omit the duplicated DecisionSignal excerpt."""
        r = _with_decision_signal_summary(_make_result())

        summary_out = render("markdown", [r], summary_only=True)
        self.assertIsNotNone(summary_out)
        self.assertNotIn("AI 决策信号", summary_out)

        full_out = render("markdown", [r], summary_only=False)
        self.assertIsNotNone(full_out)
        self.assertNotIn("AI 决策信号", full_out)
        self.assertNotIn("理由: 技术面走弱", full_out)

    def test_render_markdown_phase_decision_section(self) -> None:
        """Markdown renders phase_decision when present."""
        r = _make_result(
            dashboard={
                "core_conclusion": {"one_sentence": "等待确认"},
                "intelligence": {"risk_alerts": []},
                "phase_decision": {
                    "action_window": "盘中跟踪",
                    "immediate_action": "等待确认",
                    "watch_conditions": ["放量突破"],
                    "next_check_time": "14:30",
                    "confidence_reason": "数据质量可用",
                    "data_limitations": ["quote: stale"],
                },
                "battle_plan": {"sniper_points": {"stop_loss": "110"}},
            }
        )

        out = render("markdown", [r], summary_only=False)

        self.assertIsNotNone(out)
        self.assertIn("盘中决策护栏", out)
        self.assertIn("盘中跟踪", out)
        self.assertIn("放量突破", out)
        self.assertIn("quote: stale", out)

    def test_render_markdown_skips_context_only_phase_decision_shape(self) -> None:
        """Markdown skips mechanically shaped phase_decision without actionable content."""
        r = _make_result(
            dashboard={
                "core_conclusion": {"one_sentence": "持有观望"},
                "intelligence": {"risk_alerts": []},
                "phase_decision": {
                    "phase_context": {"phase": "intraday", "market": "cn"},
                    "action_window": None,
                    "immediate_action": None,
                    "watch_conditions": [],
                    "next_check_time": None,
                    "confidence_reason": None,
                    "data_limitations": [],
                },
                "battle_plan": {"sniper_points": {"stop_loss": "110"}},
            }
        )

        out = render("markdown", [r], summary_only=False)

        self.assertIsNotNone(out)
        self.assertNotIn("盘中决策护栏", out)

    def test_render_wechat(self) -> None:
        """Wechat platform renders."""
        r = _make_result()
        out = render("wechat", [r])
        self.assertIsNotNone(out)
        self.assertIn("贵州茅台", out)

    def test_render_wechat_omits_decision_signal_excerpt(self) -> None:
        """Wechat reports omit the duplicated DecisionSignal excerpt."""
        r = _with_decision_signal_summary(_make_result())

        summary_out = render("wechat", [r], summary_only=True)
        self.assertIsNotNone(summary_out)
        self.assertNotIn("AI 决策信号", summary_out)

        full_out = render("wechat", [r], summary_only=False)
        self.assertIsNotNone(full_out)
        self.assertNotIn("AI 决策信号", full_out)
        self.assertNotIn("理由: 技术面走弱", full_out)

    def test_render_brief(self) -> None:
        """Brief platform renders 3-5 sentence summary."""
        r = _make_result()
        out = render("brief", [r])
        self.assertIsNotNone(out)
        self.assertIn("决策简报", out)
        self.assertIn("贵州茅台", out)

    def test_render_brief_omits_decision_signal_excerpt(self) -> None:
        r = _with_decision_signal_summary(_make_result())

        out = render("brief", [r])

        self.assertIsNotNone(out)
        self.assertNotIn("AI 决策信号", out)

    def test_render_brief_respects_model_visibility_toggle(self) -> None:
        r = _make_result(model_used="gemini/gemini-2.5-flash")

        with patch("src.services.report_renderer.get_config", return_value=_make_renderer_config(True)):
            visible = render("brief", [r])
        with patch("src.services.report_renderer.get_config", return_value=_make_renderer_config(False)):
            hidden = render("brief", [r])

        self.assertIsNotNone(visible)
        self.assertIsNotNone(hidden)
        self.assertIn("分析模型: gemini/gemini-2.5-flash", visible)
        self.assertNotIn("分析模型", hidden)
        self.assertNotIn("gemini/gemini-2.5-flash", hidden)

    def test_render_templates_show_compact_market_status_only(self) -> None:
        r = _make_result()
        r.market_phase_summary = {
            "phase": "intraday",
            "market": "cn",
            "trigger_source": "api",
            "is_partial_bar": True,
        }
        r.analysis_context_pack_overview = {
            "data_quality": {
                "level": "limited",
                "limitations": ["quote: stale", "news: missing", "technical: fallback"],
            }
        }
        r.raw_response = "raw context pack should not appear"

        out = render("brief", [r])

        self.assertIsNotNone(out)
        self.assertIn("市场状态：A股 · 盘中", out)
        self.assertNotIn("阶段：intraday", out)
        self.assertNotIn("盘中数据提示", out)
        self.assertNotIn("数据质量: limited", out)
        self.assertNotIn("限制: quote: stale", out)
        self.assertNotIn("限制: news: missing", out)
        self.assertNotIn("technical: fallback", out)
        self.assertNotIn("raw context pack", out)

    def test_render_templates_skip_phase_pack_excerpt_when_summary_missing(self) -> None:
        r = _make_result()

        out = render("brief", [r])

        self.assertIsNotNone(out)
        self.assertNotIn("摘要来源", out)
        self.assertNotIn("evaluator snapshot", out)

    def test_render_market_status_preserves_input_order(self) -> None:
        cn = _make_result(
            code="600519",
            name="贵州茅台",
            sentiment_score=60,
        )
        cn.market_phase_summary = {"market": "cn", "phase": "postmarket"}
        us = _make_result(
            code="AAPL",
            name="Apple",
            sentiment_score=90,
        )
        us.market_phase_summary = {"market": "us", "phase": "premarket"}

        out = render("markdown", [cn, us], summary_only=True)

        self.assertIsNotNone(out)
        self.assertIn("市场状态：A股 · 盘后", out)
        self.assertNotIn("市场状态：美股 · 盘前", out)

    def test_render_markdown_footer_uses_consistent_separator(self) -> None:
        r = _make_result(model_used="gemini/gemini-2.5-flash")

        with patch("src.services.report_renderer.get_config", return_value=_make_renderer_config(True)):
            out = render("markdown", [r], summary_only=True)

        self.assertIsNotNone(out)
        self.assertIn("报告生成时间：", out)
        self.assertIn("分析模型：gemini/gemini-2.5-flash", out)
        self.assertNotIn("分析模型: gemini/gemini-2.5-flash", out)

    def test_render_markdown_in_english(self) -> None:
        """Markdown renderer switches headings and summary labels for English reports."""
        r = _make_result(
            name="Kweichow Moutai",
            operation_advice="Buy",
            analysis_summary="Momentum remains constructive.",
            report_language="en",
        )
        out = render("markdown", [r], summary_only=True)
        self.assertIsNotNone(out)
        self.assertIn("Decision Dashboard", out)
        self.assertIn("Summary", out)
        self.assertIn("Buy", out)

    def test_render_markdown_market_snapshot_uses_template_context(self) -> None:
        """Market snapshot macro should render localized labels with template context."""
        r = _make_result(
            code="AAPL",
            name="Apple",
            operation_advice="Buy",
            report_language="en",
        )
        r.market_snapshot = {
            "close": "180.10",
            "prev_close": "178.25",
            "open": "179.00",
            "high": "181.20",
            "low": "177.80",
            "pct_chg": "+1.04%",
            "change_amount": "1.85",
            "amplitude": "1.91%",
            "volume": "1200000",
            "amount": "215000000",
            "price": "180.35",
            "volume_ratio": "1.2",
            "turnover_rate": "0.8%",
            "source": "polygon",
        }

        out = render("markdown", [r], summary_only=False)

        self.assertIsNotNone(out)
        self.assertIn("Market Snapshot", out)
        self.assertIn("Volume Ratio", out)

    def test_render_markdown_collapses_unavailable_chip_structure(self) -> None:
        r = _make_result(
            dashboard={
                "core_conclusion": {"one_sentence": "持有观望"},
                "data_perspective": {
                    "chip_structure": {
                        "profit_ratio": "数据缺失，无法判断",
                        "avg_cost": "数据缺失，无法判断",
                        "concentration": "数据缺失，无法判断",
                        "chip_health": "数据缺失，无法判断",
                    }
                },
            }
        )

        out = render("markdown", [r], summary_only=False)

        self.assertIsNotNone(out)
        self.assertIn("**筹码**: 筹码分布未启用或数据源暂不可用，未纳入筹码判断。", out)
        self.assertEqual(out.count("数据缺失，无法判断"), 0)

    def test_render_unknown_platform_returns_none(self) -> None:
        """Unknown platform returns None (caller fallback)."""
        r = _make_result()
        out = render("unknown_platform", [r])
        self.assertIsNone(out)

    def test_render_empty_results_returns_content(self) -> None:
        """Empty results still produces header."""
        out = render("markdown", [], summary_only=True)
        self.assertIsNotNone(out)
        self.assertIn("0", out)
