# -*- coding: utf-8 -*-
"""Contract tests for the default multidimensional timing strategy."""

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from src.agent.executor import AGENT_SYSTEM_PROMPT
from src.agent.factory import resolve_skill_prompt_state
from src.analyzer import GeminiAnalyzer
from src.schemas.report_schema import AnalysisReportSchema

from src.schemas.strategy_signal import (
    align_score_to_strategy_signal,
    normalize_strategy_signal_payload,
    strategy_signal_definition,
)


def test_six_strategy_signals_map_to_legacy_contract() -> None:
    expected = {
        "exit": ((0, 19), "sell", "sell", "适合清仓"),
        "reduce": ((20, 39), "reduce", "sell", "适合减仓"),
        "watch": ((40, 49), "watch", "hold", "继续观察"),
        "hold": ((50, 59), "hold", "hold", "适合持有"),
        "low_buy": ((60, 79), "buy", "buy", "适合低吸"),
        "accumulate": ((80, 100), "buy", "buy", "适合抢筹"),
    }

    for code, (score_range, action, decision_type, label) in expected.items():
        definition = strategy_signal_definition(code)
        assert definition is not None
        assert (definition.min_score, definition.max_score) == score_range
        assert definition.action == action
        assert definition.decision_type == decision_type
        assert definition.label_zh == label


def test_strategy_signal_score_is_clamped_to_its_compatible_band() -> None:
    assert align_score_to_strategy_signal("low_buy", 35) == 60
    assert align_score_to_strategy_signal("low_buy", 72) == 72
    assert align_score_to_strategy_signal("watch", 88) == 49
    assert align_score_to_strategy_signal("unknown", 88) == 88


def test_strategy_signal_payload_normalizes_known_code_and_label() -> None:
    normalized = normalize_strategy_signal_payload(
        {
            "signal_code": "LOW_BUY",
            "signal_label": "随意文本",
            "confidence": "中",
            "summary": "接近支撑但仍需确认",
            "reasons": ["[价格位置] 接近前低", "[量价资金] 抛压减弱"],
        }
    )

    assert normalized is not None
    assert normalized["signal_code"] == "low_buy"
    assert normalized["signal_label"] == "适合低吸"
    assert normalize_strategy_signal_payload({"signal_code": "unknown"}) is None


def test_default_strategy_uses_multidimensional_gates_without_position_assumptions() -> None:
    strategy_path = Path(__file__).resolve().parents[1] / "strategies" / "volume_contraction_timing.yaml"
    strategy = yaml.safe_load(strategy_path.read_text(encoding="utf-8"))

    assert strategy["default_active"] is True
    assert strategy["default_router"] is True
    required_tools = {
        "get_realtime_quote",
        "get_daily_history",
        "analyze_trend",
        "get_volume_analysis",
        "get_stock_info",
        "get_market_indices",
        "get_sector_rankings",
        "get_capital_flow",
        "get_chip_distribution",
        "search_stock_news",
    }
    assert required_tools <= set(strategy["required_tools"])

    instructions = strategy["instructions"]
    for phrase in (
        "数据质量门槛",
        "长期逻辑过滤",
        "市场与行业环境",
        "价格结构与位置",
        "量价、资金与筹码确认",
        "事件风险否决",
        "成交量不能独立决定",
        "下跌缩量但没有止跌结构",
        "上涨缩量但距离压力位较远",
    ):
        assert phrase in instructions
    for personalized_position_text in ("20%-30%", "25%-50%", "建议仓位", "持仓比例"):
        assert personalized_position_text not in instructions


def test_implicit_default_run_uses_multidimensional_skill_aware_prompt() -> None:
    state = resolve_skill_prompt_state(
        config=SimpleNamespace(agent_skills=[], agent_skill_dir=None),
    )

    assert state.skills_to_activate == ["volume_contraction_timing"]
    assert state.use_legacy_default_prompt is False
    assert state.default_skill_policy == ""
    assert "基本面约束下的多维择时" in state.skill_instructions


def test_normal_and_agent_prompts_require_the_same_six_state_contract() -> None:
    for prompt in (GeminiAnalyzer.SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT):
        for code in ("watch", "low_buy", "accumulate", "hold", "reduce", "exit"):
            assert code in prompt
        for phrase in (
            '"strategy_signal"',
            '"action"',
            "3-5",
            "至少三个独立维度",
            "单一成交量",
            "事件风险否决",
            "数据缺失",
            "仅在维度缺失时调用对应工具",
            "参考点位无可靠数据时必须标记 N/A",
        ):
            assert phrase in prompt
        assert '"operation_advice": "继续观察/适合低吸/适合抢筹/适合持有/适合减仓/适合清仓"' in prompt
        assert '"position_advice"' not in prompt
        assert "空仓者建议" not in prompt
        assert "分持仓建议" not in prompt
        assert '"operation_advice": "买入/加仓/持有/减仓/卖出/观望"' not in prompt
        assert "Canonical 评分与动作口径" not in prompt
        assert "80-100：强烈买入，`action=buy`" not in prompt
        assert "0-19：卖出，`action=sell`" not in prompt


def test_agent_prompt_requests_every_multidimensional_data_tool() -> None:
    for tool_name in (
        "get_realtime_quote",
        "get_daily_history",
        "analyze_trend",
        "get_volume_analysis",
        "get_stock_info",
        "get_market_indices",
        "get_sector_rankings",
        "get_capital_flow",
        "get_chip_distribution",
        "search_stock_news",
    ):
        assert tool_name in AGENT_SYSTEM_PROMPT


def test_report_schema_accepts_complete_and_missing_strategy_signal() -> None:
    complete = AnalysisReportSchema.model_validate(
        {
            "stock_name": "贵州茅台",
            "sentiment_score": 68,
            "trend_prediction": "震荡",
            "operation_advice": "适合低吸",
            "dashboard": {
                "strategy_signal": {
                    "signal_code": "low_buy",
                    "signal_label": "适合低吸",
                    "confidence": "中",
                    "summary": "支撑附近抛压减弱",
                    "reasons": [
                        "[基本面] 盈利与现金流未见明显恶化",
                        "[价格位置] 接近可验证支撑",
                        "[量价资金] 下跌量能收敛",
                    ],
                    "upgrade_trigger": "放量收复短期结构",
                    "downgrade_trigger": "支撑失守且风险上升",
                }
            },
        }
    )
    legacy = AnalysisReportSchema.model_validate(
        {
            "stock_name": "贵州茅台",
            "sentiment_score": 55,
            "trend_prediction": "震荡",
            "operation_advice": "持有",
            "dashboard": {"core_conclusion": {"one_sentence": "继续持有"}},
        }
    )

    assert complete.dashboard is not None
    assert complete.dashboard.strategy_signal is not None
    assert complete.dashboard.strategy_signal.signal_code == "low_buy"
    assert legacy.dashboard is not None
    assert legacy.dashboard.strategy_signal is None


def test_parser_uses_strategy_signal_as_primary_compatible_action() -> None:
    analyzer = GeminiAnalyzer()
    response = json.dumps(
        {
            "stock_name": "贵州茅台",
            "sentiment_score": 35,
            "trend_prediction": "看空",
            "operation_advice": "卖出",
            "decision_type": "sell",
            "action": "sell",
            "confidence_level": "高",
            "analysis_summary": "测试",
            "dashboard": {
                "strategy_signal": {
                    "signal_code": "low_buy",
                    "signal_label": "错误标签",
                    "confidence": "中",
                    "summary": "支撑附近出现低吸条件",
                    "reasons": [
                        "[基本面] 长期逻辑未恶化",
                        "[价格位置] 接近支撑",
                        "[量价资金] 抛压减弱",
                    ],
                    "upgrade_trigger": "形成更高低点",
                    "downgrade_trigger": "有效跌破支撑",
                }
            },
        },
        ensure_ascii=False,
    )

    result = analyzer._parse_response(
        response,
        "600519",
        "贵州茅台",
        synthesize_strategy_signal=True,
    )

    assert result.sentiment_score == 60
    assert result.operation_advice == "适合低吸"
    assert result.action == "buy"
    assert result.action_label == "买入"
    assert result.decision_type == "buy"
    assert result.confidence_level == "中"
    assert result.dashboard["strategy_signal"]["signal_label"] == "适合低吸"
    calibration = result.dashboard["decision_score_calibration"]
    assert calibration["raw_score"] == 35
    assert calibration["adjusted_score"] == 60
    assert calibration["final_action"] == "buy"


def test_parser_without_valid_strategy_signal_synthesizes_six_state_contract() -> None:
    analyzer = GeminiAnalyzer()
    response = json.dumps(
        {
            "stock_name": "贵州茅台",
            "sentiment_score": 35,
            "trend_prediction": "看空",
            "operation_advice": "减仓",
            "decision_type": "sell",
            "action": "reduce",
            "confidence_level": "中",
            "analysis_summary": "长期逻辑尚未反转，但价格结构和量价表现偏弱。",
            "fundamental_analysis": "盈利仍有韧性，现金流需要继续验证。",
            "technical_analysis": "价格跌破短期均线并接近前低。",
            "volume_analysis": "下跌期间量能放大，资金承接不足。",
            "sector_position": "行业相对强弱处于后排。",
            "risk_warning": "若前低失守，结构风险将进一步扩大。",
            "dashboard": {
                "core_conclusion": {"one_sentence": "先降低风险暴露"},
                "strategy_signal": {"signal_code": "unknown"},
            },
        },
        ensure_ascii=False,
    )

    result = analyzer._parse_response(
        response,
        "600519",
        "贵州茅台",
        synthesize_strategy_signal=True,
    )

    assert result.sentiment_score == 35
    assert result.operation_advice == "适合减仓"
    assert result.action == "reduce"
    assert result.decision_type == "sell"
    strategy = result.dashboard["strategy_signal"]
    assert strategy["signal_code"] == "reduce"
    assert strategy["signal_label"] == "适合减仓"
    assert 3 <= len(strategy["reasons"]) <= 5
    assert any(reason.startswith("[基本面]") for reason in strategy["reasons"])
    assert any(reason.startswith("[价格结构]") for reason in strategy["reasons"])
    assert any(reason.startswith("[量价资金]") for reason in strategy["reasons"])
    assert result.dashboard["decision_score_calibration"]["strategy_signal_source"] == "score_fallback"
