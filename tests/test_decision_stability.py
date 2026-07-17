# -*- coding: utf-8 -*-
"""Tests for structure-aware decision stability calibration."""

from types import SimpleNamespace

from src.analyzer import AnalysisResult, _capital_flow_bias, stabilize_decision_with_structure


def _result(
    *,
    decision_type: str,
    operation_advice: str,
    score: int,
    current_price: float,
    change_pct: float = 0.0,
) -> AnalysisResult:
    return AnalysisResult(
        code="002812",
        name="恩捷股份",
        sentiment_score=score,
        trend_prediction="看多" if decision_type == "buy" else "看空",
        operation_advice=operation_advice,
        decision_type=decision_type,
        report_language="zh",
        current_price=current_price,
        change_pct=change_pct,
        dashboard={
            "core_conclusion": {"one_sentence": "原始结论"},
            "data_perspective": {
                "price_position": {
                    "current_price": current_price,
                    "support_level": 30.0,
                    "resistance_level": 34.0,
                }
            },
        },
    )


def _fund_flow(main: float, five_day: float = 0.0, ten_day: float = 0.0) -> dict:
    return {
        "capital_flow": {
            "status": "ok",
            "data": {
                "stock_flow": {
                    "main_net_inflow": main,
                    "inflow_5d": five_day,
                    "inflow_10d": ten_day,
                }
            },
        }
    }


def _unsupported_fund_flow() -> dict:
    return {"capital_flow": {"status": "not_supported", "data": {}}}


def _unsupported_fund_flow_caps() -> dict:
    return {"capital_flow": {"status": "NOT_SUPPORTED", "data": {"stock_flow": {"main_net_inflow": 0}}}}


def _attach_timing_state(
    result: AnalysisResult,
    *,
    phase: str = "declining",
    signal: str = "low_buy",
    confidence: str = "高",
) -> None:
    result.dashboard["timing_state"] = {
        "phase": phase,
        "price_zone": "low",
        "volume_state": "contraction",
        "momentum_state": "decelerating_down",
        "data_quality": "full",
        "suggested_signal": signal,
        "confidence": confidence,
        "summary": "下跌过程中量能和跌速收敛",
        "evidence": ["[价格位置] 处于低位", "[动能] 跌速放缓"],
        "reference_points": {
            "zone_label": "低吸观察区",
            "zone": "30.00-31.00",
            "risk_label": "结构失效参考",
            "risk_line": "29.50",
        },
    }


def test_capital_flow_bias_is_unavailable_when_stock_flow_data_is_missing() -> None:
    assert _capital_flow_bias(_unsupported_fund_flow()) == "unavailable"
    assert _capital_flow_bias({"capital_flow": {"status": "ok", "data": {}}}) == "unavailable"


def test_capital_flow_bias_is_neutral_when_missing_main_windows_conflict() -> None:
    context = {
        "capital_flow": {
            "data": {
                "stock_flow": {
                    "inflow_5d": 2_000_000,
                    "inflow_10d": -1_000_000,
                }
            }
        }
    }

    assert _capital_flow_bias(context) == "neutral"


def test_capital_flow_bias_is_neutral_when_main_conflicts_with_windows() -> None:
    context = _fund_flow(main=-500_000, five_day=1_200_000, ten_day=2_000_000)

    assert _capital_flow_bias(context) == "neutral"


def test_downgrades_buy_near_resistance_without_fund_confirmation() -> None:
    result = _result(
        decision_type="buy",
        operation_advice="买入",
        score=65,
        current_price=33.4,
    )

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=-1_000_000, five_day=-2_000_000),
    )

    assert result.decision_type == "hold"
    assert result.sentiment_score <= 59
    assert result.operation_advice == "震荡观望"
    assert result.dashboard["decision_stability"]["applied"] is True
    assert "不宜仅因短线反弹追买" in result.risk_warning
    assert result.dashboard["core_conclusion"]["signal_type"] == "🟡持有观望"


def test_downgrades_buy_mid_range_with_neutral_fund_flow() -> None:
    result = _result(
        decision_type="buy",
        operation_advice="买入",
        score=66,
        current_price=32.0,
    )

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=0, five_day=0, ten_day=0),
    )

    assert result.decision_type == "hold"
    assert result.sentiment_score <= 59
    assert result.operation_advice == "震荡观望"
    assert "资金流不明确" in result.risk_warning


def test_downgrades_buy_when_capital_flow_is_unavailable() -> None:
    buy_result = _result(
        decision_type="buy",
        operation_advice="买入",
        score=66,
        current_price=32.0,
    )
    sell_result = _result(
        decision_type="sell",
        operation_advice="卖出",
        score=30,
        current_price=30.4,
        change_pct=-2.1,
    )

    stabilize_decision_with_structure(
        buy_result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _unsupported_fund_flow(),
    )
    stabilize_decision_with_structure(
        sell_result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _unsupported_fund_flow(),
    )

    assert buy_result.decision_type == "hold"
    assert buy_result.operation_advice == "持有观察"
    assert buy_result.confidence_level == "低"
    assert buy_result.sentiment_score <= 59
    assert buy_result.dashboard["decision_stability"]["applied"] is True
    assert "买入结论缺少资金面确认" in buy_result.dashboard["decision_stability"]["reason"]
    assert buy_result.dashboard["core_conclusion"]["signal_type"] == "🟡持有观望"
    assert sell_result.decision_type == "sell"
    assert sell_result.operation_advice == "卖出"
    assert sell_result.dashboard["decision_stability"]["applied"] is False
    assert "未使用资金流校准" in sell_result.dashboard["decision_stability"]["reason"]


def test_strategy_signal_downgrade_updates_all_public_action_fields() -> None:
    result = _result(
        decision_type="buy",
        operation_advice="适合低吸",
        score=66,
        current_price=32.0,
    )
    result.action = "buy"
    result.dashboard["strategy_signal"] = {
        "signal_code": "low_buy",
        "signal_label": "适合低吸",
        "confidence": "中",
        "summary": "支撑附近出现低吸条件",
        "reasons": [
            "[基本面] 长期逻辑未恶化",
            "[价格结构] 接近支撑",
            "[量价资金] 抛压减弱",
        ],
        "upgrade_trigger": "资金回流并形成转强结构",
        "downgrade_trigger": "有效跌破支撑",
    }
    result.dashboard["core_conclusion"]["position_advice"] = {
        "no_position": "空仓者建议",
        "has_position": "持仓者建议",
    }
    result.dashboard["battle_plan"] = {
        "position_strategy": {"suggested_position": "三成仓位"}
    }

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _unsupported_fund_flow(),
    )

    assert result.sentiment_score == 49
    assert result.operation_advice == "继续观察"
    assert result.action == "watch"
    assert result.decision_type == "hold"
    assert result.confidence_level == "低"
    strategy = result.dashboard["strategy_signal"]
    assert strategy["signal_code"] == "watch"
    assert strategy["signal_label"] == "继续观察"
    assert "资金流" in strategy["summary"]
    assert result.dashboard["operation_advice"] == "继续观察"
    assert result.dashboard["decision_type"] == "hold"
    assert result.dashboard["action"] == "watch"
    assert "position_advice" not in result.dashboard["core_conclusion"]
    assert "position_strategy" not in result.dashboard["battle_plan"]


def test_downgrades_buy_when_capital_flow_values_are_na() -> None:
    result = _result(
        decision_type="buy",
        operation_advice="买入",
        score=66,
        current_price=33.0,
    )

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        {
            "capital_flow": {
                "status": "ok",
                "data": {
                    "stock_flow": {
                        "main_net_inflow": "N/A",
                        "inflow_5d": "N/A",
                        "inflow_10d": "N/A",
                    }
                },
            }
        },
    )

    assert result.decision_type == "hold"
    assert result.operation_advice == "持有观察"
    assert result.dashboard["decision_stability"]["applied"] is True
    assert "资金流数据缺失" in result.dashboard["decision_stability"]["capital_flow_status"]


def test_downgrades_buy_advice_when_decision_type_is_hold_and_capital_flow_unavailable() -> None:
    result = _result(
        decision_type="hold",
        operation_advice="建议买入",
        score=68,
        current_price=32.0,
    )

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _unsupported_fund_flow(),
    )

    assert result.decision_type == "hold"
    assert result.operation_advice == "持有观察"
    assert result.sentiment_score <= 59
    assert result.dashboard["decision_stability"]["applied"] is True
    assert "买入结论缺少资金面确认" in result.dashboard["decision_stability"]["reason"]


def test_downgrades_buy_when_capital_flow_status_is_unavailable_case_insensitive() -> None:
    buy_result = _result(
        decision_type="buy",
        operation_advice="买入",
        score=66,
        current_price=32.0,
    )

    stabilize_decision_with_structure(
        buy_result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _unsupported_fund_flow_caps(),
    )

    assert buy_result.decision_type == "hold"
    assert buy_result.operation_advice == "持有观察"
    assert buy_result.dashboard["decision_stability"]["applied"] is True
    assert "暂不支持" in str(buy_result.dashboard["decision_stability"]["capital_flow_status"])


def test_skips_downgrade_when_only_generic_risk_warning_and_sell_near_support() -> None:
    result = _result(
        decision_type="sell",
        operation_advice="卖出",
        score=30,
        current_price=30.4,
        change_pct=1.0,
    )
    result.risk_warning = "注意常见回撤风险，建议关注仓位。"

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=500_000, five_day=300_000),
    )

    assert result.decision_type == "hold"
    assert result.operation_advice == "洗盘观察"
    assert "价格贴近支撑且未见资金持续流出" in result.risk_warning


def test_stability_can_infer_decision_from_natural_chinese_phrases_in_analyzer_path() -> None:
    result = _result(
        decision_type="建议卖出",
        operation_advice="建议卖出",
        score=30,
        current_price=30.4,
        change_pct=1.0,
    )

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=500_000, five_day=300_000),
    )

    assert result.decision_type == "hold"
    assert result.operation_advice == "洗盘观察"
    assert result.dashboard["decision_stability"]["applied"] is True


def test_downgrades_sell_near_support_without_sustained_outflow() -> None:
    result = _result(
        decision_type="sell",
        operation_advice="卖出",
        score=30,
        current_price=30.4,
        change_pct=-2.1,
    )

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=800_000, five_day=1_200_000),
    )

    assert result.decision_type == "hold"
    assert result.sentiment_score >= 45
    assert result.operation_advice == "洗盘观察"
    assert "不宜仅因单日下跌直接卖出" in result.risk_warning


def test_preserves_sell_signal_when_significant_risk_exists_near_support() -> None:
    result = _result(
        decision_type="sell",
        operation_advice="卖出",
        score=30,
        current_price=30.4,
        change_pct=-2.1,
    )
    result.risk_warning = "重大利空消息：公司发布重大减持计划"
    result.dashboard["intelligence"] = {"risk_alerts": ["股东高位减持预告"]}

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=800_000, five_day=1_200_000),
    )

    assert result.decision_type == "sell"
    assert result.operation_advice == "卖出"


def test_refines_hold_pullback_near_support_as_shakeout_watch() -> None:
    result = _result(
        decision_type="hold",
        operation_advice="持有",
        score=52,
        current_price=30.5,
        change_pct=-1.6,
    )

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=0, five_day=500_000),
    )

    assert result.decision_type == "hold"
    assert result.operation_advice == "洗盘观察"
    assert "更适合按洗盘观察处理" in result.risk_warning


def test_timing_entry_survives_missing_capital_flow_with_lower_confidence() -> None:
    result = _result(
        decision_type="buy",
        operation_advice="适合低吸",
        score=85,
        current_price=30.4,
        change_pct=-1.0,
    )
    _attach_timing_state(result)

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _unsupported_fund_flow(),
    )

    strategy = result.dashboard["strategy_signal"]
    assert strategy["signal_code"] == "low_buy"
    assert strategy["confidence"] == "中"
    assert result.operation_advice == "适合低吸"
    assert result.action == "buy"
    assert any("仅下调置信度，不改变主信号" in reason for reason in strategy["reasons"])


def test_material_fundamental_risk_overrides_extreme_low_accumulation() -> None:
    result = _result(
        decision_type="buy",
        operation_advice="适合抢筹",
        score=90,
        current_price=30.0,
        change_pct=-1.2,
    )
    _attach_timing_state(
        result,
        phase="declining_exhaustion",
        signal="accumulate",
    )
    result.risk_warning = "公司被实施退市风险警示，审计无法表示意见"

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=800_000, five_day=1_200_000),
    )

    strategy = result.dashboard["strategy_signal"]
    assert strategy["signal_code"] == "exit"
    assert strategy["cycle_phase"] == "structural_risk"
    assert strategy["reference_points"]["zone_label"] == "清仓触发"
    assert "低吸" not in str(strategy["reference_points"])
    assert "抢筹" not in str(strategy["reference_points"])
    assert result.operation_advice == "适合清仓"
    assert result.action == "sell"


def test_generic_financial_data_limitation_does_not_force_exit() -> None:
    result = _result(
        decision_type="buy",
        operation_advice="适合低吸",
        score=69,
        current_price=30.4,
    )
    _attach_timing_state(result, confidence="中")
    result.risk_warning = "部分财务数据缺失，需后续验证"

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=0),
    )

    assert result.dashboard["strategy_signal"]["signal_code"] == "low_buy"


def test_negated_material_risk_text_does_not_force_exit() -> None:
    result = _result(
        decision_type="buy",
        operation_advice="适合低吸",
        score=69,
        current_price=30.4,
    )
    _attach_timing_state(result, confidence="中")
    result.risk_warning = "未发现重大风险，但仍需跟踪行业需求"

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=0),
    )

    assert result.dashboard["strategy_signal"]["signal_code"] == "low_buy"


def test_downgraded_exhaustion_watch_replaces_accumulation_points() -> None:
    result = _result(
        decision_type="hold",
        operation_advice="继续观察",
        score=44,
        current_price=30.0,
    )
    _attach_timing_state(
        result,
        phase="declining_exhaustion",
        signal="accumulate",
        confidence="中",
    )
    result.dashboard["strategy_signal"] = {
        "signal_code": "watch",
        "confidence": "中",
        "summary": "基本面仍需验证，暂不抢筹",
        "reasons": ["[基本面] 关键证据仍缺失"],
    }

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=0),
    )

    strategy = result.dashboard["strategy_signal"]
    assert strategy["signal_code"] == "watch"
    assert strategy["reference_points"]["zone_label"] == "震荡观察区"
    assert "抢筹" not in strategy["reference_points"]["zone_label"]


def test_external_capital_confirmation_can_complete_declining_low_buy() -> None:
    result = _result(
        decision_type="buy",
        operation_advice="适合低吸",
        score=68,
        current_price=30.2,
    )
    _attach_timing_state(result, phase="declining", signal="watch", confidence="中")
    result.dashboard["timing_state"].update(
        {
            "momentum_state": "weak_down",
            "metrics": {"price_percentile_120": 18.0, "support_20": 29.5, "resistance_20": 33.0},
        }
    )
    result.dashboard["strategy_signal"] = {
        "signal_code": "low_buy",
        "confidence": "中",
        "summary": "低位缩量下跌中，资金和筹码开始改善",
        "reasons": ["[资金筹码] 主力资金回流，低位承接改善"],
    }

    stabilize_decision_with_structure(result, None, _fund_flow(main=600_000))

    strategy = result.dashboard["strategy_signal"]
    assert strategy["signal_code"] == "low_buy"
    assert strategy["cycle_phase"] == "declining"
    assert result.dashboard["timing_state"]["external_confirmation_dimensions"] == ["capital_chip"]


def test_external_confirmation_cannot_override_accelerating_decline() -> None:
    result = _result(
        decision_type="buy",
        operation_advice="适合低吸",
        score=68,
        current_price=30.2,
    )
    _attach_timing_state(result, phase="declining", signal="watch", confidence="中")
    result.dashboard["timing_state"].update(
        {
            "momentum_state": "accelerating_down",
            "metrics": {"price_percentile_120": 18.0, "support_20": 29.5, "resistance_20": 33.0},
        }
    )
    result.dashboard["strategy_signal"] = {
        "signal_code": "low_buy",
        "confidence": "中",
        "summary": "资金回流但跌速仍在加快",
        "reasons": ["[资金筹码] 主力资金回流，低位承接改善"],
    }

    stabilize_decision_with_structure(result, None, _fund_flow(main=600_000))

    assert result.dashboard["strategy_signal"]["signal_code"] == "watch"


def test_external_industry_weakness_completes_high_level_exit_evidence() -> None:
    result = _result(
        decision_type="sell",
        operation_advice="适合清仓",
        score=12,
        current_price=48.0,
    )
    _attach_timing_state(result, phase="advancing", signal="hold", confidence="中")
    result.dashboard["timing_state"].update(
        {
            "price_zone": "extreme_high",
            "volume_state": "contraction",
            "momentum_state": "healthy_up",
            "weakening_dimensions": ["volume"],
            "metrics": {"support_20": 42.0, "resistance_20": 48.5, "resistance_60": 49.0},
        }
    )
    result.dashboard["strategy_signal"] = {
        "signal_code": "exit",
        "confidence": "中",
        "summary": "极高位量价背离且行业相对强弱转弱",
        "reasons": ["[行业市场] 行业相对强弱持续转弱，板块明显弱于大盘"],
    }

    stabilize_decision_with_structure(result, None, _fund_flow(main=-800_000))

    strategy = result.dashboard["strategy_signal"]
    assert strategy["signal_code"] == "exit"
    assert strategy["cycle_phase"] == "advancing_exhaustion"
    assert result.dashboard["timing_state"]["phase"] == "advancing"
    assert result.trend_prediction == "上涨极致"


def test_high_level_volume_weakness_alone_cannot_trigger_exit() -> None:
    result = _result(
        decision_type="sell",
        operation_advice="适合清仓",
        score=12,
        current_price=48.0,
    )
    _attach_timing_state(result, phase="advancing", signal="hold", confidence="中")
    result.dashboard["timing_state"].update(
        {
            "price_zone": "extreme_high",
            "volume_state": "contraction",
            "momentum_state": "healthy_up",
            "weakening_dimensions": ["volume"],
            "metrics": {"support_20": 42.0, "resistance_20": 48.5, "resistance_60": 49.0},
        }
    )
    result.dashboard["strategy_signal"] = {
        "signal_code": "exit",
        "confidence": "中",
        "summary": "极高位仅出现上涨缩量",
        "reasons": ["[量能] 上涨缩量，5日均量继续收缩"],
    }

    stabilize_decision_with_structure(result, None, _fund_flow(main=0))

    assert result.dashboard["strategy_signal"]["signal_code"] == "hold"
