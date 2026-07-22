# -*- coding: utf-8 -*-
"""Six-state strategy signal contract and compatibility helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


StrategySignalCode = Literal[
    "watch",
    "low_buy",
    "accumulate",
    "hold",
    "reduce",
    "exit",
]

TimingPhase = Literal[
    "declining",
    "declining_exhaustion",
    "range_bound",
    "advancing",
    "advancing_weakening",
    "advancing_exhaustion",
    "high_level_breakdown",
    "structural_risk",
    "unknown",
]


class TimingState(BaseModel):
    """Deterministic cycle state computed from completed daily bars."""

    code: Optional[str] = None
    phase: TimingPhase = "unknown"
    price_zone: str = "mid"
    volume_state: str = "unknown"
    momentum_state: str = "neutral"
    data_quality: str = "insufficient"
    suggested_signal: StrategySignalCode = "watch"
    confidence: Optional[str] = None
    summary: Optional[str] = None
    completed_bar_date: Optional[str] = None
    evidence: list[str] = Field(default_factory=list)
    safety_evidence: list[str] = Field(default_factory=list)
    weakening_evidence: list[str] = Field(default_factory=list)
    weakening_dimensions: list[str] = Field(default_factory=list)
    external_confirmation_dimensions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    reference_points: dict[str, str] = Field(default_factory=dict)


class DecisionOverride(BaseModel):
    """Optional final-decision override without mutating the machine phase."""

    applied: bool = False
    reason_code: Optional[str] = None
    source_phase: Optional[TimingPhase] = None
    final_phase: Optional[TimingPhase] = None
    category: Optional[str] = None
    event_date: Optional[str] = None
    evidence: Optional[str] = None
    source: Optional[str] = None


class StrategySignal(BaseModel):
    """Optional dashboard contract emitted by the multidimensional strategy."""

    signal_code: Optional[StrategySignalCode] = None
    signal_label: Optional[str] = None
    confidence: Optional[str] = None
    summary: Optional[str] = None
    reasons: list[str] = Field(default_factory=list)
    upgrade_trigger: Optional[str] = None
    downgrade_trigger: Optional[str] = None
    cycle_phase: Optional[TimingPhase] = None
    price_zone: Optional[str] = None
    reference_points: dict[str, str] = Field(default_factory=dict)
    decision_override: Optional[DecisionOverride] = None


@dataclass(frozen=True)
class StrategySignalDefinition:
    code: StrategySignalCode
    min_score: int
    max_score: int
    action: str
    decision_type: str
    label_zh: str
    label_en: str
    label_ko: str

    def label_for_language(self, language: Any = "zh") -> str:
        normalized = str(language or "zh").strip().lower()
        if normalized.startswith("en"):
            return self.label_en
        if normalized.startswith("ko"):
            return self.label_ko
        return self.label_zh


STRATEGY_SIGNAL_DEFINITIONS: tuple[StrategySignalDefinition, ...] = (
    StrategySignalDefinition("exit", 0, 19, "sell", "sell", "适合清仓", "Exit", "청산 적합"),
    StrategySignalDefinition("reduce", 20, 39, "reduce", "sell", "适合减仓", "Reduce", "비중 축소 적합"),
    StrategySignalDefinition("watch", 40, 49, "watch", "hold", "继续观察", "Watch", "계속 관찰"),
    StrategySignalDefinition("hold", 50, 59, "hold", "hold", "适合持有", "Hold", "보유 적합"),
    StrategySignalDefinition("low_buy", 60, 79, "buy", "buy", "适合低吸", "Buy the Dip", "저가 매수 적합"),
    StrategySignalDefinition("accumulate", 80, 100, "buy", "buy", "适合抢筹", "Accumulate", "적극 매수 적합"),
)


MULTIDIMENSIONAL_STRATEGY_POLICY_PROMPT_ZH = """
## 反向周期择时策略契约

先读取系统提供的 `dashboard.timing_state` 或 `[系统计算的周期状态]`。该对象由已完成日线机械计算，是技术阶段的权威约束；不得用主观判断、单日涨跌或旧趋势评分改写阶段。`sentiment_score` 只是旧接口兼容字段，不参与信号选择。

1. **硬风险优先**：只有 `dashboard.intelligence.hard_risk_assessment.status=confirmed`，且类别、日期、证据和来源完整的重大业绩恶化、财务或治理风险、监管处罚、退市风险等才进入 `structural_risk/exit`。普通 `risk_alerts` 只用于展示，不能改变主信号；“若出现”“可能”“未发现”“未检索到”“数据缺失”“无法判断”“需核验”等条件、否定或缺失表述不得触发清仓。
2. **下跌低吸**：`declining` 阶段无需等待趋势反转，但必须同时具备偏低位置、缩量、跌速未加快和独立安全证据。120日30%分位只是一项强位置证据，不是绝对开关；30%-45%的偏低区必须再增加一项独立安全确认。加速下跌或放量破位只能 `watch`。
3. **极限抢筹**：价格极低、量能极缩且跌速未加快时，至少再有一项动能、波动、承接或方向明确的外部改善证据才能输出 `accumulate`。模型可因基本面或市场风险降级为 `low_buy/watch`，不可绕过价格、量能和跌速前置条件。
4. **横盘观察**：`range_bound` 固定为 `watch`，描述箱体上下沿和突破条件，不输出理想买点。
5. **健康上涨**：`advancing` 默认 `hold`。高位或上涨本身绝不是买入依据；只有机器证据与方向明确的外部转弱证据合计满足后两条时，才可升级为减仓或清仓。
6. **上涨减仓**：价格位置由120/240日分位和中期结构共同确认。强高位通常需要两个独立衰减维度；55%-70%的偏高区必须至少三个维度，且至少一个不是成交量，不能把70%当成绝对开关。机器动能/量价证据可与资金派发、行业转弱或基本面走弱证据合并计数。
7. **极致清仓与高位破位**：技术性 `exit` 必须同时具备综合极高位置、完整已收盘日线、至少三个独立衰竭维度且至少一个不是成交量。普通或强高位转为下跌时进入 `high_level_breakdown` 并输出 `reduce`；只有仍满足上述极致条件时才可 `exit`。
8. **数据约束**：资金流缺失只降低置信度，不能自动改变主信号；关键信息缺失必须披露，禁止编造数字或点位。

兼容动作映射保持：`watch→watch/hold`、`low_buy→buy/buy`、`accumulate→buy/buy`、`hold→hold/hold`、`reduce→reduce/sell`、`exit→sell/sell`。不得读取或推断个人持仓、成本和仓位比例。

`dashboard` 必须包含以下对象：

```json
"strategy_signal": {
    "signal_code": "watch|low_buy|accumulate|hold|reduce|exit",
    "signal_label": "继续观察/适合低吸/适合抢筹/适合持有/适合减仓/适合清仓",
    "confidence": "高/中/低",
    "summary": "一句话综合结论",
    "reasons": ["[基本面] 实际证据", "[价格位置] 实际证据", "[量价资金] 实际证据"],
    "upgrade_trigger": "升级为更积极信号的可验证条件",
    "downgrade_trigger": "降级或失效的可验证条件",
    "cycle_phase": "declining/declining_exhaustion/range_bound/advancing/advancing_weakening/advancing_exhaustion/high_level_breakdown/structural_risk/unknown",
    "price_zone": "extreme_low/low/mid/high/extreme_high",
    "reference_points": {"zone_label": "动作对应点位名称", "zone": "可靠价格区间或N/A", "risk_label": "失效或下一触发名称", "risk_line": "可靠价格或条件"}
}
```

`dashboard.intelligence` 还必须包含以下对象；没有确认硬风险时使用 `status=none`，不得把数据缺失写成已确认风险：

```json
"hard_risk_assessment": {
    "status": "confirmed|unconfirmed|none",
    "category": "delisting|fraud|adverse_audit|regulatory_action|earnings_collapse|default_or_insolvency|fundamental_reversal|null",
    "event_date": "YYYY-MM-DD或空字符串",
    "evidence": "引用输入中已确认的事实或空字符串",
    "source": "公告、财报或新闻来源或空字符串"
}
```

`reasons` 必须有 3-5 条并覆盖至少三个独立维度，每条引用实际数据、结构或已检索事件；缺失数据要写明，禁止补造数字。`signal_code`、`signal_label`、`operation_advice`、`action` 与 `decision_type` 必须一致；不要根据 `sentiment_score` 反推信号。
"""

_DEFINITIONS_BY_CODE = {definition.code: definition for definition in STRATEGY_SIGNAL_DEFINITIONS}


def _normalize_code(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def strategy_signal_definition(value: Any) -> Optional[StrategySignalDefinition]:
    """Return the immutable definition for a recognized six-state code."""

    return _DEFINITIONS_BY_CODE.get(_normalize_code(value))  # type: ignore[arg-type]


def strategy_signal_definition_for_score(score: Any) -> StrategySignalDefinition:
    """Return the canonical six-state definition for a score."""

    try:
        normalized_score = int(float(score))
    except (TypeError, ValueError):
        normalized_score = 50
    normalized_score = min(max(normalized_score, 0), 100)
    for definition in STRATEGY_SIGNAL_DEFINITIONS:
        if definition.min_score <= normalized_score <= definition.max_score:
            return definition
    return _DEFINITIONS_BY_CODE["hold"]


def strategy_signal_definition_for_timing_state(payload: Any) -> StrategySignalDefinition:
    """Resolve the deterministic candidate without consulting the legacy score."""

    if isinstance(payload, Mapping):
        suggested = strategy_signal_definition(payload.get("suggested_signal"))
        if suggested is not None:
            return suggested
        phase = _normalize_code(payload.get("phase"))
    else:
        phase = ""
    phase_defaults = {
        "declining": "watch",
        "declining_exhaustion": "accumulate",
        "range_bound": "watch",
        "advancing": "hold",
        "advancing_weakening": "reduce",
        "advancing_exhaustion": "exit",
        "high_level_breakdown": "reduce",
        "structural_risk": "exit",
        "unknown": "watch",
    }
    return _DEFINITIONS_BY_CODE[phase_defaults.get(phase, "watch")]


def compatibility_score_for_strategy_signal(code: Any) -> int:
    """Return a stable legacy score that never participates in signal selection."""

    definition = strategy_signal_definition(code) or _DEFINITIONS_BY_CODE["watch"]
    return (definition.min_score + definition.max_score) // 2


def align_score_to_strategy_signal(code: Any, score: Any) -> Any:
    """Clamp a numeric score to the score band required by a strategy signal."""

    definition = strategy_signal_definition(code)
    if definition is None:
        return score
    try:
        normalized_score = int(float(score))
    except (TypeError, ValueError):
        return score
    return min(max(normalized_score, definition.min_score), definition.max_score)


def normalize_strategy_signal_payload(
    payload: Any,
    language: Any = "zh",
) -> Optional[dict[str, Any]]:
    """Normalize a valid strategy signal while preserving its evidence fields."""

    if not isinstance(payload, Mapping):
        return None
    definition = strategy_signal_definition(payload.get("signal_code"))
    if definition is None:
        return None

    normalized = dict(payload)
    normalized["signal_code"] = definition.code
    normalized["signal_label"] = definition.label_for_language(language)

    reasons = normalized.get("reasons")
    if reasons is None:
        normalized["reasons"] = []
    elif not isinstance(reasons, list):
        normalized["reasons"] = [str(reasons).strip()] if str(reasons).strip() else []
    else:
        normalized["reasons"] = [str(reason).strip() for reason in reasons if str(reason).strip()]
    return normalized
