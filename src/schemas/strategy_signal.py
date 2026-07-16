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


class StrategySignal(BaseModel):
    """Optional dashboard contract emitted by the multidimensional strategy."""

    signal_code: Optional[StrategySignalCode] = None
    signal_label: Optional[str] = None
    confidence: Optional[str] = None
    summary: Optional[str] = None
    reasons: list[str] = Field(default_factory=list)
    upgrade_trigger: Optional[str] = None
    downgrade_trigger: Optional[str] = None


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
## 多维择时策略契约

`sentiment_score` 表示当前择时机会与风险，不代表公司的长期质量。先按层级判断，再输出唯一主信号；禁止用简单加权让重大风险被其他指标抵消。

优先复用分析上下文中已有且时效有效的数据，仅在维度缺失时调用对应工具，避免重复请求。参考点位无可靠数据时必须标记 N/A，不得编造价格。

1. **数据质量门槛**：检查行情、K 线、财务、资金与新闻的日期和完整性。数据缺失必须明确披露；关键信息缺失时不得输出高置信度的 `accumulate` 或 `exit`。
2. **长期逻辑过滤**：综合盈利增长、现金流、负债、业绩预期、估值与行业地位。长期逻辑明显恶化时，超跌或缩量不能解释为低吸。
3. **市场与行业环境**：结合大盘趋势、波动、行业强弱和个股相对强弱；高波动或行业转弱时降低进攻信号。
4. **价格结构与位置**：判断支撑压力、均线结构、回撤、乖离、形态及止跌或转强确认。
5. **量价、资金与筹码确认**：成交量不能独立决定信号，禁止由单一成交量指标下结论；必须与价格位置、资金流、筹码或相对强弱中的独立证据交叉确认。
6. **事件风险否决**：重大业绩恶化、财务或治理风险、监管处罚、退市风险等优先于普通技术信号。

唯一主信号与兼容字段必须遵守以下映射：
- `exit`（适合清仓）：0-19，`action=sell`，`decision_type=sell`。
- `reduce`（适合减仓）：20-39，`action=reduce`，`decision_type=sell`。
- `watch`（继续观察）：40-49，`action=watch`，`decision_type=hold`。
- `hold`（适合持有）：50-59，`action=hold`，`decision_type=hold`。
- `low_buy`（适合低吸）：60-79，`action=buy`，`decision_type=buy`。
- `accumulate`（适合抢筹）：80-100，`action=buy`，`decision_type=buy`。

冲突处理：下跌缩量但没有止跌结构只能 `watch`；上涨缩量但距离压力位较远且长期逻辑强应为 `hold`；纯技术破位但长期逻辑未失效优先 `reduce` 而非 `exit`；基本面反转或重大硬风险可以否决超跌信号。不得读取或推断个人持仓、成本或仓位比例，不得把未经回测的固定阈值作为绝对触发条件。

`dashboard` 必须包含以下对象：

```json
"strategy_signal": {
    "signal_code": "watch|low_buy|accumulate|hold|reduce|exit",
    "signal_label": "继续观察/适合低吸/适合抢筹/适合持有/适合减仓/适合清仓",
    "confidence": "高/中/低",
    "summary": "一句话综合结论",
    "reasons": ["[基本面] 实际证据", "[价格位置] 实际证据", "[量价资金] 实际证据"],
    "upgrade_trigger": "升级为更积极信号的可验证条件",
    "downgrade_trigger": "降级或失效的可验证条件"
}
```

`reasons` 必须有 3-5 条并覆盖至少三个独立维度，每条引用实际数据、结构或已检索事件；缺失数据要写明，禁止补造数字。`signal_code`、`signal_label`、`sentiment_score`、`operation_advice`、`action` 与 `decision_type` 必须一致。
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
