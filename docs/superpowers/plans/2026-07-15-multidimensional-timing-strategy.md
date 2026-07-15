# Multidimensional Timing Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the default stock analysis to a fundamentals-constrained, multidimensional timing strategy that emits one of six primary signals and presents evidence-first reasons in every report format.

**Architecture:** Keep the existing prompt-driven analysis pipeline and optional dashboard contract. Add a small six-state mapping module used by both parser paths, inject the same contract into normal and Agent prompts, and make the report renderer prefer `dashboard.strategy_signal` while retaining every legacy section when that object is absent.

**Tech Stack:** Python 3.11, Pydantic v2, PyYAML, Jinja2, pytest/unittest, GitHub Actions.

## Global Constraints

- The only default-active and default-router skill remains `volume_contraction_timing`.
- Do not read or infer personal holdings, cost basis, or position percentages.
- Volume, one technical indicator, or one trading day must never independently determine a signal.
- `exit` requires a fundamental thesis break, hard event risk, or severe structural damage with independent confirmation.
- Existing reports without `dashboard.strategy_signal` must render exactly through the legacy path.
- No new data provider or standalone quantitative trading engine is introduced.

---

### Task 1: Define The Six-State Compatibility Contract

**Files:**
- Create: `src/schemas/strategy_signal.py`
- Modify: `src/schemas/decision_scale.py`
- Test: `tests/test_multidimensional_timing_strategy.py`
- Test: `tests/test_decision_scale.py` or the existing decision-scale assertions discovered by `rg`

**Interfaces:**
- Produces: `StrategySignalCode`, `strategy_signal_definition(code)`, `normalize_strategy_signal_payload(payload, language)`, and `align_score_to_strategy_signal(code, score)`.
- Produces: canonical score bands `exit=0-19`, `reduce=20-39`, `watch=40-49`, `hold=50-59`, `low_buy=60-79`, `accumulate=80-100`.

- [ ] **Step 1: Write failing mapping tests**

```python
def test_six_strategy_signals_map_to_score_action_and_decision_type():
    expected = {
        "exit": ((0, 19), "sell", "sell"),
        "reduce": ((20, 39), "reduce", "sell"),
        "watch": ((40, 49), "watch", "hold"),
        "hold": ((50, 59), "hold", "hold"),
        "low_buy": ((60, 79), "buy", "buy"),
        "accumulate": ((80, 100), "buy", "buy"),
    }
    for code, (score_range, action, decision_type) in expected.items():
        definition = strategy_signal_definition(code)
        assert (definition.min_score, definition.max_score) == score_range
        assert definition.action == action
        assert definition.decision_type == decision_type
```

- [ ] **Step 2: Run the mapping test and verify RED**

Run: `pytest -q tests/test_multidimensional_timing_strategy.py`

Expected: import failure because `src.schemas.strategy_signal` does not exist.

- [ ] **Step 3: Implement the minimal mapping and split the neutral canonical band**

Implement immutable definitions for the six codes, canonical labels, score bounds, actions, and decision types. Change the existing canonical scale from one `40-59 watch` band to `40-49 watch` and `50-59 hold`; leave all directional bands unchanged.

- [ ] **Step 4: Run mapping and existing decision-scale tests**

Run: `pytest -q tests/test_multidimensional_timing_strategy.py tests/test_stock_analyzer_bias.py tests/test_decision_action.py`

Expected: all selected tests pass after updating directly affected score-53 expectations from watch to hold.

### Task 2: Replace The Volume-Only Skill With The Approved Multidimensional Rules

**Files:**
- Modify: `strategies/volume_contraction_timing.yaml`
- Modify: `src/analyzer.py`
- Modify: `src/agent/executor.py`
- Test: `tests/test_multidimensional_timing_strategy.py`
- Test: `tests/test_agent_executor.py`

**Interfaces:**
- Consumes: six-state contract from Task 1.
- Produces: identical required signal codes, mappings, evidence rules, and risk vetoes in the normal and Agent JSON prompts.

- [ ] **Step 1: Write failing strategy and prompt tests**

```python
def test_default_strategy_is_multidimensional_and_not_position_personalized():
    skill = load_skill_from_yaml("strategies/volume_contraction_timing.yaml")
    assert skill.default_active and skill.default_router
    for tool in REQUIRED_MULTIDIMENSIONAL_TOOLS:
        assert tool in skill.required_tools
    for phrase in ("数据质量门槛", "长期逻辑过滤", "事件风险否决", "成交量不能独立决定"):
        assert phrase in skill.instructions
    assert "建议仓位：20%-30%" not in skill.instructions

def test_normal_and_agent_prompts_require_same_six_state_contract():
    for prompt in (GeminiAnalyzer.SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT):
        for code in ("watch", "low_buy", "accumulate", "hold", "reduce", "exit"):
            assert code in prompt
        assert '"strategy_signal"' in prompt
        assert "3-5" in prompt
        assert "单一成交量" in prompt
        assert "事件风险否决" in prompt
```

- [ ] **Step 2: Run the prompt tests and verify RED**

Run: `pytest -q tests/test_multidimensional_timing_strategy.py tests/test_agent_executor.py -k "multidimensional or strategy_signal"`

Expected: assertions fail because the YAML and prompts still expose the old volume/position framing.

- [ ] **Step 3: Rewrite only the default strategy instructions**

Replace fixed position percentages and single-factor score adjustments with the six ordered gates: data quality, long-term thesis, market/sector regime, price structure, volume/capital/chip confirmation, and hard-event veto. Include the approved conflict rules and all existing tool names, but no personal-position assumptions.

- [ ] **Step 4: Add `strategy_signal` to normal and Agent JSON contracts**

Add the exact optional dashboard object:

```json
"strategy_signal": {
  "signal_code": "watch|low_buy|accumulate|hold|reduce|exit",
  "signal_label": "继续观察/适合低吸/适合抢筹/适合持有/适合减仓/适合清仓",
  "confidence": "高/中/低",
  "summary": "一句话综合结论",
  "reasons": ["[基本面] ...", "[价格位置] ...", "[量价资金] ..."],
  "upgrade_trigger": "可验证升级条件",
  "downgrade_trigger": "可验证降级或失效条件"
}
```

Require 3-5 evidence-bearing reasons across at least three dimensions, explicit missing-data disclosure, and signal/score/action consistency.

- [ ] **Step 5: Run prompt and default-skill tests**

Run: `pytest -q tests/test_multidimensional_timing_strategy.py tests/test_agent_executor.py tests/test_agent_registry.py tests/test_agent_pipeline.py`

Expected: all selected tests pass and only `volume_contraction_timing` resolves as the primary default.

### Task 3: Parse And Enforce Strategy-Signal Consistency

**Files:**
- Modify: `src/schemas/report_schema.py`
- Modify: `src/analyzer.py`
- Test: `tests/test_report_schema.py`
- Test: `tests/test_multidimensional_timing_strategy.py`

**Interfaces:**
- Consumes: mapping helpers from Task 1.
- Produces: optional Pydantic `StrategySignal` under `Dashboard.strategy_signal`.
- Produces: shared `populate_decision_action_fields()` behavior for both normal and Agent conversion paths.

- [ ] **Step 1: Write failing schema and parser tests**

```python
def test_schema_accepts_complete_and_missing_strategy_signal():
    complete = AnalysisReportSchema.model_validate(COMPLETE_STRATEGY_REPORT)
    assert complete.dashboard.strategy_signal.signal_code == "low_buy"
    legacy = AnalysisReportSchema.model_validate(LEGACY_REPORT)
    assert legacy.dashboard.strategy_signal is None

def test_parser_uses_strategy_signal_as_primary_compatible_action():
    result = analyzer._parse_response(json.dumps(LOW_BUY_WITH_CONFLICTING_LEGACY_FIELDS), "600519", "贵州茅台")
    assert result.sentiment_score == 60
    assert result.operation_advice == "适合低吸"
    assert result.action == "buy"
    assert result.decision_type == "buy"
```

- [ ] **Step 2: Run schema/parser tests and verify RED**

Run: `pytest -q tests/test_report_schema.py tests/test_multidimensional_timing_strategy.py -k "strategy_signal or six_strategy"`

Expected: schema attribute or parser consistency assertions fail.

- [ ] **Step 3: Add the optional Pydantic object**

Use `Literal` for `signal_code`, optional strings for display fields, and a list for reasons. Keep the entire object optional so historical reports remain valid.

- [ ] **Step 4: Enforce compatibility in the shared result finalizer**

When a valid strategy signal exists, normalize its label, bound the score to that signal's interval, record any score adjustment in `dashboard.decision_score_calibration`, and set compatible `operation_advice`, `action`, and `decision_type`. When absent or invalid, preserve the current behavior.

- [ ] **Step 5: Run ordinary and Agent conversion tests**

Run: `pytest -q tests/test_report_schema.py tests/test_multidimensional_timing_strategy.py tests/test_agent_pipeline.py -k "strategy_signal or agent_result_to_analysis_result or parse_response"`

Expected: both parser paths produce consistent six-state and legacy fields.

### Task 4: Make All Three Reports Prefer The New Signal

**Files:**
- Modify: `src/report_language.py`
- Modify: `src/services/report_renderer.py`
- Modify: `templates/report_markdown.j2`
- Modify: `templates/report_wechat.j2`
- Modify: `templates/report_brief.j2`
- Test: `tests/test_report_renderer.py`
- Test: `tests/test_report_language.py`

**Interfaces:**
- Consumes: normalized `dashboard.strategy_signal` from Task 3.
- Produces: enriched template entry `strategy_signal` and six-state summary text.

- [ ] **Step 1: Write failing report tests for all platforms and legacy fallback**

```python
def test_all_reports_prioritize_strategy_signal_and_keep_legacy_sections():
    for platform in ("markdown", "wechat", "brief"):
        output = render(platform, [_make_strategy_result()])
        assert "适合低吸" in output
        assert "盈利与现金流未见恶化" in output
    markdown = render("markdown", [_make_strategy_result()], summary_only=False)
    assert markdown.index("综合策略判断") < markdown.index("重要信息速览")
    assert "行情" in markdown and "量能" in markdown and "作战计划" in markdown

def test_reports_without_strategy_signal_keep_legacy_output():
    output = render("markdown", [_make_result()], summary_only=False)
    assert "综合策略判断" not in output
    assert "核心结论" in output
```

- [ ] **Step 2: Run report tests and verify RED**

Run: `pytest -q tests/test_report_renderer.py -k "strategy_signal"`

Expected: the new heading, label, and evidence are absent.

- [ ] **Step 3: Add localized report labels and renderer context**

Add labels for strategy heading, confidence, reasons, upgrade trigger, and downgrade/invalid trigger. Rename only the display labels for the existing strongest bullish/bearish fields to “主要支持因素/主要风险因素”; keep their JSON keys unchanged.

- [ ] **Step 4: Update all templates conditionally**

Render strategy signal, confidence, summary, reasons, and triggers before old detailed sections. Use the old `e.signal_text` and old layout unchanged when `strategy_signal` is missing. Keep market snapshot, news, averages/price, volume, chips, phase guardrail, battle plan, reference prices, checklist, attribution, and history sections.

- [ ] **Step 5: Run renderer and language tests**

Run: `pytest -q tests/test_report_renderer.py tests/test_report_language.py tests/test_notification_report_fixtures.py`

Expected: all selected tests pass in Chinese, English, and Korean report modes.

### Task 5: Full Verification And Main Push

**Files:**
- Verify all modified files and generated reports.
- No new production scope.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: verified commits on `main` and matching `origin/main`.

- [ ] **Step 1: Run focused verification**

Run: `pytest -q tests/test_multidimensional_timing_strategy.py tests/test_report_schema.py tests/test_report_renderer.py tests/test_report_language.py tests/test_agent_executor.py tests/test_agent_registry.py tests/test_agent_pipeline.py tests/test_stock_analyzer_bias.py`

Expected: zero failures.

- [ ] **Step 2: Run the repository's broader test gate and static checks discovered from project configuration**

Run the configured pytest suite or the largest feasible CI-equivalent groups, plus YAML parsing and Python compilation for touched modules. Record any unrelated pre-existing failure separately; do not claim completion while an introduced failure remains.

- [ ] **Step 3: Render representative new and legacy reports**

Generate markdown, wechat, and brief output from deterministic fixtures. Assert the new signal comes first, reasons and triggers are present, old sections remain, and legacy fixtures have no new empty heading.

- [ ] **Step 4: Review the diff requirement by requirement**

Run: `git diff --check && git diff --stat origin/main...HEAD && git status --short`

Expected: no whitespace errors, only approved files, and no unrelated generated artifacts.

- [ ] **Step 5: Commit and push main**

Create focused conventional commits, fetch/rebase only if the remote advanced, then run `git push origin main`.

- [ ] **Step 6: Verify remote state**

Run: `git fetch origin main && git rev-parse HEAD && git rev-parse origin/main && git status --short --branch`

Expected: local and remote SHAs match and `main` is not ahead or behind.
