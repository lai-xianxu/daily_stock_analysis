"""Deterministic OHLCV scenarios for the contrarian cycle timer."""

from datetime import date, datetime, time
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data_provider.realtime_types import RealtimeSource, UnifiedRealtimeQuote
from src.services.timing_state_analyzer import TimingStateAnalyzer, analyze_timing_state


def _ohlcv(close: np.ndarray, volume: np.ndarray, *, upper_wick: float = 0.006) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    open_price = np.r_[close[0], close[:-1]]
    high = np.maximum(open_price, close) * (1 + upper_wick)
    low = np.minimum(open_price, close) * 0.994
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=len(close)),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _decline(*, recent_volume: float, accelerating: bool = False) -> pd.DataFrame:
    head = np.linspace(132, 105, 149)
    if accelerating:
        tail = np.array([104, 103, 102, 101, 100, 99, 96, 92, 87, 81, 74], dtype=float)
    else:
        tail = np.array([104, 102, 100, 98, 96, 94, 93.5, 93, 92.5, 92.2, 92], dtype=float)
    close = np.r_[head, tail]
    volume = np.full(len(close), 1000.0)
    volume[-5:] = recent_volume
    return _ohlcv(close, volume)


def _range_bound() -> pd.DataFrame:
    x = np.linspace(0, 16 * np.pi, 160)
    close = 100 + np.sin(x) * 1.2
    volume = 900 + np.cos(x) * 60
    return _ohlcv(close, volume)


def _low_efficiency_accelerating_selloff() -> pd.DataFrame:
    head = 100 + np.sin(np.linspace(0, 12 * np.pi, 140))
    tail = np.array(
        [100, 99, 98, 97, 96, 96, 97, 99, 102, 106, 110, 114, 118, 122, 126, 120, 113, 107, 102, 99],
        dtype=float,
    )
    return _ohlcv(np.r_[head, tail], np.full(160, 1000.0))


def _healthy_advance() -> pd.DataFrame:
    close = np.linspace(72, 124, 160) + np.sin(np.linspace(0, 8, 160)) * 0.35
    volume = np.full(160, 1000.0)
    return _ohlcv(close, volume)


def _weakening_advance(*, extreme: bool) -> pd.DataFrame:
    if extreme:
        head = np.linspace(70, 112, 149)
        tail = np.array([113, 115, 117, 119, 121, 123, 124, 125, 125.6, 126.0, 126.3])
    else:
        head = np.r_[np.linspace(72, 122, 105), np.full(20, 134.0), np.linspace(130, 105, 24)]
        tail = np.array([111, 113, 116, 119, 122, 123, 123.6, 124.1, 124.5, 124.8, 125.0])
    close = np.r_[head, tail]
    volume = np.full(len(close), 1000.0)
    volume[-5:] = 500.0
    return _ohlcv(close, volume)


def _high_level_breakdown() -> pd.DataFrame:
    head = np.linspace(70, 112, 149)
    tail = np.array([114, 117, 121, 126, 132, 138, 136, 131, 124, 116, 108], dtype=float)
    close = np.r_[head, tail]
    volume = np.full(len(close), 1000.0)
    volume[-5:] = 1900.0
    return _ohlcv(close, volume)


def test_declining_contraction_can_emit_low_buy_before_reversal() -> None:
    state = analyze_timing_state(_decline(recent_volume=650), "TEST").to_dict()

    assert state["phase"] == "declining"
    assert state["suggested_signal"] == "low_buy"
    assert state["volume_state"] == "contraction"
    assert state["metrics"]["volume_contraction_streak"] > 0


def test_extreme_low_and_extreme_contraction_with_exhaustion_emits_accumulate() -> None:
    state = analyze_timing_state(_decline(recent_volume=350), "TEST").to_dict()

    assert state["phase"] == "declining_exhaustion"
    assert state["suggested_signal"] == "accumulate"
    assert state["price_zone"] == "extreme_low"
    assert state["volume_state"] == "extreme_contraction"
    assert state["safety_evidence"]


def test_accelerating_breakdown_cannot_emit_entry_signal() -> None:
    state = analyze_timing_state(
        _decline(recent_volume=1800, accelerating=True),
        "TEST",
    ).to_dict()

    assert state["phase"] == "declining"
    assert state["suggested_signal"] == "watch"
    assert state["momentum_state"] == "accelerating_down"
    assert state["metrics"]["support_distance_atr"] < 0


def test_range_bound_market_emits_watch() -> None:
    state = analyze_timing_state(_range_bound(), "TEST").to_dict()

    assert state["phase"] == "range_bound"
    assert state["suggested_signal"] == "watch"


def test_accelerating_selloff_is_not_misclassified_as_range_bound() -> None:
    state = analyze_timing_state(_low_efficiency_accelerating_selloff(), "TEST").to_dict()

    assert state["metrics"]["efficiency_ratio_20"] < 0.25
    assert state["phase"] == "declining"
    assert state["momentum_state"] == "accelerating_down"
    assert state["suggested_signal"] == "watch"


def test_healthy_advance_emits_hold_even_at_high_price() -> None:
    state = analyze_timing_state(_healthy_advance(), "TEST").to_dict()

    assert state["phase"] == "advancing"
    assert state["suggested_signal"] == "hold"
    assert state["price_zone"] in {"high", "extreme_high"}
    assert state["metrics"]["rsi_12"] == 100


def test_high_advance_with_volume_and_momentum_weakening_emits_reduce() -> None:
    state = analyze_timing_state(_weakening_advance(extreme=False), "TEST").to_dict()

    assert state["phase"] == "advancing_weakening"
    assert state["suggested_signal"] == "reduce"
    assert {"volume", "momentum"} <= set(state["weakening_dimensions"])


def test_extreme_advance_with_only_two_weakening_dimensions_emits_reduce() -> None:
    state = analyze_timing_state(_weakening_advance(extreme=True), "TEST").to_dict()

    assert state["phase"] == "advancing_weakening"
    assert state["suggested_signal"] == "reduce"
    assert len(state["weakening_dimensions"]) >= 2


def test_soft_low_position_can_emit_low_buy_with_extra_confirmation() -> None:
    analyzer = TimingStateAnalyzer()
    metrics = {
        "price_percentile_120": 36.0,
        "price_percentile_240": 42.0,
        "drawdown_60": -0.25,
        "return_60": -0.20,
        "current_price": 90.0,
        "ma20": 95.0,
        "ma60": 100.0,
        "return_5": -0.01,
        "prior_return_5": -0.03,
        "return_20": -0.08,
        "ma20_slope_atr": -0.5,
        "efficiency_ratio_20": 0.5,
        "bb_width_percentile_120": 60.0,
        "ma_spread_atr": 2.0,
        "support_distance_atr": 0.5,
    }
    metrics.update(analyzer._position_profile(metrics))

    phase, _, signal = analyzer._classify(
        metrics=metrics,
        price_zone="mid",
        volume_state="contraction",
        safety_evidence=["[动能] 跌速放缓", "[结构] 接近支撑"],
        weakening_dimensions=[],
    )

    assert metrics["entry_position_strength"] == "soft"
    assert phase == "declining"
    assert signal == "low_buy"


def test_soft_high_position_can_reduce_with_three_dimensions() -> None:
    analyzer = TimingStateAnalyzer()
    metrics = {
        "price_percentile_120": 62.0,
        "price_percentile_240": 68.0,
        "drawdown_60": -0.08,
        "return_60": 0.25,
        "current_price": 110.0,
        "ma20": 105.0,
        "ma60": 98.0,
        "return_5": 0.01,
        "prior_return_5": 0.04,
        "return_20": 0.05,
        "ma20_slope_atr": 0.5,
        "efficiency_ratio_20": 0.5,
        "bb_width_percentile_120": 60.0,
        "ma_spread_atr": 2.0,
    }
    metrics.update(analyzer._position_profile(metrics))

    phase, _, signal = analyzer._classify(
        metrics=metrics,
        price_zone="mid",
        volume_state="contraction",
        safety_evidence=[],
        weakening_dimensions=["volume", "momentum", "industry_market"],
    )

    assert metrics["exit_position_strength"] == "soft"
    assert phase == "advancing_weakening"
    assert signal == "reduce"


def test_high_level_breakdown_is_not_misclassified_as_low_entry_decline() -> None:
    state = analyze_timing_state(_high_level_breakdown(), "TEST").to_dict()

    assert state["phase"] == "high_level_breakdown"
    assert state["suggested_signal"] in {"reduce", "exit"}
    assert len(state["weakening_dimensions"]) >= 3


def test_high_position_accelerating_breakdown_below_extreme_only_reduces() -> None:
    analyzer = TimingStateAnalyzer()
    metrics = {
        "price_percentile_120": 78.75,
        "price_percentile_240": 88.96,
        "drawdown_60": -0.2068,
        "return_60": 0.2957,
        "current_price": 139.88,
        "ma20": 152.0285,
        "ma60": 135.076,
        "return_5": -0.1419,
        "prior_return_5": 0.1619,
        "return_20": -0.0058,
        "ma20_slope_atr": 0.446,
        "efficiency_ratio_20": 0.0466,
        "bb_width_percentile_120": 55.42,
        "ma_spread_atr": 1.388,
        "support_distance_atr": 0.5731,
        "macd_bar_slope_5": -5.6719,
        "negative_divergence": False,
        "rsi_12": 43.3745,
        "rsi_slope_5": -17.7153,
        "atr_percentile_120": 99.58,
        "atr_change_5": 1.0965,
        "upper_wick_ratio": 0.3335,
    }
    metrics.update(analyzer._position_profile(metrics))
    weakening = analyzer._advance_weakening_evidence(metrics, "expansion")
    dimensions = list(dict.fromkeys(item[0] for item in weakening))

    phase, momentum, signal = analyzer._classify(
        metrics=metrics,
        price_zone="high",
        volume_state="expansion",
        safety_evidence=["[结构] 接近20日支撑"],
        weakening_dimensions=dimensions,
    )

    assert metrics["blended_price_percentile"] < 90
    assert set(dimensions) == {"momentum", "structure", "volume", "volatility"}
    assert phase == "high_level_breakdown"
    assert momentum == "breakdown_from_high"
    assert signal == "reduce"


def test_extreme_high_three_dimension_exhaustion_requires_full_data_for_exit() -> None:
    analyzer = TimingStateAnalyzer()
    metrics = {
        "current_price": 120.0,
        "ma20": 110.0,
        "return_5": 0.02,
        "return_20": 0.15,
        "ma20_slope_atr": 1.0,
        "efficiency_ratio_20": 0.6,
        "bb_width_percentile_120": 60.0,
        "ma_spread_atr": 2.0,
        "exit_position_strength": "extreme",
    }
    dimensions = ["volume", "momentum", "structure"]

    full_phase, _, full_signal = analyzer._classify(
        metrics=metrics,
        price_zone="extreme_high",
        volume_state="contraction",
        safety_evidence=[],
        weakening_dimensions=dimensions,
        data_quality="full",
    )
    partial_phase, _, partial_signal = analyzer._classify(
        metrics=metrics,
        price_zone="extreme_high",
        volume_state="contraction",
        safety_evidence=[],
        weakening_dimensions=dimensions,
        data_quality="partial",
    )

    assert (full_phase, full_signal) == ("advancing_exhaustion", "exit")
    assert (partial_phase, partial_signal) == ("advancing_weakening", "reduce")


def test_target_date_excludes_later_intraday_or_future_bar() -> None:
    frame = _healthy_advance()
    completed = frame.iloc[-2]["date"].date()
    frame.loc[frame.index[-1], ["close", "high", "volume"]] = [10_000, 10_010, 10_000_000]

    state = analyze_timing_state(frame, "TEST", target_date=completed).to_dict()

    assert state["completed_bar_date"] == completed.isoformat()
    assert state["metrics"]["bar_count"] == len(frame) - 1
    assert state["metrics"]["current_price"] < 200


def test_insufficient_history_is_unknown_watch() -> None:
    frame = _healthy_advance().tail(40)
    state = analyze_timing_state(frame, "TEST", target_date=date(2026, 12, 31)).to_dict()

    assert state["phase"] == "unknown"
    assert state["suggested_signal"] == "watch"
    assert state["data_quality"] == "insufficient"


def test_missing_recent_volume_is_unknown_instead_of_normal_volume() -> None:
    frame = _healthy_advance()
    frame.loc[frame.index[-5:], "volume"] = np.nan

    state = analyze_timing_state(frame, "TEST").to_dict()

    assert state["volume_state"] == "unknown"
    assert any("成交量数据不可用" in item for item in state["limitations"])


def test_pipeline_history_failure_falls_back_to_unknown_watch_not_legacy_score() -> None:
    from src.core.pipeline import StockAnalysisPipeline

    pipeline = object.__new__(StockAnalysisPipeline)
    with (
        patch("src.services.history_loader.get_frozen_target_date", return_value=date(2026, 7, 16)),
        patch("src.services.history_loader.load_history_df", side_effect=RuntimeError("offline")),
    ):
        state = pipeline._load_timing_state("600519")

    assert state is not None
    assert state["phase"] == "unknown"
    assert state["suggested_signal"] == "watch"
    assert state["source"] == "timing_analysis_error"


def test_pipeline_uses_validated_postmarket_quote_as_completed_daily_bar() -> None:
    from src.core.pipeline import StockAnalysisPipeline

    frame = _healthy_advance()
    last_close = float(frame.iloc[-1]["close"])
    target = (frame.iloc[-1]["date"] + pd.offsets.BDay(1)).date()
    quote = UnifiedRealtimeQuote(
        code="600519",
        source=RealtimeSource.TENCENT,
        provider_timestamp=datetime.combine(
            target,
            time(15, 30),
            tzinfo=ZoneInfo("Asia/Shanghai"),
        ).isoformat(),
        price=last_close * 1.01,
        pre_close=last_close,
        open_price=last_close,
        high=last_close * 1.02,
        low=last_close * 0.99,
        volume=1_200_000,
        amount=150_000_000,
        change_pct=1.0,
        # A final close can exceed the intraday TTL by the time a long batch
        # reaches this stock; its post-market provider timestamp remains valid.
        is_stale=True,
    )
    pipeline = object.__new__(StockAnalysisPipeline)

    with (
        patch("src.services.history_loader.get_frozen_target_date", return_value=target),
        patch("src.services.history_loader.load_history_df", return_value=(frame, "fixture")),
        patch("src.core.pipeline.build_market_phase_context") as provider_phase,
    ):
        provider_phase.return_value.phase.value = "postmarket"
        state = pipeline._load_timing_state(
            "600519",
            realtime_quote=quote,
            market_phase="postmarket",
        )

    assert state is not None
    assert state["completed_bar_date"] == target.isoformat()
    assert state["metrics"]["current_price"] == round(quote.price, 4)
    assert state["completed_bar_source"] == "realtime_close"
    assert state["data_freshness"] == "current"
    assert state["source"] == "fixture+realtime_close"


def test_pipeline_rejects_stale_quote_for_completed_daily_bar() -> None:
    from src.core.pipeline import StockAnalysisPipeline

    frame = _healthy_advance()
    target = (frame.iloc[-1]["date"] + pd.offsets.BDay(1)).date()
    quote = UnifiedRealtimeQuote(
        code="600519",
        source=RealtimeSource.TENCENT,
        price=125.0,
        pre_close=float(frame.iloc[-1]["close"]),
        open_price=124.0,
        high=126.0,
        low=123.0,
        volume=1_000_000,
        is_stale=True,
    )

    _, used_quote, reason = StockAnalysisPipeline._overlay_completed_realtime_bar(
        frame,
        quote,
        code="600519",
        target_date=target,
        market_phase="postmarket",
    )

    assert used_quote is False
    assert reason == "realtime_quote_stale"
