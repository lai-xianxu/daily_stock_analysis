# -*- coding: utf-8 -*-
"""Deterministic market-cycle timing analysis from completed daily bars."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


MIN_BARS = 60
FULL_DATA_BARS = 120
MAX_LOOKBACK_BARS = 240


@dataclass
class TimingStateResult:
    """Machine-computed cycle state used to constrain the LLM decision."""

    code: str
    phase: str = "unknown"
    price_zone: str = "mid"
    volume_state: str = "unknown"
    momentum_state: str = "neutral"
    data_quality: str = "insufficient"
    suggested_signal: str = "watch"
    confidence: str = "低"
    summary: str = "历史数据不足，继续观察"
    completed_bar_date: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    safety_evidence: List[str] = field(default_factory=list)
    weakening_evidence: List[str] = field(default_factory=list)
    weakening_dimensions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    reference_points: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    return value


def _percentile_rank(values: pd.Series, current: Optional[float] = None) -> Optional[float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    value = float(clean.iloc[-1] if current is None else current)
    less = float((clean < value).sum())
    equal = float((clean == value).sum())
    return round((less + equal * 0.5) / len(clean) * 100, 2)


def _period_return(close: pd.Series, periods: int) -> Optional[float]:
    if len(close) <= periods:
        return None
    base = float(close.iloc[-periods - 1])
    if not np.isfinite(base) or base <= 0:
        return None
    return float(close.iloc[-1] / base - 1.0)


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return None
    return float(numerator / denominator)


def _fmt_price(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value) or value <= 0:
        return "N/A"
    return f"{value:.2f}"


def _fmt_zone(first: Optional[float], second: Optional[float]) -> str:
    values = [float(item) for item in (first, second) if item is not None and np.isfinite(item) and item > 0]
    if not values:
        return "N/A"
    low, high = min(values), max(values)
    if abs(high - low) / low < 0.005:
        return _fmt_price(low)
    return f"{low:.2f}-{high:.2f}"


class TimingStateAnalyzer:
    """Classify decline, range, advance, weakening, and exhaustion states."""

    def analyze(
        self,
        df: Optional[pd.DataFrame],
        code: str,
        *,
        target_date: Optional[date] = None,
    ) -> TimingStateResult:
        prepared = self._prepare(df, target_date=target_date)
        if prepared is None or prepared.empty:
            return TimingStateResult(
                code=code,
                limitations=["缺少有效的已完成日线数据"],
            )

        prepared = prepared.tail(MAX_LOOKBACK_BARS).reset_index(drop=True)
        bar_count = len(prepared)
        completed_bar_date = self._completed_bar_date(prepared)
        if bar_count < MIN_BARS:
            return TimingStateResult(
                code=code,
                completed_bar_date=completed_bar_date,
                limitations=[f"仅有{bar_count}个有效交易日，至少需要{MIN_BARS}日"],
                metrics={"bar_count": bar_count},
            )

        data_quality = "full" if bar_count >= FULL_DATA_BARS else "partial"
        metrics = self._calculate_metrics(prepared)
        price_zone = self._price_zone(metrics.get("price_percentile_120"))
        volume_state = self._volume_state(metrics)

        safety_evidence = self._decline_safety_evidence(metrics)
        weakening = self._advance_weakening_evidence(metrics, volume_state)
        weakening_dimensions = list(dict.fromkeys(item[0] for item in weakening))
        weakening_evidence = [item[1] for item in weakening]

        phase, momentum_state, suggested_signal = self._classify(
            metrics=metrics,
            price_zone=price_zone,
            volume_state=volume_state,
            safety_evidence=safety_evidence,
            weakening_dimensions=weakening_dimensions,
        )
        confidence = self._confidence(
            phase=phase,
            data_quality=data_quality,
            safety_count=len(safety_evidence),
            weakening_dimension_count=len(weakening_dimensions),
        )
        evidence = self._evidence(
            metrics=metrics,
            phase=phase,
            price_zone=price_zone,
            volume_state=volume_state,
            safety_evidence=safety_evidence,
            weakening_evidence=weakening_evidence,
        )
        limitations: List[str] = []
        if data_quality == "partial":
            limitations.append(f"仅有{bar_count}个交易日，长期分位按可用样本计算")
        if volume_state == "unknown":
            limitations.append("成交量数据不可用，进攻或退出信号置信度受限")

        return TimingStateResult(
            code=code,
            phase=phase,
            price_zone=price_zone,
            volume_state=volume_state,
            momentum_state=momentum_state,
            data_quality=data_quality,
            suggested_signal=suggested_signal,
            confidence=confidence,
            summary=self._summary(phase, suggested_signal),
            completed_bar_date=completed_bar_date,
            evidence=evidence[:8],
            safety_evidence=safety_evidence,
            weakening_evidence=weakening_evidence,
            weakening_dimensions=weakening_dimensions,
            limitations=limitations,
            metrics=metrics,
            reference_points=self._reference_points(prepared, metrics, suggested_signal),
        )

    @staticmethod
    def _prepare(
        df: Optional[pd.DataFrame],
        *,
        target_date: Optional[date],
    ) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return None

        clean = df.copy()
        clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
        clean = clean.dropna(subset=["date"])
        if target_date is not None:
            clean = clean[clean["date"].dt.date <= target_date]
        for column in ("open", "high", "low", "close", "volume"):
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean = clean.dropna(subset=["open", "high", "low", "close"])
        clean = clean[clean["close"] > 0]
        clean = clean.sort_values("date").drop_duplicates("date", keep="last")
        return clean

    @staticmethod
    def _completed_bar_date(df: pd.DataFrame) -> Optional[str]:
        if df.empty:
            return None
        value = df.iloc[-1]["date"]
        return value.date().isoformat() if isinstance(value, pd.Timestamp) else str(value)[:10]

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_price = df["open"].astype(float)
        volume = pd.to_numeric(df["volume"], errors="coerce").astype(float)

        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = true_range.ewm(alpha=1 / 14, adjust=False).mean()
        atr_now = float(atr14.iloc[-1])

        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.ewm(alpha=1 / 12, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1 / 12, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi12 = 100 - 100 / (1 + rs)
        rsi12 = rsi12.mask((avg_loss == 0) & (avg_gain > 0), 100)
        rsi12 = rsi12.mask((avg_gain == 0) & (avg_loss > 0), 0).fillna(50)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_dif = ema12 - ema26
        macd_dea = macd_dif.ewm(span=9, adjust=False).mean()
        macd_bar = (macd_dif - macd_dea) * 2

        rolling_std20 = close.rolling(20).std(ddof=0)
        bb_width = (rolling_std20 * 4 / ma20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

        price_window_120 = close.tail(min(120, len(close)))
        price_window_240 = close.tail(min(240, len(close)))
        volume_ma5 = volume.rolling(5).mean()
        volume_pct = _percentile_rank(volume_ma5.tail(min(120, len(volume_ma5))))
        volume_contraction_streak = 0
        for delta_value in volume_ma5.diff().tail(20).iloc[::-1]:
            if pd.notna(delta_value) and float(delta_value) < 0:
                volume_contraction_streak += 1
            else:
                break
        recent_volume = float(volume.tail(5).mean())
        prior_volume_window = volume.iloc[-25:-5] if len(volume) >= 25 else volume.iloc[:-5]
        prior_volume = float(prior_volume_window.mean()) if not prior_volume_window.empty else float("nan")
        volume_ratio = _safe_ratio(recent_volume, prior_volume)

        direction = close.diff()
        last20 = df.tail(20).copy()
        last20_direction = direction.tail(20)
        up_volume = pd.to_numeric(last20.loc[last20_direction > 0, "volume"], errors="coerce")
        down_volume = pd.to_numeric(last20.loc[last20_direction < 0, "volume"], errors="coerce")
        up_volume_mean = float(up_volume.mean()) if not up_volume.empty else float("nan")
        down_volume_mean = float(down_volume.mean()) if not down_volume.empty else float("nan")
        down_up_volume_ratio = _safe_ratio(down_volume_mean, up_volume_mean)

        ma20_slope_atr = _safe_ratio(float(ma20.iloc[-1] - ma20.iloc[-6]), atr_now)
        ma60_slope_atr = _safe_ratio(float(ma60.iloc[-1] - ma60.iloc[-11]), atr_now)
        ma_spread_atr = _safe_ratio(abs(float(ma20.iloc[-1] - ma60.iloc[-1])), atr_now)
        return_5 = _period_return(close, 5)
        return_20 = _period_return(close, 20)
        return_60 = _period_return(close, 60)
        prior_return_5 = None
        if len(close) >= 11 and close.iloc[-11] > 0:
            prior_return_5 = float(close.iloc[-6] / close.iloc[-11] - 1)

        movement = float(close.tail(20).diff().abs().sum())
        efficiency_ratio_20 = _safe_ratio(abs(float(close.iloc[-1] - close.iloc[-20])), movement)
        rsi_slope_5 = float(rsi12.iloc[-1] - rsi12.iloc[-6])
        macd_bar_slope_5 = float(macd_bar.iloc[-1] - macd_bar.iloc[-6])
        atr_change_5 = _safe_ratio(float(atr14.iloc[-1]), float(atr14.iloc[-6]))
        atr_percentile = _percentile_rank(atr14.tail(min(120, len(atr14))))
        bb_width_percentile = _percentile_rank(bb_width.tail(min(120, len(bb_width))))

        positive_divergence, negative_divergence = self._momentum_divergences(close, rsi12, macd_bar)
        candle_range = float(high.iloc[-1] - low.iloc[-1])
        upper_wick_ratio = 0.0
        if candle_range > 0:
            upper_wick_ratio = float((high.iloc[-1] - max(open_price.iloc[-1], close.iloc[-1])) / candle_range)

        low20 = float(low.tail(20).min())
        prior_low20_window = low.iloc[-21:-1] if len(low) >= 21 else low.iloc[:-1]
        prior_low20 = float(prior_low20_window.min()) if not prior_low20_window.empty else low20
        high20 = float(high.tail(20).max())
        low60 = float(low.tail(60).min())
        high60 = float(high.tail(60).max())
        current = float(close.iloc[-1])
        drawdown_60 = float(current / high60 - 1) if high60 > 0 else None
        support_distance_atr = _safe_ratio(current - prior_low20, atr_now)

        return {
            "bar_count": len(df),
            "current_price": current,
            "price_percentile_120": _percentile_rank(price_window_120),
            "price_percentile_240": _percentile_rank(price_window_240),
            "return_5": return_5,
            "prior_return_5": prior_return_5,
            "return_20": return_20,
            "return_60": return_60,
            "drawdown_60": drawdown_60,
            "ma20": float(ma20.iloc[-1]),
            "ma60": float(ma60.iloc[-1]),
            "ma20_slope_atr": ma20_slope_atr,
            "ma60_slope_atr": ma60_slope_atr,
            "ma_spread_atr": ma_spread_atr,
            "volume_percentile_120": volume_pct,
            "volume_ratio_5_20": volume_ratio,
            "volume_contraction_streak": volume_contraction_streak,
            "down_up_volume_ratio_20": down_up_volume_ratio,
            "rsi_12": float(rsi12.iloc[-1]),
            "rsi_slope_5": rsi_slope_5,
            "macd_bar": float(macd_bar.iloc[-1]),
            "macd_bar_slope_5": macd_bar_slope_5,
            "positive_divergence": positive_divergence,
            "negative_divergence": negative_divergence,
            "atr_14": atr_now,
            "atr_percentile_120": atr_percentile,
            "atr_change_5": atr_change_5,
            "bb_width_percentile_120": bb_width_percentile,
            "efficiency_ratio_20": efficiency_ratio_20,
            "upper_wick_ratio": upper_wick_ratio,
            "support_20": low20,
            "prior_support_20": prior_low20,
            "resistance_20": high20,
            "support_60": low60,
            "resistance_60": high60,
            "support_distance_atr": support_distance_atr,
        }

    @staticmethod
    def _momentum_divergences(
        close: pd.Series,
        rsi: pd.Series,
        macd_bar: pd.Series,
    ) -> tuple[bool, bool]:
        if len(close) < 25:
            return False, False
        previous = close.iloc[-25:-5]
        recent = close.iloc[-5:]
        if previous.empty or recent.empty:
            return False, False

        prev_low_idx = previous.idxmin()
        recent_low_idx = recent.idxmin()
        prev_high_idx = previous.idxmax()
        recent_high_idx = recent.idxmax()
        positive = bool(
            close.loc[recent_low_idx] <= close.loc[prev_low_idx] * 1.01
            and (
                rsi.loc[recent_low_idx] >= rsi.loc[prev_low_idx] + 3
                or macd_bar.loc[recent_low_idx] > macd_bar.loc[prev_low_idx]
            )
        )
        negative = bool(
            close.loc[recent_high_idx] >= close.loc[prev_high_idx] * 0.99
            and (
                rsi.loc[recent_high_idx] <= rsi.loc[prev_high_idx] - 3
                or macd_bar.loc[recent_high_idx] < macd_bar.loc[prev_high_idx]
            )
        )
        return positive, negative

    @staticmethod
    def _price_zone(percentile: Optional[float]) -> str:
        if percentile is None:
            return "mid"
        if percentile <= 10:
            return "extreme_low"
        if percentile <= 30:
            return "low"
        if percentile < 70:
            return "mid"
        if percentile < 90:
            return "high"
        return "extreme_high"

    @staticmethod
    def _volume_state(metrics: Dict[str, Any]) -> str:
        percentile = metrics.get("volume_percentile_120")
        ratio = metrics.get("volume_ratio_5_20")
        if percentile is None or ratio is None:
            return "unknown"
        if percentile <= 10 and ratio <= 0.60:
            return "extreme_contraction"
        if percentile <= 30 and ratio <= 0.80:
            return "contraction"
        if percentile >= 90 and ratio >= 1.50:
            return "climax"
        if percentile >= 70 or ratio >= 1.20:
            return "expansion"
        return "normal"

    @staticmethod
    def _decline_safety_evidence(metrics: Dict[str, Any]) -> List[str]:
        evidence: List[str] = []
        return_5 = metrics.get("return_5")
        prior_return_5 = metrics.get("prior_return_5")
        if return_5 is not None and prior_return_5 is not None and return_5 > prior_return_5 + 0.005:
            evidence.append("[动能] 最近5日跌速较前5日放缓")
        if metrics.get("positive_divergence"):
            evidence.append("[动能] 价格低点附近出现RSI或MACD底背离")
        if metrics.get("rsi_12", 50) <= 40 and metrics.get("rsi_slope_5", 0) > 2:
            evidence.append("[动能] RSI处于弱势区但已开始回升")
        if metrics.get("macd_bar_slope_5", 0) > 0:
            evidence.append("[动能] MACD柱较5日前改善")
        atr_change = metrics.get("atr_change_5")
        if atr_change is not None and atr_change <= 0.95:
            evidence.append("[波动] ATR回落，波动正在收敛")
        support_distance = metrics.get("support_distance_atr")
        if support_distance is not None and support_distance <= 1.0:
            evidence.append("[结构] 价格接近20日低点支撑区")
        down_up_ratio = metrics.get("down_up_volume_ratio_20")
        if down_up_ratio is not None and down_up_ratio <= 0.90:
            evidence.append("[承接] 近20日下跌日均量低于上涨日")
        return evidence

    @staticmethod
    def _advance_weakening_evidence(
        metrics: Dict[str, Any],
        volume_state: str,
    ) -> List[tuple[str, str]]:
        evidence: List[tuple[str, str]] = []
        if volume_state in {"contraction", "extreme_contraction"}:
            evidence.append(("volume", "[量能] 上涨阶段5日均量处于收缩区"))
        elif volume_state == "climax":
            evidence.append(("volume", "[量能] 成交量处于历史极高分位，存在高潮量风险"))

        return_5 = metrics.get("return_5")
        prior_return_5 = metrics.get("prior_return_5")
        momentum_weak = False
        if return_5 is not None and prior_return_5 is not None and return_5 < prior_return_5 - 0.01:
            momentum_weak = True
        if metrics.get("macd_bar_slope_5", 0) < 0:
            momentum_weak = True
        if metrics.get("negative_divergence"):
            momentum_weak = True
        if metrics.get("rsi_12", 50) >= 70 and metrics.get("rsi_slope_5", 0) < -2:
            momentum_weak = True
        if momentum_weak:
            evidence.append(("momentum", "[动能] ROC、RSI或MACD显示上涨动能衰减"))

        atr_percentile = metrics.get("atr_percentile_120")
        atr_change = metrics.get("atr_change_5")
        upper_wick = metrics.get("upper_wick_ratio", 0)
        if (
            atr_percentile is not None
            and atr_percentile >= 80
            and atr_change is not None
            and atr_change >= 1.10
            and upper_wick >= 0.35
        ):
            evidence.append(("volatility", "[波动] 高波动伴随明显上影，冲高承接减弱"))
        return evidence

    @staticmethod
    def _is_decline_accelerating(metrics: Dict[str, Any], volume_state: str) -> bool:
        return_5 = metrics.get("return_5")
        prior_return_5 = metrics.get("prior_return_5")
        if return_5 is None or prior_return_5 is None:
            return False
        price_break = metrics.get("support_distance_atr") is not None and metrics["support_distance_atr"] < -0.25
        faster = return_5 <= -0.02 and return_5 < prior_return_5 - 0.01
        return bool(faster or (price_break and volume_state in {"expansion", "climax"}))

    def _classify(
        self,
        *,
        metrics: Dict[str, Any],
        price_zone: str,
        volume_state: str,
        safety_evidence: List[str],
        weakening_dimensions: List[str],
    ) -> tuple[str, str, str]:
        current = metrics.get("current_price") or 0
        ma20 = metrics.get("ma20") or current
        return_5 = metrics.get("return_5") or 0
        return_20 = metrics.get("return_20") or 0
        ma20_slope = metrics.get("ma20_slope_atr") or 0

        decline = (return_20 < -0.01 and ma20_slope < 0) or (current < ma20 and return_5 < 0)
        advance = (return_20 > 0.01 and ma20_slope > 0) or (current > ma20 and return_5 > 0)
        low_efficiency = (metrics.get("efficiency_ratio_20") or 1) <= 0.25
        range_confirmation = any(
            (
                abs(ma20_slope) <= 0.50,
                (metrics.get("bb_width_percentile_120") or 100) <= 40,
                (metrics.get("ma_spread_atr") or 100) <= 1.50,
            )
        )
        direction_is_weak = abs(return_20) < 0.08 and abs(ma20_slope) < 1.0
        accelerating_down = self._is_decline_accelerating(metrics, volume_state)

        # A sharp recent sell-off can have a low 20-day efficiency ratio after
        # reversing an earlier rally.  It is still a decline, not a quiet range.
        if decline and accelerating_down:
            return "declining", "accelerating_down", "watch"

        if low_efficiency and range_confirmation and direction_is_weak:
            return "range_bound", "neutral", "watch"

        if decline:
            extreme_entry = (
                price_zone == "extreme_low"
                and volume_state == "extreme_contraction"
                and bool(safety_evidence)
                and not accelerating_down
            )
            if extreme_entry:
                return "declining_exhaustion", "decelerating_down", "accumulate"
            low_buy = (
                price_zone in {"extreme_low", "low", "mid"}
                and (metrics.get("price_percentile_120") or 100) <= 40
                and volume_state in {"contraction", "extreme_contraction"}
                and bool(safety_evidence)
                and not accelerating_down
            )
            if low_buy:
                return "declining", "decelerating_down", "low_buy"
            return (
                "declining",
                "accelerating_down" if accelerating_down else "weak_down",
                "watch",
            )

        if advance:
            if price_zone == "extreme_high" and len(weakening_dimensions) >= 2:
                return "advancing_exhaustion", "exhausted_up", "exit"
            if price_zone in {"high", "extreme_high"} and len(weakening_dimensions) >= 2:
                return "advancing_weakening", "weakening_up", "reduce"
            return "advancing", "healthy_up", "hold"

        return "range_bound", "neutral", "watch"

    @staticmethod
    def _confidence(
        *,
        phase: str,
        data_quality: str,
        safety_count: int,
        weakening_dimension_count: int,
    ) -> str:
        if data_quality == "insufficient" or phase == "unknown":
            return "低"
        evidence_count = safety_count if phase.startswith("declining") else weakening_dimension_count
        if data_quality == "full" and evidence_count >= 3:
            return "高"
        if data_quality == "full" or evidence_count >= 2:
            return "中"
        return "低"

    @staticmethod
    def _evidence(
        *,
        metrics: Dict[str, Any],
        phase: str,
        price_zone: str,
        volume_state: str,
        safety_evidence: List[str],
        weakening_evidence: List[str],
    ) -> List[str]:
        price_pct = metrics.get("price_percentile_120")
        volume_pct = metrics.get("volume_percentile_120")
        volume_ratio = metrics.get("volume_ratio_5_20")
        contraction_streak = metrics.get("volume_contraction_streak") or 0
        evidence = [
            f"[价格位置] 120日价格分位{price_pct:.1f}%，区间={price_zone}"
            if price_pct is not None
            else "[价格位置] 长周期价格分位不可用",
            f"[量能] 5日均量分位{volume_pct:.1f}%，5/20量比{volume_ratio:.2f}，持续缩量{contraction_streak}日，状态={volume_state}"
            if volume_pct is not None and volume_ratio is not None
            else "[量能] 多周期成交量证据不可用",
        ]
        if phase.startswith("declining"):
            evidence.extend(safety_evidence[:4])
        elif phase.startswith("advancing"):
            evidence.extend(weakening_evidence[:4])
        else:
            evidence.append(
                f"[震荡] 20日效率比{(metrics.get('efficiency_ratio_20') or 0):.2f}，均线与波动缺乏明确方向"
            )
        return evidence

    @staticmethod
    def _summary(phase: str, signal: str) -> str:
        summaries = {
            "declining_exhaustion": "价格与量能进入下跌衰竭区，并有独立确认",
            "declining": "仍处于下跌过程，按缩量和跌速决定低吸或观察",
            "range_bound": "趋势效率偏低，处于横盘震荡观察期",
            "advancing": "上涨结构仍健康，暂未出现足够退出证据",
            "advancing_weakening": "高位上涨动能跨维度减弱，适合减仓",
            "advancing_exhaustion": "极高位出现双重衰竭证据，适合清仓",
            "structural_risk": "基本面或结构性硬风险优先，适合清仓",
            "unknown": "数据不足，无法可靠识别周期阶段",
        }
        return summaries.get(phase, f"周期阶段={phase}，建议={signal}")

    @staticmethod
    def _reference_points(
        df: pd.DataFrame,
        metrics: Dict[str, Any],
        signal: str,
    ) -> Dict[str, str]:
        current = metrics.get("current_price")
        atr = metrics.get("atr_14")
        support20 = metrics.get("support_20")
        support60 = metrics.get("support_60")
        resistance20 = metrics.get("resistance_20")
        resistance60 = metrics.get("resistance_60")
        ma20 = metrics.get("ma20")

        if signal in {"low_buy", "accumulate"}:
            lower = support60 if signal == "accumulate" else support20
            upper = min(float(current), float(lower) + float(atr)) if all(
                item is not None and np.isfinite(item) for item in (current, lower, atr)
            ) else current
            return {
                "zone_label": "抢筹观察区" if signal == "accumulate" else "低吸观察区",
                "zone": _fmt_zone(lower, upper),
                "risk_label": "结构失效参考",
                "risk_line": _fmt_price(support60 if signal == "accumulate" else support20),
            }
        if signal in {"reduce", "exit"}:
            return {
                "zone_label": "清仓触发区" if signal == "exit" else "减仓压力区",
                "zone": _fmt_zone(current, resistance60),
                "risk_label": "重新转强参考",
                "risk_line": _fmt_price(max(item for item in (resistance20, resistance60) if item is not None)),
            }
        if signal == "hold":
            defense = max(item for item in (ma20, support20) if item is not None)
            return {
                "zone_label": "持有防守参考",
                "zone": _fmt_price(defense),
                "risk_label": "转弱条件",
                "risk_line": "跌破防守位且动能继续恶化",
            }
        return {
            "zone_label": "震荡区间",
            "zone": _fmt_zone(support20, resistance20),
            "risk_label": "下一触发",
            "risk_line": "放量突破上沿或跌破下沿后重新评估",
        }


def analyze_timing_state(
    df: Optional[pd.DataFrame],
    code: str,
    *,
    target_date: Optional[date] = None,
) -> TimingStateResult:
    """Convenience entry point shared by the pipeline and Agent tool."""

    return TimingStateAnalyzer().analyze(df, code, target_date=target_date)
