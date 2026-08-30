"""数据信号检验（三审 3.2）：Amazon Health 复购周期信号。

健康域时间信号的本质是**同一商品被周期性复购**（慢性病每月买血糖试纸/用药间隔代理），
因此决定性闸门基于**同商品复购间隔**（(user, item) 对内的购买间隔），而非相邻任意交互间隔。
若数据里不存在周期复购信号，时间信号贡献应降级/移除、健康叙事弱化。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PERIOD_BIN_MIN = 20  # 复购周期候选窗口（天）
PERIOD_BIN_MAX = 40
SECONDS_PER_DAY = 86400.0


@dataclass
class IntervalStats:
    """用户整体购物节奏（相邻交互间隔），仅作背景上下文。"""

    median_days: float
    mean_days: float
    p25_days: float
    p75_days: float
    frac_short_cycle: float  # 间隔 ≤ 7 天占比（高频补货）

    def summary(self) -> str:
        return (
            f"cadence[days] median={self.median_days:.1f} mean={self.mean_days:.1f} "
            f"p25={self.p25_days:.1f} p75={self.p75_days:.1f} frac<=7d={self.frac_short_cycle:.3f}"
        )


@dataclass
class RepurchaseReport:
    """同商品复购周期信号（健康叙事可行性闸门）。"""

    repurchase_pairs: int        # (user,item) 复购 ≥ min_purchases 且时间跨度达标的对数
    periodic_pairs: int          # 其中间隔集中（MAD 一致）且中位数落在 20–40 天窗口的对数
    fraction_periodic: float     # periodic_pairs / repurchase_pairs
    dominant_period_days: float  # 周期性复购对的中位间隔均值
    n_users_with_signal: int
    healthy_signal: bool

    def summary(self) -> str:
        verdict = "SIGNAL_OK 存在复购周期信号" if self.healthy_signal else "SIGNAL_WEAK 信号弱，时间信号贡献需降级/移除"
        return (
            f"repurchase: pairs={self.repurchase_pairs} periodic={self.periodic_pairs} "
            f"fraction_20-40d={self.fraction_periodic:.3f} dominant_period={self.dominant_period_days:.1f}d "
            f"users_with_signal={self.n_users_with_signal} -> {verdict}"
        )


def interval_stats(df: pd.DataFrame) -> IntervalStats:
    """全体用户相邻交互间隔（天）的分布（用户购物节奏，背景上下文）。"""
    deltas = _consecutive_deltas_days(df)
    p25, med, p75 = np.percentile(deltas, [25, 50, 75])
    frac = float(np.mean(deltas <= 7.0))
    return IntervalStats(
        median_days=float(med), mean_days=float(deltas.mean()),
        p25_days=float(p25), p75_days=float(p75), frac_short_cycle=frac,
    )


def repurchase_check(
    df: pd.DataFrame,
    min_purchases: int = 3,
    min_span_days: int = 60,
    mad_rel: float = 0.3,
    signal_threshold: float = 0.1,
    min_pairs: int = 10,
) -> RepurchaseReport:
    """同商品复购间隔的周期判据：间隔**集中**（MAD ≤ max(5d, mad_rel×中位数)）且中位数落在 20–40 天。

    相比直方图峰检测，中位数+MAD 对真实世界周期抖动鲁棒，且对随机噪声（指数间隔）天然免疫。

    性能：候选先用向量化 groupby 过滤（(user,item) 频次 ≥ min_purchases 且时间跨度达标），
    只对候选对循环算间隔——避免对全体用户做 Python 双层循环（千万级用户时慢一个量级）。
    """
    grp = df.groupby(["user", "item"])
    size = grp.size()
    cand = size[size >= min_purchases]
    if not cand.empty:
        ts_min = grp["ts"].min()
        ts_max = grp["ts"].max()
        span_ok = (ts_max - ts_min) >= min_span_days * SECONDS_PER_DAY
        cand = cand[span_ok[cand.index]]

    total_pairs = int(len(cand))
    if total_pairs == 0:
        return RepurchaseReport(
            repurchase_pairs=0, periodic_pairs=0, fraction_periodic=0.0,
            dominant_period_days=0.0, n_users_with_signal=0, healthy_signal=False,
        )

    cand_df = df.set_index(["user", "item"]).loc[cand.index].reset_index()
    periodic_pairs = 0
    medians: list[float] = []
    users_with_signal: set[int] = set()
    for (user_id, _item_id), sub in cand_df.groupby(["user", "item"], sort=False):
        deltas = np.diff(np.sort(sub["ts"].to_numpy())) / SECONDS_PER_DAY
        deltas = deltas[(deltas >= 1) & (deltas <= 365)]
        if len(deltas) < 2:
            continue
        med = float(np.median(deltas))
        if not (PERIOD_BIN_MIN <= med <= PERIOD_BIN_MAX):
            continue
        mad = float(np.median(np.abs(deltas - med)))
        if mad > max(5.0, mad_rel * med):
            continue
        periodic_pairs += 1
        medians.append(med)
        users_with_signal.add(int(user_id))

    fraction = periodic_pairs / total_pairs if total_pairs else 0.0
    dominant = float(np.mean(medians)) if medians else 0.0
    healthy = total_pairs >= min_pairs and fraction >= signal_threshold
    return RepurchaseReport(
        repurchase_pairs=total_pairs,
        periodic_pairs=periodic_pairs,
        fraction_periodic=fraction,
        dominant_period_days=dominant,
        n_users_with_signal=len(users_with_signal),
        healthy_signal=healthy,
    )


def _consecutive_deltas_days(df: pd.DataFrame) -> np.ndarray:
    deltas = df.sort_values(["user", "ts"]).groupby("user")["ts"].diff().dropna().to_numpy()
    deltas = deltas[deltas > 0] / SECONDS_PER_DAY
    return deltas
