"""合成数据生成器：供管线测试与信号检验脚本验证。

- 用户按流行度加权随机游走选物品（Zipf 式流行度，alpha 控制长尾）。
- 时间戳：**unix epoch 秒**（与真实 Amazon 交互文件一致），随机游走的指数间隔；
  periodic_fraction>0 时按 `period_days` 周期插入"复购商品"，用于验证信号检验能否检测出周期性。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_interactions(
    n_users: int = 2000,
    n_items: int = 800,
    min_len: int = 5,
    max_len: int = 40,
    alpha: float = 0.6,
    seed: int = 42,
    periodic_fraction: float = 0.0,
    period_days: float = 30.0,
    mean_gap_days: float = 5.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    popularity = np.arange(1, n_items + 1, dtype=float) ** (-alpha)
    popularity /= popularity.sum()

    rows: list[tuple[int, int, float]] = []
    n_periodic = int(n_users * periodic_fraction)
    second = 86400.0
    epoch0 = 1_500_000_000.0  # 固定起点，保证秒级时间戳
    for u in range(n_users):
        length = rng.integers(min_len, max_len + 1)
        ts = 0.0
        seq: list[tuple[int, float]] = []
        for _ in range(length):
            ts += rng.exponential(mean_gap_days * second)  # 秒
            item = int(rng.choice(n_items, p=popularity))  # 0-based
            seq.append((item, ts))
        if u < n_periodic:
            # 周期性复购：随机挑一个"主商品"，约每 period_days 出现一次
            main_item = int(rng.integers(0, n_items))
            t = 0.0
            max_t = seq[-1][1]
            while t < max_t:
                t += period_days * second + float(rng.normal(0, 1.5 * second))
                if t < max_t:
                    seq.append((main_item, t))
        seq.sort(key=lambda x: x[1])
        for item, t in seq:
            rows.append((u, item + 1, epoch0 + float(t)))  # 物品 1..n_items
    return pd.DataFrame(rows, columns=["user", "item", "ts"])
