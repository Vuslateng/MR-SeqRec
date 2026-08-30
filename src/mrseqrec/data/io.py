"""原始交互数据读取：`user item ts` 三列，空格分隔。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["user", "item", "ts"]


def load_interactions(path: str | Path, sep: str = " ") -> pd.DataFrame:
    """读取交互文件并做基础清洗（仅去完全重复三元组，保留复购记录）。

    健康域里同用户重复购买同一商品是核心信号（复购周期），因此**不**按 (user, item) 去重。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"interactions file not found: {p}")
    df = pd.read_csv(
        p,
        sep=sep,
        header=None,
        names=COLUMNS,
        dtype={"user": np.int64, "item": np.int64, "ts": np.float64},
    )
    df = df.drop_duplicates(subset=COLUMNS).sort_values(["user", "ts"], kind="mergesort")
    return df
