"""共享夹具：S2 合成测试台（确定性）。"""

import numpy as np
import pytest

from mrseqrec.data.preprocess import preprocess
from mrseqrec.data.synthetic import generate_interactions
from mrseqrec.missing.sampler import mnar_select
from mrseqrec.s2.environ import build_environments

NATURAL_DESC_MISSING = 0.38


@pytest.fixture(scope="session")
def s2_testbed():
    """合成交互 → 预处理 → 自然可用性（冷门缺 desc，仿真实 β̂<0）→ 训练环境。"""
    df = generate_interactions(n_users=120, n_items=60, min_len=5, max_len=15, seed=7)
    data = preprocess(df, min_interactions=5, num_negatives=20, seed=0)
    counts_orig = df.groupby("item").size()
    counts = np.zeros(data.item_vocab_size, dtype=float)
    base_avail = {}
    for new_id, orig in data.item_map.items():
        counts[new_id] = float(counts_orig.get(orig, 1))
        base_avail[new_id] = {"text": True, "image": True, "desc": True}
    miss = mnar_select(counts[1:], NATURAL_DESC_MISSING)
    for new_id in data.item_map:
        base_avail[new_id]["desc"] = not miss[new_id - 1]
    schemes = [
        {"name": "obs"},
        {"name": "mcar", "mcar_p": 0.5},
        {"name": "mnar", "mnar_rate": 0.5},
    ]
    env_data = build_environments(
        data.train_seqs, base_avail, counts, data.item_vocab_size,
        schemes, split_obs=True, seed=0,
    )
    return data, base_avail, counts, env_data
