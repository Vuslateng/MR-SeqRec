import numpy as np
import pandas as pd

from mrseqrec.data.preprocess import k_core_filter, preprocess, reindex
from mrseqrec.data.synthetic import generate_interactions


def _df() -> pd.DataFrame:
    rows = [(u, it, float(t * 10)) for u in range(3) for t, it in enumerate([1, 2, 3, 4, 5, 6])]
    return pd.DataFrame(rows, columns=["user", "item", "ts"])


def test_k_core_keeps_sufficient():
    assert len(k_core_filter(_df(), 3)) == len(_df())


def test_k_core_removes_sparse():
    df = pd.concat(
        [_df(), pd.DataFrame([(99, 1, 1.0)], columns=["user", "item", "ts"])]
    )
    out = k_core_filter(df, 2)
    assert 99 not in out["user"].values


def test_reindex_items_start_at_1():
    df = _df()
    r, vocab = reindex(df)
    assert r["item"].min() >= 1
    assert vocab == r["item"].nunique() + 1  # +1 含 pad


def test_preprocess_no_crash_tiny_vocab():
    """极端小词表（负采样池可能为空）必须优雅降级而不是崩溃。"""
    data = preprocess(_df(), min_interactions=3, num_negatives=5, seed=1)
    assert data.valid_negatives.shape == (3, 5)
    assert data.test_negatives.shape == (3, 5)


def test_preprocess_realistic_negatives():
    """真实规模：负例不含正例与历史已购（词表 >> 历史时严格成立）。"""
    df = generate_interactions(n_users=80, n_items=150, min_len=6, max_len=25, seed=3)
    data = preprocess(df, min_interactions=5, num_negatives=20, seed=1)
    assert data.valid_negatives.shape == (data.n_users, 20)
    assert data.valid_targets.shape == (data.n_users,)
    for s, t, neg in zip(data.valid_input_seqs, data.valid_targets, data.valid_negatives):
        assert t not in neg  # 负例不含正例
        assert not (set(np.unique(s)) & set(neg.tolist()))  # 负例不含历史已购
