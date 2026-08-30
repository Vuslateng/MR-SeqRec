"""数据预处理：k-core 过滤 → 重编号 → 用户序列 → leave-one-out 划分 → 负采样。

划分语义（标准 SASRec 式）：
- 每个用户序列 [i1..in]：test = 末位，valid = 倒数第二位，train = 前 n-2 位。
- 正例固定在候选集合索引 0，负例为 K 个均匀采样（排除目标与其历史已购）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mrseqrec.utils.log import get_logger

logger = get_logger("preprocess")

PAD_ID = 0  # 物品重编号从 1 开始，0 恒为 pad


@dataclass
class PreprocessedData:
    """预处理产物：用户序列、划分、负采样与词表规模。"""

    train_seqs: list[np.ndarray] = field(default_factory=list)          # 训练用用户序列
    valid_input_seqs: list[np.ndarray] = field(default_factory=list)    # 预测 valid_target 的输入
    valid_targets: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    test_input_seqs: list[np.ndarray] = field(default_factory=list)     # 预测 test_target 的输入
    test_targets: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    valid_negatives: np.ndarray | None = None                           # (U, K)
    test_negatives: np.ndarray | None = None                            # (U, K)
    item_vocab_size: int = 0                                            # 含 pad
    n_users: int = 0
    seq_lens: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))


def k_core_filter(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """迭代式 k-core：用户与物品的交互数均 ≥ k。"""
    prev_len = -1
    while len(df) != prev_len:
        prev_len = len(df)
        keep_items = df["item"].value_counts()[df["item"].value_counts() >= k].index
        keep_users = df["user"].value_counts()[df["user"].value_counts() >= k].index
        df = df[df["item"].isin(keep_items) & df["user"].isin(keep_users)]
    return df


def reindex(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """用户重编号 0..U-1；物品重编号 1..V（0 保留给 pad）。返回 (df, vocab_size)。"""
    user_map = {u: i for i, u in enumerate(sorted(df["user"].unique()))}
    item_map = {i: j + 1 for j, i in enumerate(sorted(df["item"].unique()))}
    df = df.assign(user=df["user"].map(user_map), item=df["item"].map(item_map))
    return df, len(item_map) + 1


def build_user_sequences(df: pd.DataFrame) -> list[np.ndarray]:
    """按用户返回时间升序的物品序列（df 已按 (user, ts) 排序）。"""
    grouped = df.groupby("user", sort=True)["item"]
    return [np.asarray(seq, dtype=np.int64) for seq in grouped.apply(list).to_numpy()]


def _leave_one_out(seqs: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
    train, valid_in, valid_t, test_in, test_t = [], [], [], [], []
    for seq in seqs:
        n = len(seq)
        train.append(seq[:-2])
        valid_in.append(seq[:-2])
        valid_t.append(seq[-2])
        test_in.append(seq[:-1])
        test_t.append(seq[-1])
    return (
        train,
        valid_in,
        np.asarray(valid_t, dtype=np.int64),
        test_in,
        np.asarray(test_t, dtype=np.int64),
    )


def _sample_negatives(
    seq: np.ndarray,
    target: int,
    vocab: int,
    k: int,
    rng: np.random.Generator,
    pool_all: np.ndarray,
) -> np.ndarray:
    """采样 K 个负例：优先排除目标与其历史已购；池子不足时放宽"排除已购"，但始终排除正例。

    用预计算 pool_all（[1..vocab-1]）+ 布尔掩码替代 setdiff1d——setdiff1d 每次做全量 unique(排序)，
    在真实规模（V~6万、U~13万）下是 O(U·VlogV)，实测慢 3 个数量级；掩码法 O(V) 且 pool 顺序不变，
    因此 RNG 消费流与旧实现完全一致（负采样结果逐位相同）。
    """
    excluded = np.unique(np.concatenate([seq, [target]]))
    mask = np.ones(vocab, dtype=bool)
    mask[excluded] = False
    pool = pool_all[mask[1:]]
    if len(pool) >= k:
        return rng.choice(pool, size=k, replace=False)
    fallback_mask = np.ones(vocab, dtype=bool)
    fallback_mask[target] = False  # 放宽：仅排除正例，允许已购
    fallback = pool_all[fallback_mask[1:]]
    if len(fallback) == 0:
        raise ValueError("vocab too small for negative sampling")
    idx = rng.integers(0, len(fallback), size=k)
    return fallback[idx]


def preprocess(df: pd.DataFrame, min_interactions: int, num_negatives: int, seed: int) -> PreprocessedData:
    """完整预处理流水线（划分 + 负采样，负采样统一用固定种子保证可复现）。"""
    if df.empty:
        raise ValueError("empty interactions input")
    filtered = k_core_filter(df, k=min_interactions)
    logger.info("k-core filter: %d -> %d interactions", len(df), len(filtered))
    reindexed, vocab = reindex(filtered)
    seqs = build_user_sequences(reindexed)
    min_len = min(len(s) for s in seqs)
    if min_len < 3:
        raise ValueError(f"sequence too short after k-core (min length {min_len}); raise min_interactions")
    train, valid_in, valid_t, test_in, test_t = _leave_one_out(seqs)

    rng = np.random.default_rng(seed)
    pool_all = np.arange(1, vocab, dtype=np.int64)
    valid_neg = np.stack([_sample_negatives(s, t, vocab, num_negatives, rng, pool_all) for s, t in zip(valid_in, valid_t)])
    test_neg = np.stack([_sample_negatives(s, t, vocab, num_negatives, rng, pool_all) for s, t in zip(test_in, test_t)])

    return PreprocessedData(
        train_seqs=train,
        valid_input_seqs=valid_in,
        valid_targets=valid_t,
        test_input_seqs=test_in,
        test_targets=test_t,
        valid_negatives=valid_neg,
        test_negatives=test_neg,
        item_vocab_size=vocab,
        n_users=len(seqs),
        seq_lens=np.asarray([len(s) for s in seqs], dtype=np.int64),
    )
