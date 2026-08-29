"""排序评估指标（numpy 实现）。约定：候选集合中正例恒在索引 0。"""

from __future__ import annotations

import numpy as np


def rank_of_positive(scores: np.ndarray) -> np.ndarray:
    """每个用户正例在降序排序中的 0-based 位次。scores: (U, C)。"""
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.argmax(order == 0, axis=1)


def recall_at_k(pos_ranks: np.ndarray, k: int) -> float:
    return float(np.mean(pos_ranks < k))


def ndcg_at_k(pos_ranks: np.ndarray, k: int) -> float:
    """单一相关物品：命中 top-k 时计 1/log2(rank+2)，否则 0。"""
    hits = pos_ranks < k
    return float(np.mean(np.where(hits, 1.0 / np.log2(pos_ranks + 2.0), 0.0)))


def evaluate(scores: np.ndarray, topks: list[int]) -> dict[str, float]:
    """scores: (U, C)。返回 {recall@k, ndcg@k}。"""
    pos_ranks = rank_of_positive(scores)
    out: dict[str, float] = {}
    for k in topks:
        out[f"recall@{k}"] = recall_at_k(pos_ranks, k)
        out[f"ndcg@{k}"] = ndcg_at_k(pos_ranks, k)
    return out
