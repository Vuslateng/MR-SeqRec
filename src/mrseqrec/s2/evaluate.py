"""S2 OOD 评估（§2.4 OOD 协议）：训练未见缺失方案上的排序指标与 retention 曲线。

对每个评估方案（obs 作参考 + 训练未见的缺失类型/缺失率），用该方案实现的物品
可用性（历史序列 + 正负候选一致）评估 Recall/NDCG；指标经 rank_of_positive 的
NaN 守卫（与 S1 同源），防缺失配置引发融合 NaN 致指标虚高。
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mrseqrec.data.dataset import left_pad
from mrseqrec.eval.metrics import evaluate
from mrseqrec.s2.environ import MODALITIES, scheme_key, scheme_realization


class EnvEvalDataset(Dataset):
    """评估样本：输入序列（左 pad）+ 候选（正例 idx0 + K 负例），附带可用性。

    realization : (V, M) bool——该方案下每个物品的可用性；历史与候选一致使用。
    """

    def __init__(self, input_seqs, targets: np.ndarray, negatives: np.ndarray, realization: np.ndarray, max_len: int):
        self.inputs = [left_pad(s, max_len) for s in input_seqs]
        self.targets = targets
        self.negatives = negatives
        self.realization = realization

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        cand = np.concatenate([[self.targets[idx]], self.negatives[idx]])
        inp = self.inputs[idx]
        return (
            torch.from_numpy(inp),
            torch.from_numpy(self.realization[inp]),
            torch.from_numpy(cand),
            torch.from_numpy(self.realization[cand]),
        )


def env_rank_metrics(model, eval_ds, topks: list[int], batch_size: int, device: torch.device) -> dict[str, float]:
    """批量评分（左 pad → 末位=最近物品）→ 排序指标。"""
    model.eval()
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    scores_list: list[np.ndarray] = []
    with torch.no_grad():
        for input_ids, input_avail, cand, cand_avail in loader:
            input_ids = input_ids.to(device)
            input_avail = input_avail.float().to(device)
            cand = cand.to(device)
            cand_avail = cand_avail.float().to(device)
            hidden = model(input_ids, input_avail)[:, -1, :]
            scores = model.score_candidates(hidden, cand, cand_avail)
            scores_list.append(scores.float().cpu().numpy())
    return evaluate(np.concatenate(scores_list, axis=0), topks)


def evaluate_schemes(
    model,
    input_seqs: list[np.ndarray],
    targets: np.ndarray,
    negatives: np.ndarray,
    base_avail: dict,
    counts: np.ndarray | None,
    vocab: int,
    schemes: list[dict],
    *,
    max_len: int,
    batch_size: int,
    device: torch.device,
    topks: list[int],
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """对每个评估方案返回 {recall@k, ndcg@k}。obs 作自然参考，其余为 OOD。"""
    out: dict[str, dict[str, float]] = {}
    for s in schemes:
        bits = scheme_realization(s, base_avail, counts, vocab, seed)
        ds = EnvEvalDataset(input_seqs, targets, negatives, bits, max_len)
        out[scheme_key(s)] = env_rank_metrics(model, ds, topks, batch_size, device)
    return out


def retention_curve(
    metrics_by_scheme: dict[str, dict[str, float]],
    reference: str = "obs",
    topk: int = 10,
) -> dict[str, float | None]:
    """相对参考方案（默认自然 obs）的 recall@topk 保留率百分比。"""
    base = metrics_by_scheme[reference][f"recall@{topk}"]
    out: dict[str, float | None] = {}
    for k, v in metrics_by_scheme.items():
        out[k] = round(100.0 * v[f"recall@{topk}"] / base, 2) if base > 0 else None
    return out
