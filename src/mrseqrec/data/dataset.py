"""PyTorch Dataset 与 padding 工具。

- 训练集：每用户一行，input = 训练序列（右侧 pad），target = 右移一位（pad=0 由 CE ignore_index 忽略）。
- 评估集：每用户一个输入（**左侧** pad，保证末位 = 最近物品，取 hidden[-1]）+ 候选集合（正例在索引 0）。
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

PAD = 0


def right_pad(seq: np.ndarray, max_len: int, pad_val: int = PAD) -> np.ndarray:
    if len(seq) >= max_len:
        return seq[-max_len:].astype(np.int64)
    out = np.full(max_len, pad_val, dtype=np.int64)
    out[: len(seq)] = seq
    return out


def left_pad(seq: np.ndarray, max_len: int, pad_val: int = PAD) -> np.ndarray:
    if len(seq) >= max_len:
        return seq[-max_len:].astype(np.int64)
    out = np.full(max_len, pad_val, dtype=np.int64)
    out[max_len - len(seq) :] = seq
    return out


class SeqTrainDataset(Dataset):
    """训练样本：每用户一行（全前缀监督，SASRec 式）。"""

    def __init__(self, seqs: list[np.ndarray], max_len: int) -> None:
        self.seqs = seqs
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.seqs[idx][-self.max_len :]
        n = len(seq)
        input_ids = right_pad(seq, self.max_len)               # [i1..in, 0..0]
        target = np.roll(seq, -1)                              # [i2..in, i1]
        target[-1] = PAD                                       # [i2..in, pad]
        target_ids = right_pad(target, self.max_len)           # [i2..in, pad, 0..0]
        return torch.from_numpy(input_ids), torch.from_numpy(target_ids)


class EvalDataset(Dataset):
    """评估样本：输入序列（左 pad）+ 候选集合（正例 idx0 + K 负例）。"""

    def __init__(
        self,
        input_seqs: list[np.ndarray],
        targets: np.ndarray,
        negatives: np.ndarray,
        max_len: int,
    ) -> None:
        self.inputs = [left_pad(s, max_len) for s in input_seqs]
        self.targets = targets
        self.negatives = negatives

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cand = np.concatenate([[self.targets[idx]], self.negatives[idx]])
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(cand)
