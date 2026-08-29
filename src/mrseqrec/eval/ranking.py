"""RankingEvaluator：批量评分候选集合 + 指标汇总。"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from mrseqrec.eval.metrics import evaluate
from mrseqrec.models.sasrec import SASRec


class RankingEvaluator:
    def __init__(
        self,
        model: SASRec,
        topks: list[int],
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.topks = topks
        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def evaluate(self, eval_ds) -> dict[str, float]:
        self.model.eval()
        loader = DataLoader(eval_ds, batch_size=self.batch_size, shuffle=False)
        scores_list: list[np.ndarray] = []
        for input_ids, candidates in loader:
            input_ids = input_ids.to(self.device)
            candidates = candidates.to(self.device)
            hidden = self.model(input_ids)[:, -1, :]  # 左 pad → 末位=最近物品
            scores = self.model.score_candidates(hidden, candidates)
            scores_list.append(scores.float().cpu().numpy())
        return evaluate(np.concatenate(scores_list, axis=0), self.topks)
