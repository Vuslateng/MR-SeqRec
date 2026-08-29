"""训练循环：下一物品交叉熵（SASRec 全前缀监督）。

损失两档，按词表规模自动切换：
- 词表小（≤ num_negatives+1）：全词表 logits 交叉熵，pad(0) 由 ignore_index 忽略。
- 词表大（真实 Amazon 规模）：采样交叉熵——每位置取 1 正 + num_negatives 个均匀负例，
  仅对 (B,L,1+K) 个候选打分。这是全词表 softmax 的无偏蒙特卡洛估计，与 SASRec 原文
  训练目标及本仓库评估协议（正例 + K 负例）一致；将瓶颈 matmul 从 B·L·V 降到 B·L·(1+K)，
  使真实词表（V~6 万）训练速度提升约两个数量级。

注意：两档损失量纲不同，loss 曲线仅同档内可比。
"""

from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mrseqrec.utils.log import get_logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float,
        weight_decay: float,
        grad_clip: float | None,
        log_every: int,
        num_negatives: int = 100,
        item_vocab_size: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.num_negatives = num_negatives
        self.item_vocab_size = item_vocab_size
        self.logger = get_logger("trainer")

    def _loss(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """下一物品 CE。input_ids/target_ids: (B, L)，pad 恒为 0。"""
        hidden = self.model(input_ids)  # (B, L, d)
        b, l = input_ids.shape
        vocab = self.item_vocab_size
        if vocab is None or vocab <= self.num_negatives + 1:
            # 小词表：全词表 softmax，pad 标签为 0 由 ignore_index 忽略
            logits = hidden @ self.model.item_emb.weight.T  # (B, L, V)
            return F.cross_entropy(
                logits.reshape(b * l, -1), target_ids.reshape(b * l), ignore_index=0
            )
        # 大词表：采样交叉熵（1 正 + K 均匀负例，K=num_negatives）
        neg = torch.randint(1, vocab, (b, l, self.num_negatives), device=target_ids.device)
        while True:  # 拒绝采样剔除与正例相等的负例（词表大时几乎不进循环）
            bad = neg == target_ids.unsqueeze(-1)
            if not bad.any():
                break
            neg[bad] = torch.randint(1, vocab, (int(bad.sum()),), device=target_ids.device)
        candidates = torch.cat([target_ids.unsqueeze(-1), neg], dim=-1)  # (B, L, 1+K)
        cand_emb = self.model.item_emb(candidates)  # (B, L, 1+K, d)
        scores = torch.einsum("bld,blkd->blk", hidden, cand_emb)  # (B, L, 1+K)
        labels = torch.zeros(b, l, dtype=torch.long, device=target_ids.device)  # 正例恒为索引 0
        labels[target_ids == 0] = -100  # pad 位置忽略
        return F.cross_entropy(
            scores.reshape(b * l, self.num_negatives + 1),
            labels.reshape(b * l),
            ignore_index=-100,
        )

    def fit(self, train_ds, epochs: int, batch_size: int) -> list[float]:
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        epoch_losses: list[float] = []
        for epoch in range(1, epochs + 1):
            total, n_steps = 0.0, 0
            for step, (input_ids, target_ids) in enumerate(
                tqdm(loader, desc=f"Epoch {epoch}", leave=False), start=1
            ):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                loss = self._loss(input_ids, target_ids)
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                total += loss.item()
                n_steps += 1
                if step % self.log_every == 0:
                    self.logger.info("epoch %d step %d loss %.4f", epoch, step, loss.item())
            epoch_losses.append(total / n_steps)
            self.logger.info("epoch %d done, avg_loss %.4f", epoch, epoch_losses[-1])
        return epoch_losses
