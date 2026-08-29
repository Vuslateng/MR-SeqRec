"""训练循环：下一物品交叉熵（SASRec 全前缀监督）。

损失为全词表 logits 交叉熵，pad(0) 由 ignore_index 忽略。
注意：真实 Amazon 词表大时该做法占内存，需配合 batch_size 或后续采样式 softmax。
"""

from __future__ import annotations

import torch
from torch import nn
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
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.logger = get_logger("trainer")

    def fit(self, train_ds, epochs: int, batch_size: int) -> list[float]:
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        epoch_losses: list[float] = []
        for epoch in range(1, epochs + 1):
            total, n_steps = 0.0, 0
            for step, (input_ids, target_ids) in enumerate(
                tqdm(loader, desc=f"Epoch {epoch}", leave=False), start=1
            ):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                hidden = self.model(input_ids)  # (B, L, d)
                logits = hidden @ self.model.item_emb.weight.T  # (B, L, V)
                b, l, v = logits.shape
                loss = loss_fn(logits.reshape(b * l, v), target_ids.reshape(b * l))
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
