"""S2 不变学习训练器（§2.4）：V-REx 与 SMD 式 dropout+ERM 对照。

两种方法共用同一套方案数据与**环境分层采样**，仅损失不同——严格隔离 L_inv
方差项的贡献，满足反平凡性（三审 3.1）"同采样 schedule、同强度"：
- vrex    : L = Σ_e R_e + β·Var_e(R_e)，β>0（R_e = 环境 e 内全前缀 CE 均值）；
- ermdrop : 同分层采样、同数据，L = Σ_e R_e（β=0）= 模态 dropout 增广 + ERM。

全前缀监督与 S1 一致：input 序列逐位置预测下一物品；每个位置的候选（正例+负例）
表示用该方案实现的可用性（sampled CE 大词表 / 全词表 CE 小词表）。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from mrseqrec.s2.environ import EnvData, collate_env
from mrseqrec.utils.log import get_logger

logger = get_logger("s2.trainer")


def vrex_loss(env_risks: list[Tensor], beta: float) -> Tensor:
    """V-REx 损失：ΣR_e + β·Var_e(R_e)（总体方差，与 Krueger 一致）。β=0 即 ERM。"""
    rs = torch.stack(env_risks)
    return rs.sum() + beta * torch.var(rs, unbiased=False)


class InvariantTrainer:
    def __init__(
        self,
        model: nn.Module,
        env_data: EnvData,
        item_vocab_size: int,
        device: torch.device,
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip: float | None = None,
        num_negatives: int = 100,
        env_batch: int = 32,
        max_seq_len: int = 50,
        beta: float = 1.0,
        seed: int = 0,
    ) -> None:
        if not env_data.env_order:
            raise ValueError("环境集为空，无法训练（检查 base_avail/schemes）")
        m = env_data.realizations[env_data.env_order[0]].shape[1]
        assert m == model.n_modalities, f"可用性模态数 {m} ≠ 模型 n_modalities={model.n_modalities}"

        self.model = model.to(device)
        self.device = device
        self.env_data = env_data
        self.vocab = item_vocab_size
        self.beta = beta
        self.num_negatives = num_negatives
        self.env_batch = env_batch
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.log_every = 100
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        # 每环境索引与实现张量（仅保留非空环境）
        self.env_order = [e for e in env_data.env_order if env_data.examples[e]]
        self.env_idx = {e: np.arange(len(env_data.examples[e])) for e in self.env_order}
        self.realizations = {
            e: torch.from_numpy(env_data.realizations[e]) for e in self.env_order
        }

    def _env_risk(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        avail: Tensor,
        realization: Tensor,
    ) -> Tensor:
        """环境内全前缀 CE（标量，均值）。candidate 可用性由 realization 按 id 索引。"""
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        avail = avail.to(self.device).float()
        realization = realization.to(self.device).float()
        hidden = self.model(input_ids, avail)  # (B, L, d)
        b, l = input_ids.shape

        if self.vocab <= self.num_negatives + 1:
            # 小词表：全词表 softmax；每个候选物品用自己的可用性
            ids = torch.arange(self.vocab, device=self.device).view(1, 1, -1).expand(b, l, -1)
            cand_avail = realization[ids]  # (B, L, V, M)
            cand_rep = self.model.item_representation(ids, cand_avail)
            logits = torch.einsum("bld,blvd->blv", hidden, cand_rep)
            return F.cross_entropy(logits.reshape(b * l, -1), target_ids.reshape(b * l), ignore_index=0)

        # 大词表：采样 CE（1 正 + K 均匀负例），与 S1 训练目标一致
        neg = torch.randint(1, self.vocab, (b, l, self.num_negatives), device=self.device)
        while True:
            bad = neg == target_ids.unsqueeze(-1)
            if not bad.any():
                break
            neg[bad] = torch.randint(1, self.vocab, (int(bad.sum()),), device=self.device)
        cand = torch.cat([target_ids.unsqueeze(-1), neg], dim=-1)  # (B, L, 1+K)
        cand_avail = realization[cand].float()  # (B, L, 1+K, M)
        cand_rep = self.model.item_representation(cand, cand_avail)
        scores = torch.einsum("bld,blkd->blk", hidden, cand_rep)  # (B, L, 1+K)
        labels = torch.zeros(b, l, dtype=torch.long, device=self.device)
        labels[target_ids == 0] = -100
        return F.cross_entropy(scores.reshape(b * l, self.num_negatives + 1), labels.reshape(b * l), ignore_index=-100)

    def _sample_batch(self, env: str, rng: np.random.Generator):
        """从环境 env 放回抽样 env_batch 例，打包为 (input_ids, target_ids, avail) 张量。"""
        idx = self.env_idx[env]
        pick = rng.integers(0, len(idx), size=self.env_batch)
        batch = [self.env_data.examples[env][i] for i in pick]
        input_ids, target_ids, avail = collate_env(batch, self.env_data.realizations[env], self.max_seq_len)
        return (
            torch.from_numpy(input_ids),
            torch.from_numpy(target_ids),
            torch.from_numpy(avail),
        )

    def fit(self, epochs: int, method: str) -> list[float]:
        """method ∈ {"vrex", "ermdrop"}；返回每 epoch 平均损失。"""
        if method not in {"vrex", "ermdrop"}:
            raise ValueError(f"method 必须为 vrex 或 ermdrop，got {method}")
        beta = self.beta if method == "vrex" else 0.0
        self.model.train()
        max_env = max(len(exs) for exs in self.env_data.examples.values())
        n_steps = max(2, int(np.ceil(max_env / self.env_batch)))
        epoch_losses: list[float] = []

        for epoch in range(1, epochs + 1):
            rng = np.random.default_rng(self.seed + epoch * 10_000)
            total = 0.0
            for step in range(1, n_steps + 1):
                risks = [
                    self._env_risk(*self._sample_batch(env, rng), self.realizations[env])
                    for env in self.env_order
                ]
                loss = vrex_loss(risks, beta)
                # NaN 守卫：全 pad 目标批次（如一批全是 1 长训练序列）CE 全被忽略
                # 时返回 nan；真实 kcore10 数据不会出现，但禁止 NaN 静默传播崩溃
                if not torch.isfinite(loss):
                    logger.warning("epoch %d step %d 损失非有限（%.3f），跳过该步",
                                   epoch, step, loss.item())
                    self.optimizer.zero_grad()
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                total += loss.item()
                if step % self.log_every == 0:
                    logger.info("epoch %d step %d/%d loss %.4f", epoch, step, n_steps, loss.item())
            avg = total / n_steps
            epoch_losses.append(avg)
            logger.info("epoch %d done, avg_loss %.4f (method=%s)", epoch, avg, method)
        return epoch_losses
