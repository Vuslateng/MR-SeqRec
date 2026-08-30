"""缺失感知 SASRec（S2 最小核）：物品表示随可用性配置变化。

在 S1 ID 基线之上叠加"可用性标志嵌入"：物品表示 = 基础物品嵌入 + 可用模态的
标志嵌入之和；模态缺失 → 该模态的标志项不加入。于是同一物品在不同缺失方案
（环境）下表示不同、预测风险不同，V-REx 才有可正则化的环境差异（§2.4）。

诚实边界（写死）：最小核只建模**可用性标志**，不含模态内容特征（text/image/desc
的内容投影在完整版中加入）；本模块验证缺失结构的模型侧支撑。avail=None 时
完全退化为 S1 的纯 ID SASRec（表示 = 基础嵌入，行为与 SASRec 逐位一致）。
"""

from __future__ import annotations

import torch
from torch import nn, Tensor

from mrseqrec.models.sasrec import SASRec


class MissingnessAwareSASRec(SASRec):
    def __init__(
        self,
        item_vocab_size: int,
        n_modalities: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        max_seq_len: int = 50,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__(item_vocab_size, hidden_dim, num_layers, num_heads, dropout, max_seq_len, pad_token_id)
        self.n_modalities = n_modalities
        # 每个模态一个可用性标志嵌入；avail=False 时不加入
        self.mod_emb = nn.Parameter(torch.zeros(n_modalities, hidden_dim))
        nn.init.normal_(self.mod_emb, mean=0.0, std=0.02)

    def item_representation(self, ids: Tensor, avail: Tensor | None) -> Tensor:
        """ids: (...,)，avail: (..., M) float/bool。返回物品表示 (..., d)。

        avail=None → 基础嵌入（S1 退化）。avail 的末维必须等于 n_modalities。
        """
        rep = self.item_emb(ids)
        if avail is not None:
            rep = rep + torch.einsum("...m,md->...d", avail.float(), self.mod_emb)
        return rep

    def forward(self, input_ids: Tensor, avail: Tensor | None = None) -> Tensor:
        """input_ids: (B, L)，avail: (B, L, M) 可选。返回全序列隐藏 (B, L, d)。"""
        b, l = input_ids.shape
        positions = torch.arange(l, device=input_ids.device).expand(b, l)
        x = self.item_representation(input_ids, avail) + self.pos_emb(positions)
        attn_mask = self._attn_mask(input_ids).repeat_interleave(self.num_heads, dim=0)
        x = self.encoder(x, mask=attn_mask)
        return self.ln(x)

    def score_candidates(self, hidden: Tensor, candidate_ids: Tensor, candidate_avail: Tensor | None = None) -> Tensor:
        """hidden: (B, d)，candidate_ids: (B, C)，candidate_avail: (B, C, M) 可选。返回 logits (B, C)。"""
        cand_rep = self.item_representation(candidate_ids, candidate_avail)
        return torch.einsum("bd,bcd->bc", hidden, cand_rep)
