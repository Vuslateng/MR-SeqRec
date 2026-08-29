"""SASRec：单向自注意力序列推荐（ID 基线）。

实现要点：
- 因果掩码 + key-padding 掩码（pad 恒为物品 id 0）。
- pre-LN TransformerEncoder 保证训练稳定。
- 位置嵌入可学习（原版做法）。
"""

from __future__ import annotations

import torch
from torch import nn, Tensor


class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(max_len, hidden_dim))
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward(self, positions: Tensor) -> Tensor:
        return self.embedding[positions]


class SASRec(nn.Module):
    def __init__(
        self,
        item_vocab_size: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        max_seq_len: int = 50,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.item_emb = nn.Embedding(item_vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.pos_emb = PositionalEmbedding(hidden_dim, max_seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(hidden_dim)

    def _attn_mask(self, input_ids: Tensor) -> Tensor:
        """合并因果 + padding 的 (B, L, L) 加性掩码，避免 torch 新旧掩码混用告警。"""
        b, l = input_ids.shape
        dtype = self.item_emb.weight.dtype
        causal = torch.triu(
            torch.full((l, l), float("-inf"), dtype=dtype, device=input_ids.device),
            diagonal=1,
        )
        pad_as_key = (input_ids == 0).unsqueeze(1).expand(b, l, l)  # 不能 attend 到 pad 位置
        return causal + torch.where(pad_as_key, torch.finfo(dtype).min, 0.0)

    def forward(self, input_ids: Tensor) -> Tensor:
        """input_ids: (B, L)。返回全序列隐藏 (B, L, d)。

        3D 掩码按 torch 约定需为 (B*num_heads, L, L)，各 head 共享同一掩码。
        """
        b, l = input_ids.shape
        positions = torch.arange(l, device=input_ids.device).expand(b, l)
        x = self.item_emb(input_ids) + self.pos_emb(positions)
        attn_mask = self._attn_mask(input_ids).repeat_interleave(self.num_heads, dim=0)
        x = self.encoder(x, mask=attn_mask)
        return self.ln(x)

    def score_candidates(self, hidden: Tensor, candidate_ids: Tensor) -> Tensor:
        """hidden: (B, d)，candidate_ids: (B, C)。返回 logits (B, C)。"""
        cand_emb = self.item_emb(candidate_ids)  # (B, C, d)
        return torch.einsum("bd,bcd->bc", hidden, cand_emb)
