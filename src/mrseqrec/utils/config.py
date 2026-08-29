"""配置加载与校验：YAML → pydantic 模型，运行期完成类型与约束校验。

所有运行参数集中到 configs/*.yaml，代码不出现魔法数字。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """数据侧配置。interactions_file: 每行 `user item ts`（空格分隔）。"""

    interactions_file: Path
    min_interactions: int = Field(5, ge=1, description="k-core 阈值")
    max_seq_len: int = Field(50, ge=1)
    num_negatives: int = Field(100, ge=1)
    seed: int = 2026


class ModelConfig(BaseModel):
    """架构配置（词表/序列长由数据侧运行期传入，不写死在配置里）。"""

    name: Literal["sasrec"] = "sasrec"
    hidden_dim: int = Field(64, ge=1)
    num_layers: int = Field(2, ge=1)
    num_heads: int = Field(2, ge=1)
    dropout: float = Field(0.2, ge=0.0, le=1.0)


class TrainConfig(BaseModel):
    """训练配置。device="auto" 依次尝试 CUDA → Intel XPU → CPU。

    num_negatives: 训练损失负采样数。词表 ≤ num_negatives+1 时自动回退全词表 CE。
    """

    batch_size: int = Field(256, ge=1)
    epochs: int = Field(50, ge=1)
    lr: float = Field(1e-3, gt=0.0)
    weight_decay: float = Field(0.0, ge=0.0)
    grad_clip: float | None = Field(None, gt=0.0)
    log_every: int = Field(100, ge=1)
    num_negatives: int = Field(100, ge=1)
    device: str = "auto"
    seed: int = 2026


class EvalConfig(BaseModel):
    topk: list[int] = Field(default_factory=lambda: [10, 20])
    batch_size: int = Field(256, ge=1)


class Config(BaseModel):
    """主配置：data / model / train / eval。"""

    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig


def load_config(path: str | Path) -> Config:
    """从 YAML 加载配置，pydantic 负责类型与约束校验。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return Config.model_validate(raw)
