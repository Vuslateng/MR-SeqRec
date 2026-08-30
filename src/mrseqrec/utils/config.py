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


class S2SchemeConfig(BaseModel):
    """S2 缺失方案（环境）。name 决定缺失类型；其余参数按 name 生效。"""

    name: Literal["obs", "mcar", "mnar", "cover"] = "mcar"
    mcar_p: float = Field(0.0, ge=0.0, le=1.0, description="MCAR 比例：每可用模态独立缺失概率")
    mnar_rate: float = Field(0.0, ge=0.0, le=1.0, description="MNAR：按流行度升序选缺 desc 的物品比例")
    coverage_p: float = Field(0.0, ge=0.0, le=1.0, description="覆盖型：ρ 比例物品整体缺某模态")
    coverage_mod: str = "desc"


class S2Config(BaseModel):
    """S2 最小核训练配置（V-REx 环境构造 + 反平凡性 OOD 评估）。

    schemes 为训练环境方案（obs 自动拆 desc 有无两环境）；ood_schemes 必须与
    训练方案**无交集**（反平凡性前提，脚本侧硬断言），用于训练未见缺失评估。
    """

    data: DataConfig
    model: ModelConfig
    meta_items: Path | None = None  # 真实模式：JSONL {"parent_asin","count","has_desc"}
    n_modalities: int = Field(3, ge=1)
    split_obs: bool = True
    beta: float = Field(1.0, ge=0.0)
    env_batch: int = Field(32, ge=1, description="每环境每步子批大小（V-REx 分层）")
    epochs: int = Field(30, ge=1)
    lr: float = Field(1e-3, gt=0.0)
    weight_decay: float = Field(0.0, ge=0.0)
    grad_clip: float | None = Field(None, gt=0.0)
    num_negatives: int = Field(100, ge=1)
    max_seq_len: int = Field(50, ge=1)
    methods: list[str] = Field(default_factory=lambda: ["vrex", "ermdrop"])
    schemes: list[S2SchemeConfig] = Field(
        default_factory=lambda: [
            S2SchemeConfig(name="obs"),
            S2SchemeConfig(name="mcar", mcar_p=0.5),
            S2SchemeConfig(name="mnar", mnar_rate=0.5),
        ]
    )
    ood_schemes: list[S2SchemeConfig] = Field(
        default_factory=lambda: [
            S2SchemeConfig(name="mcar", mcar_p=0.3),
            S2SchemeConfig(name="mcar", mcar_p=0.7),
            S2SchemeConfig(name="cover", coverage_p=0.5, coverage_mod="image"),
            S2SchemeConfig(name="mnar", mnar_rate=0.9),
        ]
    )
    device: str = "auto"
    seed: int = 2026
    eval_batch_size: int = Field(256, ge=1)
    topk: list[int] = Field(default_factory=lambda: [10, 20])


def load_s2_config(path: str | Path) -> S2Config:
    """从 YAML 加载 S2 配置。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return S2Config.model_validate(raw)
