"""S2 最小核：缺失不变学习（V-REx 环境构造 + 损失 + 反平凡性 OOD 评估）。

模块划分（§2.6 ④"MNAR 建模 + 不变学习"）：
- environ.py  缺失方案 → 环境数据集（obs 观测基 + MCAR/MNAR/cover 合成增广）；
- trainer.py  V-REx 与 SMD 式 dropout+ERM 对照训练；
- evaluate.py 训练未见缺失方案上的 OOD 排序指标与 retention 曲线。
"""

from mrseqrec.s2.environ import (
    EnvData,
    EnvExample,
    build_environments,
    collate_env,
    scheme_key,
    scheme_realization,
)
from mrseqrec.s2.evaluate import (
    EnvEvalDataset,
    env_rank_metrics,
    evaluate_schemes,
    retention_curve,
)
from mrseqrec.s2.trainer import InvariantTrainer, vrex_loss

__all__ = [
    "EnvData",
    "EnvExample",
    "build_environments",
    "collate_env",
    "scheme_key",
    "scheme_realization",
    "EnvEvalDataset",
    "env_rank_metrics",
    "evaluate_schemes",
    "retention_curve",
    "InvariantTrainer",
    "vrex_loss",
]
