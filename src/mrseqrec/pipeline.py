"""训练-评估流水线：数据 → 划分 → 训练 → 评估 → 汇总。供 CLI 与测试复用。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from mrseqrec.data.dataset import EvalDataset, SeqTrainDataset
from mrseqrec.data.io import load_interactions
from mrseqrec.data.preprocess import preprocess
from mrseqrec.eval.ranking import RankingEvaluator
from mrseqrec.eval.retention import compute_retention, RetentionReport
from mrseqrec.models.sasrec import SASRec
from mrseqrec.trainers.trainer import Trainer
from mrseqrec.utils.config import Config
from mrseqrec.utils.device import resolve_device
from mrseqrec.utils.log import get_logger
from mrseqrec.utils.seed import set_seed

logger = get_logger("pipeline")


@dataclass
class PipelineResult:
    metrics: dict[str, float] = field(default_factory=dict)   # valid/test 各指标
    losses: list[float] = field(default_factory=list)
    retention: RetentionReport | None = None                  # S1 对 ID 基线恒为 rho=0 单点
    n_users: int = 0
    item_vocab_size: int = 0


def run_training(config: Config, save_dir: Path | None = None) -> PipelineResult:
    """完整训练评估流程。ID-only SASRec 对缺失免疫，retention 报告为单点（rho=0）。"""
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)
    logger.info("device=%s data=%s", device, config.data.interactions_file)

    df = load_interactions(config.data.interactions_file)
    data = preprocess(
        df,
        min_interactions=config.data.min_interactions,
        num_negatives=config.data.num_negatives,
        seed=config.data.seed,
    )
    logger.info("users=%d vocab=%d", data.n_users, data.item_vocab_size)

    train_ds = SeqTrainDataset(data.train_seqs, config.data.max_seq_len)
    valid_ds = EvalDataset(data.valid_input_seqs, data.valid_targets, data.valid_negatives, config.data.max_seq_len)
    test_ds = EvalDataset(data.test_input_seqs, data.test_targets, data.test_negatives, config.data.max_seq_len)

    model = SASRec(
        item_vocab_size=data.item_vocab_size,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        max_seq_len=config.data.max_seq_len,
    )
    trainer = Trainer(
        model,
        device=device,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        grad_clip=config.train.grad_clip,
        log_every=config.train.log_every,
        num_negatives=config.train.num_negatives,
        item_vocab_size=data.item_vocab_size,
    )
    losses = trainer.fit(train_ds, epochs=config.train.epochs, batch_size=config.train.batch_size)

    evaluator = RankingEvaluator(model, topks=config.eval.topk, batch_size=config.eval.batch_size, device=device)
    valid_metrics = evaluator.evaluate(valid_ds)
    test_metrics = evaluator.evaluate(test_ds)
    metrics = {f"valid/{k}": v for k, v in valid_metrics.items()}
    metrics.update({f"test/{k}": v for k, v in test_metrics.items()})
    logger.info("valid=%s", valid_metrics)
    logger.info("test=%s", test_metrics)

    retention = compute_retention({0.0: valid_metrics[f"recall@{config.eval.topk[0]}"]})
    logger.info("\n%s", retention)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")
        (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return PipelineResult(
        metrics=metrics, losses=losses, retention=retention,
        n_users=data.n_users, item_vocab_size=data.item_vocab_size,
    )
