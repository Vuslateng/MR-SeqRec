import numpy as np
import torch

from mrseqrec.models.miss_sasrec import MissingnessAwareSASRec
from mrseqrec.s2.trainer import InvariantTrainer, vrex_loss


def _model(data):
    return MissingnessAwareSASRec(
        data.item_vocab_size, n_modalities=3, hidden_dim=16, num_layers=1,
        num_heads=2, max_seq_len=10,
    )


def _trainer(model, data, env_data, **kw):
    kw.setdefault("lr", 1e-2)
    kw.setdefault("num_negatives", 20)
    kw.setdefault("env_batch", 16)
    kw.setdefault("max_seq_len", 10)
    kw.setdefault("beta", 1.0)
    kw.setdefault("seed", 0)
    return InvariantTrainer(model, env_data, data.item_vocab_size, torch.device("cpu"), **kw)


def test_vrex_loss_math():
    """ΣR_e + β·Var_e(R_e)（总体方差）；β=0 退化为 ERM（ΣR_e）。"""
    r = [torch.tensor(0.5), torch.tensor(0.7), torch.tensor(0.6)]
    mean = 0.6
    var = sum((x - mean) ** 2 for x in [0.5, 0.7, 0.6]) / 3
    loss = vrex_loss(r, beta=2.0)
    assert abs(loss.item() - (1.8 + 2.0 * var)) < 1e-6
    assert abs(vrex_loss(r, 0.0).item() - 1.8) < 1e-6


def test_trainer_rejects_unknown_method(s2_testbed):
    data, _, _, env_data = s2_testbed
    tr = _trainer(_model(data), data, env_data)
    try:
        tr.fit(epochs=1, method="irmv1")
        assert False, "应拒绝未知 method"
    except ValueError:
        pass


def test_trainer_vrex_ermdrop_decrease(s2_testbed):
    """两种方法端到端训练：损失有限且下降；V-REx 有方差项、ERM 无。"""
    data, _, _, env_data = s2_testbed
    for method, beta in [("vrex", 1.0), ("ermdrop", 0.0)]:
        model = _model(data)
        tr = _trainer(model, data, env_data, beta=beta)
        losses = tr.fit(epochs=3, method=method)
        assert all(np.isfinite(l) for l in losses), f"{method} 损失含 NaN/inf"
        assert losses[-1] < losses[0], f"{method} 损失未下降: {losses}"


def test_trainer_full_vocab_branch(s2_testbed):
    """小词表（vocab ≤ num_negatives+1）走全词表 CE 分支，含可用性，损失有限。"""
    data, _, _, env_data = s2_testbed
    model = _model(data)
    tr = _trainer(model, data, env_data, num_negatives=100, env_batch=8, beta=1.0)  # vocab=60 ≤ 101
    losses = tr.fit(epochs=1, method="vrex")
    assert all(np.isfinite(l) for l in losses)


def test_preprocess_item_map(s2_testbed):
    """item_map 覆盖全部非 pad 物品 id（S2 对齐 meta 的前提）。"""
    data, _, _, _ = s2_testbed
    assert len(data.item_map) == data.item_vocab_size - 1
    assert set(data.item_map) == set(range(1, data.item_vocab_size))
