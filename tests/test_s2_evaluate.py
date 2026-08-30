import numpy as np
import torch

from mrseqrec.models.miss_sasrec import MissingnessAwareSASRec
from mrseqrec.s2.evaluate import EnvEvalDataset, evaluate_schemes, retention_curve
from mrseqrec.s2.trainer import InvariantTrainer


def _trained_model(data, env_data):
    model = MissingnessAwareSASRec(
        data.item_vocab_size, n_modalities=3, hidden_dim=16, num_layers=1,
        num_heads=2, max_seq_len=10,
    )
    tr = InvariantTrainer(
        model, env_data, data.item_vocab_size, torch.device("cpu"),
        lr=1e-2, num_negatives=20, env_batch=8, max_seq_len=10, beta=0.0, seed=0,
    )
    tr.fit(epochs=1, method="ermdrop")
    return model


def test_env_eval_dataset_shapes(s2_testbed):
    data, _, counts, _ = s2_testbed
    realization = np.zeros((data.item_vocab_size, 3), dtype=bool)
    realization[1:] = True
    ds = EnvEvalDataset(data.valid_input_seqs, data.valid_targets, data.valid_negatives, realization, max_len=10)
    inp, inp_avail, cand, cand_avail = ds[0]
    assert inp.ndim == 1 and inp_avail.shape == (10, 3)
    assert cand.ndim == 1 and cand_avail.shape == (len(cand), 3)


def test_evaluate_schemes_and_retention(s2_testbed):
    """OOD 评估：obs 参考 + 未见类型（cover/missing-image）/未见缺失率，指标有限且保留率定义正确。"""
    data, base_avail, counts, env_data = s2_testbed
    model = _trained_model(data, env_data)
    schemes = [
        {"name": "obs"},
        {"name": "mcar", "mcar_p": 0.3},
        {"name": "mcar", "mcar_p": 0.7},
        {"name": "cover", "coverage_p": 0.5, "coverage_mod": "image"},
        {"name": "mnar", "mnar_rate": 0.9},
    ]
    met = evaluate_schemes(
        model, data.valid_input_seqs, data.valid_targets, data.valid_negatives,
        base_avail, counts, data.item_vocab_size, schemes,
        max_len=10, batch_size=32, device=torch.device("cpu"), topks=[10], seed=0,
    )
    assert "obs" in met and "mcar_p0.3" in met and "cover_image_p0.5" in met
    for k, v in met.items():
        r = v["recall@10"]
        assert np.isfinite(r) and 0.0 <= r <= 1.0, f"{k} recall 越界: {r}"
    ret = retention_curve(met, reference="obs", topk=10)
    assert ret["obs"] == 100.0
    assert all(r is None or r >= 0 for r in ret.values())
