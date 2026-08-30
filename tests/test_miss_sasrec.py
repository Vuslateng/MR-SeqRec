import torch

from mrseqrec.models.miss_sasrec import MissingnessAwareSASRec
from mrseqrec.models.sasrec import SASRec


def _model(**kw):
    kw.setdefault("item_vocab_size", 11)
    kw.setdefault("n_modalities", 3)
    kw.setdefault("hidden_dim", 16)
    kw.setdefault("num_layers", 1)
    kw.setdefault("num_heads", 2)
    kw.setdefault("max_seq_len", 8)
    return MissingnessAwareSASRec(**kw)


def test_forward_shape_with_avail():
    model = _model()
    x = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]])
    avail = torch.zeros(1, 8, 3, dtype=torch.bool)
    avail[0, 0, 2] = True  # 位置 0 物品 1 有 desc
    h = model(x, avail)
    assert h.shape == (1, 8, 16)
    assert torch.isfinite(h).all()


def test_avail_all_false_equals_base_embedding():
    """avail 全 False → 表示 = 基础嵌入（模态标志项全部不加入）。"""
    model = _model()
    x = torch.tensor([[5]])
    a0 = torch.zeros(1, 1, 3, dtype=torch.bool)
    r0 = model.item_representation(x, a0)
    assert torch.allclose(r0, model.item_emb(torch.tensor([[5]])))


def test_avail_changes_representation():
    """可用模态翻转变表示——同一物品在不同缺失配置下表示不同（环境敏感）。"""
    model = _model()
    x = torch.tensor([[5]])
    a0 = torch.zeros(1, 1, 3, dtype=torch.bool)
    a1 = torch.zeros(1, 1, 3, dtype=torch.bool)
    a1[0, 0, 0] = True  # text 可用
    r0 = model.item_representation(x, a0)
    r1 = model.item_representation(x, a1)
    assert not torch.allclose(r0, r1)


def test_avail_none_equals_base_sasrec():
    """avail=None 时与 S1 的 SASRec 逐位一致（权重同步后输出相同）。"""
    model = _model()
    base = SASRec(11, hidden_dim=16, num_layers=1, num_heads=2, max_seq_len=8)
    for sub in ("item_emb", "pos_emb", "encoder", "ln"):
        getattr(model, sub).load_state_dict(getattr(base, sub).state_dict())
    x = torch.tensor([[1, 2, 3, 0, 0]])
    model.eval()
    base.eval()
    with torch.no_grad():
        assert torch.allclose(model(x, None), base(x), atol=1e-6)


def test_score_candidates_with_candidate_avail():
    model = _model()
    with torch.no_grad():
        h = model.item_emb(torch.tensor([[1]]))[:, -1, :]  # (1, d)
        cand = torch.tensor([[1, 2]])
        ca = torch.zeros(1, 2, 3, dtype=torch.bool)
        ca[0, 0, 0] = True
        s = model.score_candidates(h, cand, ca)
    assert s.shape == (1, 2)
    assert torch.isfinite(s).all()


def test_causal_no_peek_with_avail():
    """未来物品的可用性变化不影响过去位置输出（因果掩码仍成立）。"""
    model = _model()
    model.eval()
    x1 = torch.tensor([[1, 2, 3, 0, 0]])
    x2 = torch.tensor([[1, 2, 9, 0, 0]])
    a1 = torch.zeros(1, 5, 3, dtype=torch.bool)
    a1[0, 2, 2] = True
    a2 = torch.zeros(1, 5, 3, dtype=torch.bool)
    a2[0, 2, 2] = False
    with torch.no_grad():
        h1, h2 = model(x1, a1), model(x2, a2)
    assert torch.allclose(h1[:, 1, :], h2[:, 1, :], atol=1e-6)
