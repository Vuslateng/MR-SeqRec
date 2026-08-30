import torch

from mrseqrec.models.sasrec import SASRec


def test_forward_shape():
    model = SASRec(item_vocab_size=11, hidden_dim=16, num_layers=1, num_heads=2, max_seq_len=8)
    x = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]])
    h = model(x)
    assert h.shape == (1, 8, 16)


def test_score_candidates():
    model = SASRec(item_vocab_size=11, hidden_dim=16, num_layers=1, num_heads=2, max_seq_len=8)
    model.eval()
    with torch.no_grad():
        x = torch.tensor([[1, 2, 3]])
        h = model(x)[:, -1, :]
        s = model.score_candidates(h, torch.tensor([[3, 5]]))
    assert s.shape == (1, 2)
    assert torch.isfinite(s).all()


def test_causal_no_peek():
    """位置 t 的输出只依赖 ≤t：改变未来 token 不应影响过去的预测。"""
    model = SASRec(item_vocab_size=11, hidden_dim=16, num_layers=1, num_heads=2, max_seq_len=8)
    model.eval()
    with torch.no_grad():
        x1 = torch.tensor([[1, 2, 3, 0, 0]])
        x2 = torch.tensor([[1, 2, 9, 0, 0]])
        h1, h2 = model(x1), model(x2)
    assert torch.allclose(h1[:, 1, :], h2[:, 1, :], atol=1e-6)


def test_attn_mask_no_full_mask_row():
    """掩码每行至少保留一个有限键（对角线）——左填充下 pad 行不再全掩码。

    回归：评估用左填充，pad 行曾全掩码触发 CUDA fused softmax 的 NaN，
    导致指标被 NaN 虚高（recall@10 虚报到 0.99）。
    """
    model = SASRec(item_vocab_size=11, hidden_dim=16, num_layers=1, num_heads=2, max_seq_len=8)
    x = torch.tensor([[0, 0, 0, 0, 0, 1, 2, 3]])  # 左填充：前 5 位 pad，后 3 位真实
    mask = model._attn_mask(x)  # (1, 8, 8)
    assert (mask == 0.0).any(dim=-1).all(), "每行至少一个有限键"


def test_left_pad_output_finite():
    """左填充（评估形态）任何输入下输出必须有限，不得含 NaN。"""
    model = SASRec(item_vocab_size=11, hidden_dim=16, num_layers=2, num_heads=2, max_seq_len=8)
    model.eval()
    with torch.no_grad():
        x = torch.tensor([[0, 0, 0, 0, 0, 1, 2, 3]])
        h = model(x)
    assert torch.isfinite(h).all()
