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
