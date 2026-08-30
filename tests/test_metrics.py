import numpy as np

from mrseqrec.eval.metrics import evaluate, rank_of_positive


def test_rank_of_positive():
    scores = np.array(
        [[0.5, 0.1, 0.2],  # 正例(0)=0.5 rank 0
         [0.5, 0.9, 0.2],  # 正例=0.5 次高 rank 1
         [0.1, 0.9, 0.5]]  # 正例=0.1 最低 rank 2
    )
    assert rank_of_positive(scores).tolist() == [0, 1, 2]


def test_recall_and_ndcg():
    scores = np.array([[0.9, 0.1], [0.1, 0.9]])
    res = evaluate(scores, [1, 2])
    assert res["recall@1"] == 0.5
    assert res["recall@2"] == 1.0
    assert res["ndcg@1"] == 0.5  # rank0 命中 1 个
    assert abs(res["ndcg@2"] - (1.0 + 1.0 / np.log2(3)) / 2) < 1e-9


def test_rank_nan_guard():
    """含 NaN 的行必须记最差位次，不能被 argsort 的 NaN-最后 行为误判为正例第 1 名。

    回归：评估行含 NaN 曾使指标虚高到 recall@10=0.99。
    """
    scores = np.array([
        [np.nan, np.nan],  # 全 NaN → 最差位次 C=2
        [0.9, np.nan],     # 含 NaN → 最差位次 2
        [0.9, 0.1],        # 正常 → rank 0
    ])
    assert rank_of_positive(scores).tolist() == [2, 2, 0]
