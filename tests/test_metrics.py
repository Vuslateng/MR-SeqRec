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
