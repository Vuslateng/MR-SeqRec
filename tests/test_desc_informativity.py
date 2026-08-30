import json
import sys

import numpy as np

from scripts.desc_informativity import (
    bootstrap_delta_ci,
    bucket_examples,
    examples_from_ranks,
    layer_summary,
    stratify,
    verdict,
)


def test_stratify_rank_based():
    logc = np.array([0.1, 0.2, 0.3, 0.4, 5.0, 5.1, 5.2, 5.3, 10.0, 10.1])
    s = stratify(logc, n_strata=5)
    assert sorted(np.bincount(s)) == [2, 2, 2, 2, 2]  # 等物品数
    order = np.argsort(logc)
    assert list(s[order]) == sorted(s[order])  # 流行度升序 → 档号不降
    assert s[order[0]] == 0 and s[order[-1]] == 4  # 0=最冷门


def test_examples_from_ranks():
    hits, ndcgs = examples_from_ranks(["a", "b", "c"], np.array([0, 5, 20]), k=10)
    assert hits.tolist() == [1.0, 1.0, 0.0]
    assert abs(ndcgs[0] - 1.0) < 1e-9  # rank0 → 1/log2(2)=1
    assert abs(ndcgs[1] - 1.0 / np.log2(7.0)) < 1e-9
    assert ndcgs[2] == 0.0


def test_bucket_examples():
    # item 奇偶同时决定 stratum 与 desc，形成对角格，便于验证
    stratum_of = {f"I{i}": i % 2 for i in range(10)}
    desc_of = {f"I{i}": (i % 2 == 0) for i in range(10)}
    items = [f"I{i}" for i in range(10)]
    hits = np.ones(10)
    ndcgs = np.ones(10) * 0.5
    cells = bucket_examples(items, hits, ndcgs, stratum_of, desc_of, n_strata=2)
    assert cells[(0, True)]["n"] == 5  # I0,I2,I4,I6,I8
    assert cells[(1, False)]["n"] == 5  # I1,I3,I5,I7,I9
    assert (0, False) not in cells and (1, True) not in cells
    assert cells[(0, True)]["recall"] == 1.0


def test_layer_summary_delta():
    cells = {
        (0, True): {"n": 100, "recall": 0.50, "ndcg": 0.30},
        (0, False): {"n": 100, "recall": 0.40, "ndcg": 0.24},
        (1, True): {"n": 100, "recall": 0.60, "ndcg": 0.38},
        (1, False): {"n": 100, "recall": 0.45, "ndcg": 0.27},
    }
    rows = layer_summary(cells, n_strata=2)
    assert rows[0]["delta_recall"] == 0.10
    assert rows[1]["delta_recall"] == 0.15
    assert rows[0]["rel_delta_recall"] == 0.2


def test_bootstrap_delta_ci():
    rng = np.random.default_rng(0)
    a = rng.beta(8, 4, size=2000)  # mean≈0.667
    b = rng.beta(6, 6, size=2000)  # mean≈0.5
    ci = bootstrap_delta_ci(a, b, n_boot=500, seed=1)
    assert ci[0] > 0  # 强分离 → CI 不含 0
    assert bootstrap_delta_ci(a, b, n_boot=500, seed=1) == ci  # 种子可复现
    assert bootstrap_delta_ci(a, np.array([]), n_boot=10, seed=1) is None


def test_verdict_rules():
    def layer(lo, hi, n=500):
        return {"ci_recall": (lo, hi), "n_ok": n, "n_missing": n}

    assert verdict([layer(0.02, 0.08), layer(0.01, 0.06)])[0] == "环境轴成立"
    assert verdict([layer(-0.02, 0.03), layer(-0.01, 0.02)])[0] == "环境轴价值弱"
    assert verdict([layer(0.02, 0.08), layer(-0.05, -0.01)])[0] == "信号混杂"
    assert verdict([layer(-0.05, -0.01), layer(-0.04, -0.02)])[0] == "信号混杂"  # 全负异常
    assert verdict([{"ci_recall": None, "n_ok": 50, "n_missing": 50}])[0] == "样本不足"


def test_cli_smoke(tmp_path):
    from scripts import desc_informativity

    ex = tmp_path / "examples.jsonl"
    meta = tmp_path / "meta.jsonl"
    with open(ex, "w", encoding="utf-8") as f:
        for i in range(406):
            rank = 0 if i % 4 else 50
            it = f"Z999" if i % 203 == 0 else f"I{i % 20}"  # 每 203 例混入一个不在物品集的物品
            f.write(json.dumps({"item": it, "rank": rank}) + "\n")
    with open(meta, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(
                json.dumps({"parent_asin": f"I{i}", "count": i + 1, "has_desc": (i % 2 == 0)}) + "\n"
            )
    out = tmp_path / "out.json"
    old = sys.argv
    sys.argv = ["desc_informativity", "--examples", str(ex), "--meta-items", str(meta), "--out", str(out)]
    try:
        desc_informativity.main()
    finally:
        sys.argv = old
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n_examples"] == 406
    assert data["n_skipped"] == 2  # Z999 混入 2 例（i=0 与 i=203），其余全进判读
    assert data["n_scored"] == 404
    assert "verdict" in data and "layers" in data
