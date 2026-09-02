import json

import numpy as np
import pandas as pd
import torch

from mrseqrec.models.sasrec import SASRec
from mrseqrec.utils.config import Config


# ---------------------------------------------------------------- 纯函数：两级映射

def _item_map_file(tmp_path):
    p = tmp_path / "Health.items.jsonl"
    p.write_text(
        '{"parent_asin": "A101", "code": 101}\n'
        '{"parent_asin": "A102", "code": 102}\n'
        '{"parent_asin": "A103", "code": 103}\n'
        '{"parent_asin": "A999", "code": 999}\n',
        encoding="utf-8",
    )
    return p


def test_load_code_to_asin(tmp_path):
    from scripts.rank_dump import load_code_to_asin

    m = load_code_to_asin(_item_map_file(tmp_path))
    assert m[101] == "A101" and m[999] == "A999"
    assert isinstance(list(m.keys())[0], int)  # Health 码为整数键


def test_example_rows_chain_and_skip():
    from scripts.rank_dump import example_rows

    # new_id → Health 码 → parent_asin 全通 / 单级断链 / 两级都断
    new_to_code = {1: 101, 2: 102, 3: 103}
    code_to_asin = {101: "A101", 102: "A102"}  # 103 在 items 映射里缺失
    targets = np.array([1, 2, 3, 999])
    ranks = np.array([0, 7, 2, 5])
    rows, skipped = example_rows(targets, ranks, new_to_code, code_to_asin)
    assert rows == [("A101", 0), ("A102", 7)]
    assert skipped == 2  # new_id 3 有码无 asin、new_id 999 无码


# ---------------------------------------------------------------- 闭环：打分 → dump

def _mini_df():
    # 两个用户 × 三个 Health 码物品（各重复两次过 k-core=2）；末尾物品不同
    rows = [
        (0, 101, 1.0), (0, 102, 2.0), (0, 103, 3.0), (0, 103, 4.0), (0, 101, 5.0), (0, 102, 6.0),
        (1, 101, 7.0), (1, 102, 8.0), (1, 103, 9.0), (1, 102, 10.0), (1, 103, 11.0), (1, 101, 12.0),
    ]
    return pd.DataFrame(rows, columns=["user", "item", "ts"]).astype({"ts": float})


def _mini_config():
    return Config.model_validate({
        "data": {"interactions_file": "x.txt", "min_interactions": 2, "max_seq_len": 20,
                 "num_negatives": 2, "seed": 0},
        "model": {"hidden_dim": 16, "num_layers": 1, "num_heads": 2, "dropout": 0.1},
        "train": {"device": "cpu", "seed": 0},
        "eval": {"topk": [10], "batch_size": 8},
    })


def test_dump_ranks_full_chain(tmp_path, monkeypatch):
    """k-core 划分 → 打分 → new_id→code→parent_asin 两级映射 → JSONL。

    user0 末尾=102 → A102；user1 末尾=101 → A101。随机权重即可（只测链路与口径）。
    """
    import scripts.rank_dump as mod

    monkeypatch.setattr(mod, "load_interactions", lambda *a, **k: _mini_df())
    cfg = _mini_config()
    # 与配置同架构的随机权重 ckpt（不训练；rank 口径只依赖打分路径）
    model = SASRec(4, hidden_dim=16, num_layers=1, num_heads=2, dropout=0.1, max_seq_len=20)
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)
    out = tmp_path / "test_ranks.jsonl"

    stats = mod.dump_ranks(cfg, ckpt, _item_map_file(tmp_path), out, split="test")
    assert stats["n_examples"] == 2 and stats["n_written"] == 2 and stats["n_skipped"] == 0

    got = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert sorted(r["item"] for r in got) == ["A101", "A102"]  # test 目标为每用户末位物品
    for r in got:
        assert 0 <= r["rank"] < 3  # 候选 = 1 正 + 2 负


def test_dump_ranks_valid_split_uses_same_map(tmp_path, monkeypatch):
    """valid 划分（倒数第二位）同样经两级映射写出。"""
    import scripts.rank_dump as mod

    monkeypatch.setattr(mod, "load_interactions", lambda *a, **k: _mini_df())
    cfg = _mini_config()
    model = SASRec(4, hidden_dim=16, num_layers=1, num_heads=2, dropout=0.1, max_seq_len=20)
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)
    out = tmp_path / "valid_ranks.jsonl"

    stats = mod.dump_ranks(cfg, ckpt, _item_map_file(tmp_path), out, split="valid")
    assert stats["n_examples"] == 2 and stats["n_written"] == 2
    got = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    # user0 seq[-2]=101 → A101；user1 seq[-2]=103 → A103
    assert sorted(r["item"] for r in got) == ["A101", "A103"]


def test_dump_ranks_expect_dims_guard(tmp_path, monkeypatch):
    """口径守卫：期望用户数/词表与本次 preprocess 不一致必须拒绝（防静默错位）。"""
    import pytest
    import scripts.rank_dump as mod

    monkeypatch.setattr(mod, "load_interactions", lambda *a, **k: _mini_df())
    cfg = _mini_config()
    model = SASRec(4, hidden_dim=16, num_layers=1, num_heads=2, dropout=0.1, max_seq_len=20)
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)
    out = tmp_path / "test_ranks.jsonl"

    with pytest.raises(ValueError, match="用户数"):
        mod.dump_ranks(cfg, ckpt, _item_map_file(tmp_path), out, expect_n_users=3)
    with pytest.raises(ValueError, match="词表"):
        mod.dump_ranks(cfg, ckpt, _item_map_file(tmp_path), out, expect_vocab=99)
    # 期望值对得上 → 正常执行到写出
    stats = mod.dump_ranks(cfg, ckpt, _item_map_file(tmp_path), out, expect_n_users=2, expect_vocab=4)
    assert stats["n_written"] == 2


def test_write_examples_format(tmp_path):
    from scripts.rank_dump import write_examples

    out = tmp_path / "e.jsonl"
    write_examples(out, [("A1", 0), ("A2", 42)])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"item": "A1", "rank": 0}
    assert json.loads(lines[1]) == {"item": "A2", "rank": 42}
