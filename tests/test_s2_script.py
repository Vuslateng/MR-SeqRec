from pathlib import Path

import pytest

from mrseqrec.utils.config import S2Config, load_s2_config


def _cfg(**over):
    d = {
        "data": {"interactions_file": "x.txt", "min_interactions": 5, "num_negatives": 20},
        "model": {"hidden_dim": 16, "num_layers": 1, "num_heads": 2},
        "n_modalities": 3,
        "beta": 0.5,
        "env_batch": 8,
        "epochs": 1,
        "lr": 1e-2,
        "num_negatives": 20,
        "max_seq_len": 10,
        "methods": ["vrex", "ermdrop"],
        "schemes": [
            {"name": "obs"},
            {"name": "mcar", "mcar_p": 0.5},
            {"name": "mnar", "mnar_rate": 0.5},
        ],
        "ood_schemes": [
            {"name": "mcar", "mcar_p": 0.3},
            {"name": "cover", "coverage_p": 0.5, "coverage_mod": "image"},
        ],
        "seed": 0,
        "eval_batch_size": 32,
        "topk": [10],
    }
    d.update(over)
    return S2Config.model_validate(d)


def test_run_s2_closed_loop(tmp_path):
    """完整闭环：合成数据 → 环境 → 双方法训练 → OOD 评估 → 落盘 JSON+checkpoint。"""
    from scripts.s2_train import run_s2

    out = run_s2(_cfg(), "synthetic", None, tmp_path, n_users=120, n_items=60)
    assert set(out["methods"]) == {"vrex", "ermdrop"}
    for m in ("vrex", "ermdrop"):
        assert "obs" in out[m]["metrics"]
        assert out[m]["retention"]["obs"] == 100.0
        assert out[m]["losses"][-1] > 0.0
    assert (tmp_path / "s2_result.json").exists()
    assert (tmp_path / "vrex.pt").exists() and (tmp_path / "ermdrop.pt").exists()


def test_run_s2_ood_overlap_guard(tmp_path):
    """反平凡性守卫：OOD 方案与训练方案重复必须报错。"""
    from mrseqrec.utils.config import S2SchemeConfig
    from scripts.s2_train import run_s2

    cfg = _cfg()
    cfg.ood_schemes = [
        S2SchemeConfig(name="mcar", mcar_p=0.5),  # 与训练 mcar_p0.5 重复
    ]
    with pytest.raises(AssertionError):
        run_s2(cfg, "synthetic", None, tmp_path, n_users=120, n_items=60)


def test_run_s2_real_mode_requires_meta(tmp_path):
    from scripts.s2_train import run_s2

    with pytest.raises(ValueError, match="real"):
        run_s2(_cfg(), "real", None, tmp_path, n_users=120, n_items=60)


def test_build_real_data_meta_alignment(tmp_path, monkeypatch):
    """真实模式：meta 经 item-map（Health 码）两级查表对齐；未匹配回退 df 计数 + desc 缺失。"""
    import pandas as pd
    import scripts.s2_train as mod

    cfg = _cfg(data={"interactions_file": "x.txt", "min_interactions": 2, "num_negatives": 5})
    rows = ["0 101 1.0", "0 102 2.0", "0 101 3.0", "0 102 4.0",
            "1 101 5.0", "1 102 6.0", "1 101 7.0", "1 102 8.0"]
    df = pd.DataFrame([r.split() for r in rows], columns=["user", "item", "ts"]).astype(
        {"ts": float, "user": "int64", "item": "int64"})  # Health 码即整数 item
    monkeypatch.setattr(mod, "load_interactions", lambda *a, **k: df)
    item_map = tmp_path / "Health.items.jsonl"  # fetch_amazon 落盘格式
    item_map.write_text(
        '{"parent_asin": "A101", "code": 101}\n'
        '{"parent_asin": "A102", "code": 102}\n'
        '{"parent_asin": "A999", "code": 999}\n',
        encoding="utf-8",
    )
    meta = tmp_path / "meta.jsonl"
    meta.write_text(
        '{"parent_asin": "A101", "count": 5, "has_desc": true}\n'
        '{"parent_asin": "A999", "count": 9, "has_desc": true}\n',  # A999 不在 k-core 人群 → 忽略
        encoding="utf-8",
    )
    data, base_avail, counts = mod._build_real_data(cfg, meta, item_map)
    assert data.item_vocab_size == 3  # pad + {101, 102}
    assert base_avail[1] == {"text": True, "image": True, "desc": True}   # A101 匹配 meta
    assert base_avail[2] == {"text": True, "image": True, "desc": False}  # A102 无 meta → 回退 desc 缺失
    assert counts[1] == 5.0  # meta.count 覆盖
    assert counts[2] == 4.0  # df 实际交互计数打底


def test_build_real_data_requires_item_map(monkeypatch):
    """真实模式缺 item-map 必须拒绝：否则 meta 无法对齐，全部物品静默 desc 缺失。"""
    import pandas as pd
    import scripts.s2_train as mod

    monkeypatch.setattr(
        mod, "load_interactions",
        lambda *a, **k: pd.DataFrame({"user": [0, 0, 0], "item": [1, 2, 3], "ts": [1.0, 2.0, 3.0]}),
    )
    with pytest.raises(ValueError, match="item-map"):
        mod._build_real_data(_cfg(), Path("meta.jsonl"), None)


def test_s2_config_yaml_valid():
    """仓库自带 configs/s2_vrex.yaml 可加载且训练/OOD 方案无重叠。"""
    cfg = load_s2_config("configs/s2_vrex.yaml")
    assert cfg.beta == 1.0 and cfg.split_obs is True
    assert len(cfg.schemes) == 3 and len(cfg.ood_schemes) == 4
    from mrseqrec.s2.environ import scheme_key

    train_keys = {scheme_key(s.model_dump()) for s in cfg.schemes}
    ood_keys = {scheme_key(s.model_dump()) for s in cfg.ood_schemes}
    assert not (train_keys & ood_keys)
