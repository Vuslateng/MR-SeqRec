import numpy as np

from mrseqrec.missing.sampler import (
    env_distribution,
    env_id,
    mcar_corrupt,
    mnar_select,
    sample_missingness,
)


def test_env_id():
    assert env_id({"text": True, "image": True, "desc": True}) == "desc+image+text"
    assert env_id({"text": False, "image": True, "desc": False}) == "image"
    assert env_id({"text": False, "image": False, "desc": False}) == "none"
    # 与传入顺序无关
    assert env_id({"desc": True, "text": True, "image": False}) == "desc+text"


def test_mcar_corrupt_boundaries():
    rng = np.random.default_rng(0)
    base = {"text": True, "image": True, "desc": True}
    assert mcar_corrupt(base, 0.0, rng) == base  # p=0 不变
    assert mcar_corrupt(base, 1.0, rng) == {
        "text": False, "image": False, "desc": False
    }
    # 缺失过的模态不会被"补"回来
    base2 = {"text": False, "image": True, "desc": False}
    out = mcar_corrupt(base2, 0.5, rng)
    assert out["text"] is False and out["desc"] is False
    assert out["image"] in (True, False)


def test_mnar_select_direction():
    counts = np.arange(1, 21, dtype=float)  # 严格递增：冷门在前
    mask = mnar_select(counts, 0.3)
    assert mask.sum() == 6  # 20 × 0.3 = 6
    assert mask[:6].all() and not mask[6:].any()  # 冷门前 6 个被选


def test_sample_missingness_diversity():
    base = {f"I{i}": {"text": True, "image": True, "desc": True} for i in range(20)}
    counts = np.arange(1, 21, dtype=float)

    # MCAR 通道开 → 环境多样化
    r = sample_missingness(base, counts, mcar_p=0.3, seed=1)
    assert len(env_distribution(r)) >= 2

    # MNAR 通道：冷门前 50% 的 desc 被移除，热门不丢
    r2 = sample_missingness(base, counts, mnar_rate=0.5, seed=1)
    cold = [e for it, (_, e) in r2.items() if it in {"I0", "I1", "I2", "I3", "I4"}]
    hot = [e for it, (_, e) in r2.items() if it in {"I15", "I16", "I17", "I18", "I19"}]
    assert all("desc" not in e for e in cold)
    assert all("desc" in e for e in hot)

    # 覆盖通道：10% 物品整类缺 desc
    r3 = sample_missingness(base, counts, coverage_p=0.1, coverage_mod="desc", seed=1)
    n_missing_desc = sum(1 for a, _ in r3.values() if not a["desc"])
    assert n_missing_desc == 2  # 20 × 0.1 = 2
