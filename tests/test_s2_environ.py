import numpy as np

from mrseqrec.s2.environ import (
    EnvExample,
    build_environments,
    collate_env,
    scheme_key,
    scheme_realization,
)


def test_scheme_key():
    assert scheme_key({"name": "obs"}) == "obs"
    assert scheme_key({"name": "mcar", "mcar_p": 0.5}) == "mcar_p0.5"
    assert scheme_key({"name": "mnar", "mnar_rate": 0.9}) == "mnar_r0.9"
    assert scheme_key({"name": "cover", "coverage_p": 0.5, "coverage_mod": "image"}) == "cover_image_p0.5"


def test_obs_realization_natural():
    base_avail = {i: {"text": True, "image": True, "desc": (i % 2 == 0)} for i in range(1, 11)}
    bits = scheme_realization({"name": "obs"}, base_avail, None, vocab=11, seed=0)
    assert bits[0].all() == False  # pad 恒无模态
    assert bits[3][2] == False  # 奇号物品 desc 缺失
    assert bits[4][2] == True  # 偶号物品 desc 存在


def test_mcar_realization_corrupts_same_items():
    """MCAR：模态以 ~p 独立缺失；不引入新物品（id 空间不变，row 0 恒 False）。"""
    base_avail = {i: {"text": True, "image": True, "desc": True} for i in range(1, 21)}
    bits = scheme_realization({"name": "mcar", "mcar_p": 0.5}, base_avail, None, vocab=21, seed=0)
    rate = bits[1:].mean()
    assert 0.2 < rate < 0.8  # 期望 ~0.5
    assert bits[0].all() == False
    nonzero = set(np.argwhere(bits[1:].any(axis=1)).ravel() + 1)
    assert nonzero <= set(range(1, 21))


def test_mnar_realization_cold_missing():
    """MNAR：冷门（低流行度）优先缺 desc，与数据核查 β̂<0 方向一致。"""
    counts = np.arange(1, 21, dtype=float)  # id 越大越热门
    base_avail = {i: {"text": True, "image": True, "desc": True} for i in range(1, 21)}
    bits = scheme_realization({"name": "mnar", "mnar_rate": 0.5}, base_avail, counts, vocab=21, seed=0)
    desc_missing = [i for i in range(1, 21) if not bits[i][2]]
    assert len(desc_missing) == 10  # 50% 缺 desc
    assert max(desc_missing) < 15  # 缺的集中在冷门（高 id 的热门保留 desc）
    assert bits[20][2] == True  # 最热门必不缺


def test_mnar_realization_pad_offset_counts():
    """管线传 (V,) 含 pad 全词表 counts 时，须剥 pad 前缀后再 mnar 判选（防 zip 错位一格）。"""
    base_avail = {i: {"text": True, "image": True, "desc": True} for i in range(1, 21)}
    counts_full = np.concatenate([[0.0], np.arange(1, 21, dtype=float)])  # (V,)，id0=pad
    bits = scheme_realization({"name": "mnar", "mnar_rate": 0.5}, base_avail, counts_full, vocab=21, seed=0)
    missing = [i for i in range(1, 21) if not bits[i][2]]
    assert len(missing) == 10  # 50% 缺 desc
    # 与对齐版（无 pad）逐位一致——若不剥 pad，冷门判选会偏移一格且结果不同
    counts_align = np.arange(1, 21, dtype=float)
    bits2 = scheme_realization({"name": "mnar", "mnar_rate": 0.5}, base_avail, counts_align, vocab=21, seed=0)
    assert np.array_equal(bits, bits2)


def test_build_environments_split_obs_and_purity():
    train_seqs = [np.array([1, 2, 4]), np.array([1, 3, 5]), np.array([2, 4, 6])]
    # desc 存在：偶号（2,4,6）；末位物品：4(desc)/5(无)/6(desc) → 两观测环境都有
    base_avail = {i: {"text": True, "image": True, "desc": (i % 2 == 0)} for i in range(1, 8)}
    counts = np.arange(1, 8, dtype=float)
    schemes = [{"name": "obs"}, {"name": "mcar", "mcar_p": 0.5}, {"name": "mnar", "mnar_rate": 0.5}]
    data = build_environments(train_seqs, base_avail, counts, vocab=8, schemes=schemes, seed=0)
    # env_id 按模态字母序排序（sampler 规范）：desc+image+text / image+text
    # obs 拆 desc 有无两环境 + 2 个合成环境 = 4 个
    assert set(data.env_order) == {"obs:desc+image+text", "obs:image+text", "mcar_p0.5", "mnar_r0.5"}
    assert len(data.examples["obs:desc+image+text"]) == 2  # 末位 4,6
    assert len(data.examples["obs:image+text"]) == 1  # 末位 5
    assert len(data.examples["mcar_p0.5"]) == 3
    assert len(data.examples["mnar_r0.5"]) == 3
    # 两个 obs 环境共用同一自然实现（分组不同）
    assert np.array_equal(data.realizations["obs:desc+image+text"], data.realizations["obs:image+text"])
    # 纯度：合成环境不改变物品 id 空间（row 0 False，非零行 ⊆ {1..7}）
    for env in data.env_order:
        bits = data.realizations[env]
        assert bits[0].any() == False
        assert set(np.argwhere(bits.any(axis=1)).ravel()) <= set(range(1, 8))
    # 同一物品在不同环境可用性不同（缺失操纵真实存在）
    obs = data.realizations["obs:desc+image+text"]
    mcar = data.realizations["mcar_p0.5"]
    assert not np.array_equal(obs, mcar)
    # mnar_r0.5：冷门（1,2,3,4）缺 desc
    mnar = data.realizations["mnar_r0.5"]
    assert mnar[6][2] == True  # 热门 id 6 保留 desc
    assert mnar[1][2] == False


def test_build_environments_no_split_obs_single_env():
    train_seqs = [np.array([1, 2, 4]), np.array([1, 3, 5])]
    base_avail = {i: {"text": True, "image": True, "desc": (i % 2 == 0)} for i in range(1, 7)}
    counts = np.arange(1, 7, dtype=float)
    data = build_environments(
        train_seqs, base_avail, counts, vocab=7,
        schemes=[{"name": "obs"}], split_obs=False, seed=0,
    )
    assert data.env_order == ["obs"]
    assert len(data.examples["obs"]) == 2


def test_collate_env():
    exs = [EnvExample("e", np.array([1, 2, 3])), EnvExample("e", np.array([4, 5]))]
    realization = np.zeros((11, 3), dtype=bool)
    realization[1, 2] = True
    realization[2, 2] = True
    realization[3, 2] = True
    realization[4, 1] = True
    realization[5, 1] = True
    input_ids, target_ids, avail = collate_env(exs, realization, max_len=5)
    assert input_ids.shape == (2, 5) and target_ids.shape == (2, 5) and avail.shape == (2, 5, 3)
    assert input_ids[0].tolist() == [1, 2, 3, 0, 0]
    assert target_ids[0].tolist() == [2, 3, 0, 0, 0]  # 右移一位，末位补 pad
    assert target_ids[1].tolist() == [5, 0, 0, 0, 0]
    assert avail[0, 0, 2] == True and avail[0, 2, 2] == True  # 物品 1,3 有 desc
    assert avail[0, 3].any() == False  # pad 行恒 False
    assert avail[1, 0, 1] == True  # 物品 4 有 image
