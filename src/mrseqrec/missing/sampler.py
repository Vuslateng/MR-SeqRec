"""缺失环境采样（关口 2 §2.6，喂给 §2.4 V-REx 的环境划分）。

环境 = 物品的有效可用性配置（§2.4）。三类缺失通道（数据核查 §3.3 确定）：
- MCAR 比例型（mcar_p）：每个可用模态以 p 独立缺失——合成增广的基线通道；
- 覆盖型（coverage_p / coverage_mod）：ρ 比例物品整体缺某一模态，默认 desc
  （desc 是真实数据唯一的自然缺失通道，image≈100%/text≈99.99% 不构成环境轴）；
- MNAR 型（mnar_rate）：按倾向选缺失率最高的一批物品缺 desc。倾向与流行度
  负向（β̂<0），即冷门物品更易缺 desc——与数据核查方向一致。

通道可叠加（消融用）：三通道全开模拟"混合缺失"环境。
"""

from __future__ import annotations

from collections import Counter

import numpy as np

MODALITIES = ("text", "image", "desc")


def env_id(avail: dict) -> str:
    """由物品有效可用性得到环境 id（模态排序固定，保证 id 稳定）。

    例：{"text": True, "image": True, "desc": False} → "text+image"。
    """
    return "+".join(sorted(m for m in MODALITIES if avail.get(m, False))) or "none"


def mcar_corrupt(base: dict, p: float, rng: np.random.Generator) -> dict:
    """MCAR 比例型：每个可用模态以概率 p 独立缺失；返回新字典，不改 base。"""
    return {m: (bool(base.get(m, False)) and rng.random() >= p) for m in MODALITIES}


def mnar_select(counts: np.ndarray, rate: float) -> np.ndarray:
    """MNAR 型：选"缺失率最高"的 rate 比例物品缺 desc（冷门优先，与 β̂<0 一致）。

    counts : 物品流行度数组；rate ∈ (0,1)。返回 (n,) bool 掩码，True=该物品 desc 缺失。
    取 log1p 后按流行度升序取前 rate 比例——冷门在前。
    """
    n = len(counts)
    k = int(round(n * rate))
    order = np.argsort(np.log1p(np.asarray(counts, dtype=float)), kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def sample_missingness(
    base_avail: dict,
    counts=None,
    *,
    mcar_p: float = 0.0,
    coverage_p: float = 0.0,
    coverage_mod: str = "desc",
    mnar_rate: float = 0.0,
    seed: int = 0,
):
    """对每个物品施加缺失，返回 {item: (有效可用性 dict, 环境id)}。

    base_avail : {item: {"text": bool, "image": bool, "desc": bool}}——物品基础可用性，
                 来自真实数据（meta 里有没有 title/images/description 字段）。
    counts     : 流行度数组，顺序与 base_avail 的 key 一致；mnar_rate>0 时必需。
    三个通道可同时开（消融用）。seed 固定保证可复现。
    """
    rng = np.random.default_rng(seed)
    items = list(base_avail.keys())
    n = len(items)

    cov_items = set()
    if coverage_p > 0:
        k = min(n, int(round(n * coverage_p)))
        cov_items = set(rng.choice(items, size=k, replace=False))

    mnar_items = set()
    if mnar_rate > 0:
        # 显式要求的通道不允许被静默跳过：缺 counts 直接报错而非忽略
        assert counts is not None, "mnar_rate>0 时必须提供 counts（顺序与 base_avail 一致）"
        c = np.asarray(counts, dtype=float)
        # 防御断言：长度不对齐会让 zip 错位（item j 读到 item j-1 的流行度），
        # 错位结果"看起来正常"但冷门判选偏移，禁止静默发生
        assert len(c) == len(items), f"counts({len(c)}) 长度须与 items({len(items)}) 对齐"
        mask = mnar_select(c, mnar_rate)
        mnar_items = {it for it, m in zip(items, mask) if m}

    out = {}
    for it, base in base_avail.items():
        avail = {m: bool(base.get(m, False)) for m in MODALITIES}
        if mcar_p > 0:
            avail = mcar_corrupt(avail, mcar_p, rng)
        if it in cov_items:
            avail[coverage_mod] = False
        if it in mnar_items:
            avail["desc"] = False
        out[it] = (avail, env_id(avail))
    return out


def env_distribution(result: dict) -> dict:
    """环境 id 分布（验证多样性用）。result 为 sample_missingness 的返回值。"""
    return dict(Counter(eid for _, eid in result.values()))
