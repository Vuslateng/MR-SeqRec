"""S2 训练环境构造（§2.4/§2.6）：缺失方案 → 物品可用性实现 → 环境数据集。

环境 = 缺失方案（scheme）。每个方案对**同一批物品**施加不同缺失配置，产出该
方案的物品可用性实现 realization（(V, M) 位，V=词表含 pad，M=模态数）：
- obs（观测基）：自然可用性（来自 meta），无缺失注入。训练例按序列末位物品的
  自然配置拆成 desc 有无两个环境（§2.4"2 个可观测环境"）；
- mcar / mnar / cover（合成增广）：对全部物品受控缺失（复用 sampler.py），
  构造极端缺失率/未见类型环境。同一物品在不同方案下可用性不同——环境差异纯粹
  反映缺失（§2.4 纯度要求），V-REx 的 Var_e(R_e) 才有意义。

反平凡性（三审 3.1）的"同采样强度"由结构保证：V-REx 与 SMD 式 dropout+ERM
共用同一套方案构造的数据，仅损失不同（见 trainer.py）。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from mrseqrec.data.dataset import right_pad
from mrseqrec.missing import sampler

MODALITIES = sampler.MODALITIES  # ("text", "image", "desc")，列顺序与 mod_emb 行一致


def scheme_key(scheme: dict) -> str:
    """方案 → 确定的环境标签（同时作为种子派生与 OOD 去重依据）。"""
    name = scheme["name"]
    if name == "mcar":
        return f"mcar_p{scheme.get('mcar_p', 0.0):g}"
    if name == "mnar":
        return f"mnar_r{scheme.get('mnar_rate', 0.0):g}"
    if name == "cover":
        return f"cover_{scheme.get('coverage_mod', 'desc')}_p{scheme.get('coverage_p', 0.0):g}"
    return name  # "obs"


def scheme_realization(
    scheme: dict,
    base_avail: dict,
    counts: np.ndarray | None,
    vocab: int,
    seed: int = 0,
) -> np.ndarray:
    """方案实现：(V, M) bool；id 0（pad）恒全 False。

    base_avail : {item_id: {"text": bool, "image": bool, "desc": bool}}，自然可用性；
    counts     : (V,) 流行度，mnar 方案必需（内部由 sample_missingness 断言）。
    obs 方案不注入缺失，直接用自然可用性。
    """
    m = len(MODALITIES)
    bits = np.zeros((vocab, m), dtype=bool)
    name = scheme["name"]
    if name == "obs":
        for it, a in base_avail.items():
            bits[it] = [bool(a.get(x, False)) for x in MODALITIES]
    else:
        # counts 与 base_avail 对齐：本管线传入 (V,) 全词表（id 0=pad，不在 base_avail
        # 内）。mnar 通道按流行度选缺 desc 物品，若直接 zip 会错位一格（item j 读到
        # item j-1 的流行度、pad 的 0 计数占用一个"最冷门"名额）——先剥掉 pad 前缀。
        c = np.asarray(counts, dtype=float) if counts is not None else None
        if c is not None and len(c) != len(base_avail):
            assert len(c) == len(base_avail) + 1, \
                f"counts({len(c)}) 须与 base_avail({len(base_avail)}) 对齐或为含 pad 的全词表"
            c = c[1:]
        res = sampler.sample_missingness(
            base_avail,
            c,
            mcar_p=scheme.get("mcar_p", 0.0),
            coverage_p=scheme.get("coverage_p", 0.0),
            coverage_mod=scheme.get("coverage_mod", "desc"),
            mnar_rate=scheme.get("mnar_rate", 0.0),
            seed=seed,
        )
        for it, (avail, _) in res.items():
            bits[it] = [bool(avail.get(x, False)) for x in MODALITIES]
    bits[0] = False
    return bits


@dataclass
class EnvExample:
    """一个训练例：环境标签 + 用户序列（物品 id）。可用性由方案实现导出。"""

    env: str
    seq: np.ndarray  # (n,) int64


@dataclass
class EnvData:
    """环境数据集：环境 → 训练例 + 每个环境的物品可用性实现。"""

    examples: dict[str, list[EnvExample]]
    realizations: dict[str, np.ndarray]  # env → (V, M) bool
    env_order: list[str]


def _natural_env_id(realization: np.ndarray, item_id: int) -> str:
    bits = realization[item_id]
    return sampler.env_id({MODALITIES[i]: bool(bits[i]) for i in range(len(MODALITIES))})


def build_environments(
    train_seqs: list[np.ndarray],
    base_avail: dict,
    counts: np.ndarray | None,
    vocab: int,
    schemes: list[dict],
    split_obs: bool = True,
    seed: int = 0,
) -> EnvData:
    """由训练序列 + 物品可用性构造环境数据集。

    - obs 且 split_obs=True：按序列末位物品自然配置拆为 obs:xxx / obs:yyy 两个环境；
      两个环境共用同一自然实现（分组不同，V-REx 对二者分别计 R_e）。
    - 其余方案：每个方案一个环境，所有训练序列各一份拷贝（同一批物品，不同缺失配置）。
    - 空环境（0 例）剔除并告警，防止 Var_e 含 NaN。
    """
    examples: dict[str, list[EnvExample]] = defaultdict(list)
    realizations: dict[str, np.ndarray] = {}
    for s in schemes:
        label = scheme_key(s)
        bits = scheme_realization(s, base_avail, counts, vocab, seed)
        if s["name"] == "obs" and split_obs:
            for seq in train_seqs:
                env = f"obs:{_natural_env_id(bits, int(seq[-1]))}"
                examples[env].append(EnvExample(env, seq))
            for env in [e for e in examples if e.startswith("obs:")]:
                realizations[env] = bits
        else:
            realizations[label] = bits
            for seq in train_seqs:
                examples[label].append(EnvExample(label, seq))

    env_order = [e for e in realizations if examples.get(e)]
    dropped = [e for e in realizations if not examples.get(e)]
    if dropped:
        print(f"[environ] 空环境剔除（0 训练例）：{dropped}")
    return EnvData({e: examples[e] for e in env_order}, {e: realizations[e] for e in env_order}, env_order)


def collate_env(examples: list[EnvExample], realization: np.ndarray, max_len: int):
    """批量打包（同环境）→ (input_ids, target_ids, avail)，全为 numpy。

    input = 右 pad 的序列；target = 右移一位（末位补 pad），即全前缀监督；
    avail = realization[input_ids]，pad 位置（id 0）恒 False。
    """
    seqs = [e.seq[-max_len:] for e in examples]
    input_ids = np.stack([right_pad(s, max_len) for s in seqs])
    targets = []
    for s in seqs:
        t = np.roll(s, -1)
        t[-1] = 0
        targets.append(right_pad(t, max_len))
    target_ids = np.stack(targets)
    avail = realization[input_ids]
    return input_ids, target_ids, avail
