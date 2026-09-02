"""S1 测试例 positive-rank dump（服务器批次 ⑤）——喂给 §2.7 desc 信息性诊断。

S1 eval 在 preprocess 重编号的 new_id 空间打分（候选 = 正例 idx0 + K 负例，
rank = rank_of_positive 0-based 位次，含 NaN 守卫）。desc 信息性诊断要求
--examples 的物品 id 落在 --meta-items 的 parent_asin（字符串）键空间，故此处
做两级反向映射：
    new_id ─(data.item_map)→ Health 原始码(整数) ─(Health.items.jsonl)→ parent_asin

复现性（口径与 S1 指标严格一致）：同 config + 同 interactions 文件 + 同 seed
重放划分与负采样（不重训），打分走 RankingEvaluator 同一条路径——左 pad 取
hidden[-1]、score_candidates 点积、rank_of_positive 判位。test 划分即用户
末位物品（"用户下一交互物品"），与 §2.7 判读对象一致。

用法（服务器，S1 ckpt 之后）：
  python scripts/rank_dump.py --config configs/s1_amazon.yaml \
      --checkpoint outputs/s1_amazon/model.pt \
      --item-map data/amazon/Health.items.jsonl \
      --split test --out outputs/data_check/test_ranks.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from mrseqrec.data.dataset import EvalDataset
from mrseqrec.data.io import load_interactions
from mrseqrec.data.preprocess import preprocess
from mrseqrec.eval.metrics import rank_of_positive
from mrseqrec.models.sasrec import SASRec
from mrseqrec.utils.config import Config, load_config
from mrseqrec.utils.device import resolve_device
from mrseqrec.utils.seed import set_seed


# ---------------------------------------------------------------- 映射（纯函数）

def load_code_to_asin(path: str | Path) -> dict[int, str]:
    """读 fetch_amazon 落盘的 Health.items.jsonl → {Health 码: parent_asin 字符串}。"""
    out: dict[int, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[int(r["code"])] = str(r["parent_asin"])
    return out


def example_rows(
    targets: np.ndarray,
    ranks: np.ndarray,
    new_to_code: dict[int, int],
    code_to_asin: dict[int, str],
) -> tuple[list[tuple[str, int]], int]:
    """new_id 例 → (parent_asin, rank) 行；映射断链的例跳过并计数。

    new_to_code : preprocess 的 item_map（new_id → Health 原始码）；
    code_to_asin: Health.items.jsonl（Health 码 → parent_asin）。
    两级任一缺失即跳过——与 desc_informativity 的 n_skipped 同义（物品不在物品集）。
    """
    rows: list[tuple[str, int]] = []
    skipped = 0
    for t, r in zip(targets, ranks):
        code = new_to_code.get(int(t))
        if code is None:
            skipped += 1
            continue
        asin = code_to_asin.get(int(code))
        if asin is None:
            skipped += 1
            continue
        rows.append((asin, int(r)))
    return rows, skipped


def write_examples(out: Path, rows: list[tuple[str, int]]) -> None:
    """写 desc_informativity 的 --examples JSONL: {"item": asin, "rank": rank}。"""
    with open(out, "w", encoding="utf-8") as f:
        for asin, rank in rows:
            f.write(json.dumps({"item": asin, "rank": rank}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 打分

@torch.no_grad()
def score_examples(model: SASRec, eval_ds, batch_size: int, device: torch.device) -> np.ndarray:
    """逐例给候选集合打分（与 RankingEvaluator.evaluate 同路径），返回 (U, C)。"""
    model.eval()
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    parts: list[np.ndarray] = []
    for input_ids, candidates in loader:
        input_ids = input_ids.to(device)
        candidates = candidates.to(device)
        hidden = model(input_ids)[:, -1, :]  # 左 pad → 末位 = 最近物品
        parts.append(model.score_candidates(hidden, candidates).float().cpu().numpy())
    return np.concatenate(parts, axis=0)


def _split_arrays(data, split: str):
    if split == "test":
        return data.test_input_seqs, data.test_targets, data.test_negatives
    if split == "valid":
        return data.valid_input_seqs, data.valid_targets, data.valid_negatives
    raise ValueError(f"split 必须为 test/valid，got {split}")


def dump_ranks(
    config: Config,
    checkpoint: str | Path,
    item_map_path: str | Path,
    out: str | Path,
    split: str = "test",
    expect_n_users: int | None = None,
    expect_vocab: int | None = None,
) -> dict:
    """完整 dump 闭环：数据划分 → 打分 → 两级映射 → 写 JSONL。返回统计。

    口径自检（防"Health.txt 与 S1 训练时不一致"静默错位）：
      - 词表一致性是**硬守卫**：ckpt 的 item_emb 与本次 preprocess 词表同构
        （load_state_dict strict 形状校验），Health.txt 物品域变了会直接报错；
      - 可选 expect_n_users/expect_vocab：把 S1 训练日志里的用户数/词表传进来，
        对不上立即报错（防"物品域巧合相同、划分却变了"的静默错位）。
    """
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)
    df = load_interactions(config.data.interactions_file)
    data = preprocess(
        df,
        min_interactions=config.data.min_interactions,
        num_negatives=config.data.num_negatives,
        seed=config.data.seed,
    )
    if expect_n_users is not None and data.n_users != expect_n_users:
        raise ValueError(
            f"用户数 {data.n_users} ≠ 期望 {expect_n_users}：interactions 与 S1 训练时不一致，"
            f"rank 与 S1 指标不可比，拒绝 dump"
        )
    if expect_vocab is not None and data.item_vocab_size != expect_vocab:
        raise ValueError(
            f"词表 {data.item_vocab_size} ≠ 期望 {expect_vocab}：interactions 与 S1 训练时不一致"
        )
    inputs, targets, negatives = _split_arrays(data, split)

    model = SASRec(
        item_vocab_size=data.item_vocab_size,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        max_seq_len=config.data.max_seq_len,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    eval_ds = EvalDataset(inputs, targets, negatives, config.data.max_seq_len)
    scores = score_examples(model, eval_ds, config.eval.batch_size, device)
    ranks = rank_of_positive(scores)

    code_to_asin = load_code_to_asin(item_map_path)
    rows, skipped = example_rows(targets, ranks, data.item_map, code_to_asin)
    write_examples(Path(out), rows)
    return {
        "split": split, "n_users": data.n_users, "item_vocab_size": data.item_vocab_size,
        "n_examples": len(targets), "n_written": len(rows), "n_skipped": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/s1_amazon.yaml")
    p.add_argument("--checkpoint", default="outputs/s1_amazon/model.pt", help="S1 已训练 model.pt")
    p.add_argument("--item-map", default="data/amazon/Health.items.jsonl", help="fetch_amazon 落盘的 Health.items.jsonl")
    p.add_argument("--split", choices=["test", "valid"], default="test")
    p.add_argument("--out", default="outputs/data_check/test_ranks.jsonl")
    p.add_argument("--expect-n-users", type=int, default=None,
                   help="S1 训练日志里的用户数（对不上即报错，防 Health.txt 与训练时不一致）")
    p.add_argument("--expect-vocab", type=int, default=None,
                   help="S1 训练日志里的词表大小（含 pad；对不上即报错）")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    stats = dump_ranks(
        config, args.checkpoint, args.item_map, out, split=args.split,
        expect_n_users=args.expect_n_users, expect_vocab=args.expect_vocab,
    )
    # 口径自检锚点：S1 kcore10 基线为 ~136,118 用户 / 62,727 物品（词表 62,728，含 pad）
    print(f"口径自检: 用户 {stats['n_users']:,}  词表 {stats['item_vocab_size']:,}"
          + ("  （与 S1 期望一致 ✓）" if args.expect_n_users or args.expect_vocab else ""))
    print(f"split={stats['split']}: 例 {stats['n_examples']:,}  写出 {stats['n_written']:,}  跳过 "
          f"{stats['n_skipped']}（两级映射断链，物品不在 meta-items 物品集）")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
