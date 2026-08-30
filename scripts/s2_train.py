"""S2 最小核训练闭环（§2.6 ④）：MNAR 建模 + 缺失不变学习。

数据 → 缺失环境构造 → 训练 V-REx 与 SMD 式 dropout+ERM 对照 → 在训练未见缺失
方案上 OOD 评估（反平凡性，§2.4 OOD 协议）→ 输出对比 JSON（含 retention 曲线）。

用法：
  # 合成模式（本地冒烟，无真实数据依赖）：
  python scripts/s2_train.py --mode synthetic --s2-config configs/s2_vrex.yaml \
      --save-dir outputs/s2_minimal
  # 真实模式（服务器，需交互文件 + meta 可用性 + item-map 映射）：
  python scripts/s2_train.py --mode real --s2-config configs/s2_vrex.yaml \
      --meta <meta-items.jsonl> --item-map <Health.items.jsonl> --save-dir outputs/s2_minimal

反平凡性硬守卫：ood_schemes 与训练 schemes 标签无交集，否则报错退出（不允许把
训练见过的缺失配置当作 OOD）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from mrseqrec.data.io import load_interactions
from mrseqrec.data.preprocess import preprocess
from mrseqrec.data.synthetic import generate_interactions
from mrseqrec.models.miss_sasrec import MissingnessAwareSASRec
from mrseqrec.missing.sampler import mnar_select
from mrseqrec.s2.environ import build_environments, scheme_key
from mrseqrec.s2.evaluate import evaluate_schemes, retention_curve
from mrseqrec.s2.trainer import InvariantTrainer
from mrseqrec.utils.config import S2Config, load_s2_config
from mrseqrec.utils.device import resolve_device
from mrseqrec.utils.seed import set_seed

NATURAL_DESC_MISSING = 0.38  # 真实观测：desc 整体缺失 38.23%，冷门更缺（β̂<0）


def _build_synthetic_data(config: S2Config, n_users: int, n_items: int):
    """合成数据：交互 → 预处理 → 自然可用性（冷门缺 desc，仿真实 β̂<0）。"""
    df = generate_interactions(
        n_users=n_users, n_items=n_items, min_len=5, max_len=config.max_seq_len,
        alpha=0.6, seed=config.data.seed,
    )
    data = preprocess(df, min_interactions=config.data.min_interactions,
                      num_negatives=config.data.num_negatives, seed=config.data.seed)
    counts_orig = df.groupby("item").size()
    counts = np.zeros(data.item_vocab_size, dtype=float)
    base_avail: dict = {}
    for new_id, orig in data.item_map.items():
        counts[new_id] = float(counts_orig.get(orig, 1))
        base_avail[new_id] = {"text": True, "image": True, "desc": True}
    # 自然 desc 缺失：冷门（低流行度）缺 desc，rate≈38%
    miss_mask = mnar_select(counts[1:], NATURAL_DESC_MISSING)
    for new_id in data.item_map:
        base_avail[new_id]["desc"] = not miss_mask[new_id - 1]
    return data, base_avail, counts


def _build_real_data(config: S2Config, meta_path: Path, item_map_path: Path | None):
    """真实数据：交互预处理 + meta 可用性（has_desc / count）对齐到重编号 id。

    物品 id 空间链（fetch_amazon 契约）：Health.txt 的 item 是 parent_asin 的整数码，
    其 ↔parent_asin 映射由 Health.items.jsonl（item_map_path）落盘保存。meta 的
    parent_asin 是字符串，须经两级查表：meta parent_asin →(item_map_path) Health 码
    →(data.item_map) new_id。缺 item_map_path 拒绝运行——否则全部物品静默走 desc
    缺失回退，摧毁 S2 依赖的自然 MNAR 信号（无法事后发现）。

    查表均为 O(1)/行：真实规模 V~9万、meta~30万行时禁止在循环内线性扫 item_map
    （O(1e10) 级，服务器会卡死）。counts 先取 df 实际交互计数打底，匹配 meta 再覆盖。
    """
    if item_map_path is None:
        raise ValueError("real 模式必须提供 --item-map（fetch_amazon 落盘的 parent_asin↔code 映射）")
    df = load_interactions(config.data.interactions_file)
    data = preprocess(df, config.data.min_interactions,
                      config.data.num_negatives, config.data.seed)
    counts_orig = df.groupby("item").size()
    code_to_new = {o: i for i, o in data.item_map.items()}
    counts = np.zeros(data.item_vocab_size, dtype=float)
    for new_id, orig in data.item_map.items():
        counts[new_id] = float(counts_orig.get(orig, 1))
    asin_to_code: dict[str, int] = {}
    for line in item_map_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        asin_to_code[str(r["parent_asin"])] = int(r["code"])
    base_avail: dict = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        assert {"parent_asin", "has_desc"} <= r.keys(), f"meta 行缺字段: {line.strip()}"
        code = asin_to_code.get(str(r["parent_asin"]))
        if code is None:
            continue  # 不在 Health 物品集 → 忽略
        new_id = code_to_new.get(code)
        if new_id is None:
            continue  # 非 k-core 人群（训练人群外），忽略，避免人群错配（§2.6）
        counts[new_id] = float(r.get("count", counts[new_id]))
        base_avail[new_id] = {"text": True, "image": True, "desc": bool(r["has_desc"])}
    missing = [i for i in range(1, data.item_vocab_size) if i not in base_avail]
    if missing:
        print(f"[s2] 警告：{len(missing)} 个 k-core 物品无 meta 记录，按 text+image 处理（desc 缺失）")
        for i in missing:
            base_avail[i] = {"text": True, "image": True, "desc": False}
    return data, base_avail, counts


def _report(out: dict, topk: int) -> None:
    print("=" * 72)
    print("S2 最小核结果")
    print("=" * 72)
    print(f"训练环境（{len(out['envs'])}）：{'  '.join(out['envs'])}")
    for method in out["methods"]:
        m = out[method]
        print(f"\n[{method}] 训练损失末值 {m['losses'][-1]:.4f}")
        print(f"  {'方案':<22}{'Recall@%d' % topk:>10}{'保留率%':>10}")
        for k, v in m["metrics"].items():
            ret = m["retention"].get(k)
            ret_s = f"{ret:>9.2f}" if ret is not None else "     n/a"
            print(f"  {k:<22}{v[f'recall@{topk}'] * 100:>9.2f}{ret_s}")
    print("-" * 72)
    print("反平凡性判读：对比 V-REx 与 ermdrop 在训练未见缺失方案上的保留率。")


def run_s2(config: S2Config, mode: str, meta_path: Path | None, save_dir: Path,
           n_users: int = 5000, n_items: int = 2000,
           item_map_path: Path | None = None) -> dict:
    """完整闭环。返回结构化结果（同时落盘 s2_result.json + 各方法 checkpoint）。"""
    set_seed(config.seed)
    device = resolve_device(config.device)
    if mode == "synthetic":
        data, base_avail, counts = _build_synthetic_data(config, n_users, n_items)
    else:
        if meta_path is None:
            raise ValueError("real 模式必须提供 --meta")
        data, base_avail, counts = _build_real_data(config, meta_path, item_map_path)

    # 反平凡性守卫：训练方案与 OOD 方案标签不得重复
    train_keys = {scheme_key(s.model_dump()) for s in config.schemes}
    ood_keys = {scheme_key(s.model_dump()) for s in config.ood_schemes}
    overlap = train_keys & ood_keys
    assert not overlap, f"OOD 方案与训练方案重复（反平凡性前提被破坏）：{overlap}"

    schemes_train = [s.model_dump() for s in config.schemes]
    env_data = build_environments(
        data.train_seqs, base_avail, counts, data.item_vocab_size,
        schemes_train, split_obs=config.split_obs, seed=config.seed,
    )
    schemes_eval = [{"name": "obs"}] + [s.model_dump() for s in config.ood_schemes]
    save_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {"envs": env_data.env_order, "methods": list(config.methods)}
    for method in config.methods:
        # 重置全局种子：两方法同初始权重 + 同负例采样流（_sample_batch 用独立 per-epoch
        # rng 不变），SMD 对照严格只差 β·Var_e(R_e)，反平凡性隔离最大化
        set_seed(config.seed)
        model = MissingnessAwareSASRec(
            data.item_vocab_size, n_modalities=config.n_modalities,
            hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers,
            num_heads=config.model.num_heads, dropout=config.model.dropout,
            max_seq_len=config.max_seq_len,
        )
        trainer = InvariantTrainer(
            model, env_data, data.item_vocab_size, device=device,
            lr=config.lr, weight_decay=config.weight_decay, grad_clip=config.grad_clip,
            num_negatives=config.num_negatives, env_batch=config.env_batch,
            max_seq_len=config.max_seq_len, beta=config.beta, seed=config.seed,
        )
        losses = trainer.fit(config.epochs, method)
        metrics = evaluate_schemes(
            model, data.valid_input_seqs, data.valid_targets, data.valid_negatives,
            base_avail, counts, data.item_vocab_size, schemes_eval,
            max_len=config.max_seq_len, batch_size=config.eval_batch_size,
            device=device, topks=config.topk, seed=config.seed,
        )
        retention = retention_curve(metrics, reference="obs", topk=config.topk[0])
        out[method] = {"losses": losses, "metrics": metrics, "retention": retention}
        torch.save(model.state_dict(), save_dir / f"{method}.pt")

    (save_dir / "s2_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out, config.topk[0])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--s2-config", default="configs/s2_vrex.yaml")
    p.add_argument("--meta", default=None, help="真实模式：meta-items.jsonl（parent_asin/count/has_desc）")
    p.add_argument("--item-map", default=None, help="真实模式：fetch_amazon 落盘的 Health.items.jsonl（parent_asin↔code）")
    p.add_argument("--save-dir", default="outputs/s2_minimal")
    p.add_argument("--n-users", type=int, default=5000, help="合成模式用户数")
    p.add_argument("--n-items", type=int, default=2000, help="合成模式物品数")
    args = p.parse_args(argv)
    config = load_s2_config(args.s2_config)
    run_s2(config, args.mode, Path(args.meta) if args.meta else None,
           Path(args.save_dir), n_users=args.n_users, n_items=args.n_items,
           item_map_path=Path(args.item_map) if args.item_map else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
