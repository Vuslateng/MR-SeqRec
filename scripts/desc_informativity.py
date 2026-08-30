"""desc 信息性诊断（§2.7）：判定 desc 缺失值不值得当 V-REx 环境轴。

核心：在流行度 5 档内，比较 next-item 为 desc 缺失 vs desc 完整 的测试例
Recall@k/NDCG@k 之差 Δ，判读以**层内 bootstrap 区间**为准（5% 只作初筛）。

输入（服务器上由既有管线产出，本脚本只做聚合与判读）：
  --examples   每测试例的 positive rank（S1 eval，rank_of_positive 的 0-based 位次）
               JSONL: {"item": <id>, "rank": <int>}
  --meta-items 物品级 count + desc 有无（与 k-core 物品集一致）
               JSONL: {"parent_asin": <id>, "count": <int>, "has_desc": <bool>}
  两文件的 item id 空间必须一致（同 k-core 物品集）。

判读规则（§2.7，写死）：
  - 各层 Δ CI 不含 0 且方向一致为正 → 环境轴成立，支持"缺失承载信号"；
  - 各层 Δ CI 含 0 → desc 缺失不引入额外困难 → 环境轴价值弱（写作降级）；
  - Δ 符号层间冲突 → 信号混杂，逐层报告，不宣称整体方向。
  类目分组不在此脚本（n 稀疏时降级为描述统计，§2.7）。

判读口径（写死，防误读）：
  - 聚合/判读对象是**测试例的 next-item**（用户下一交互物品）的 desc 有无，**不是**
    用户历史序列的缺失环境——§2.7 的目标是"desc 缺失物品是否更难被推荐对"；
  - 判读只基于 Recall 的 Δ bootstrap 区间；NDCG 的 Δ 及其区间为辅助展示，**不参与判读**；
  - n_examples 为原始例数，n_scored 为进入判读的例数，
    n_skipped = n_examples − n_scored（物品不在物品集而被跳过），报告必须给出。

边界（§2.7 "它不回答什么"）：本诊断用全量训练 checkpoint，测的是"模型已见过
缺失物品后的残余困难"，不是"缺失在训练之初是否构成学习障碍"（那是强化案 A/E4）；
只回答"desc 缺失值不值得当环境轴"，不回答"V-REx 在真实 desc 缺失上会赢"。
"""

import argparse
import json
import os
import sys

import numpy as np

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


# ---------------------------------------------------------------- 分层


def stratify(log_counts: np.ndarray, n_strata: int = 5) -> np.ndarray:
    """按 log1p(count) 升序等分成 n_strata 档，返回每物品档号（0=最冷门）。

    秩分位（等物品数）而非数值分位：保证每档物品量均衡，避免超热门物品独占一档。
    n_strata > n_items 时多余档为空（调用方自行处理空格）。
    """
    n = len(log_counts)
    if n == 0 or n_strata <= 1:
        return np.zeros(n, dtype=int)
    order = np.argsort(log_counts, kind="stable")
    strata = np.empty(n, dtype=int)
    strata[order] = np.arange(n) * n_strata // n
    return strata


# ---------------------------------------------------------------- 聚合


def examples_from_ranks(items: list, ranks: np.ndarray, k: int = 10):
    """由每例 positive rank 导出 hit@k 与 ndcg@k（单一来源，k 一致）。"""
    ranks = np.asarray(ranks, dtype=float)
    hits = (ranks < k).astype(float)
    ndcgs = np.where(hits, 1.0 / np.log2(ranks + 2.0), 0.0)
    return hits, ndcgs


def bucket_examples(
    items: list,
    hits: np.ndarray,
    ndcgs: np.ndarray,
    stratum_of: dict,
    desc_of: dict,
    n_strata: int,
) -> dict:
    """按 next-item 的 (流行度档, desc 有无) 聚合。返回 {(档, desc_ok): {n, recall, ndcg}}。

    stratum_of / desc_of : item id → 档号 / 是否 desc 完整。不在物品集内的 item 跳过。
    """
    cells: dict = {}
    for it, hit, ndcg in zip(items, hits, ndcgs):
        st = stratum_of.get(it)
        if st is None:
            continue
        key = (int(st), bool(desc_of.get(it, False)))
        c = cells.setdefault(key, {"n": 0, "hits": 0.0, "ndcg": 0.0})
        c["n"] += 1
        c["hits"] += float(hit)
        c["ndcg"] += float(ndcg)
    out = {}
    for (st, d), c in cells.items():
        out[(st, d)] = {"n": c["n"], "recall": c["hits"] / c["n"], "ndcg": c["ndcg"] / c["n"]}
    return out


def layer_summary(cells: dict, n_strata: int) -> list[dict]:
    """逐层汇总 Δ（Recall/NDCG）+ 每格 n + 平均 log 流行度。返回层表。"""
    rows = []
    for st in range(n_strata):
        ok = cells.get((st, True))
        miss = cells.get((st, False))
        row = {"stratum": st, "n_ok": ok["n"] if ok else 0, "n_missing": miss["n"] if miss else 0}
        if ok and miss and ok["n"] > 0 and miss["n"] > 0:
            row.update(
                recall_ok=round(ok["recall"], 4),
                recall_missing=round(miss["recall"], 4),
                ndcg_ok=round(ok["ndcg"], 4),
                ndcg_missing=round(miss["ndcg"], 4),
                delta_recall=round(ok["recall"] - miss["recall"], 4),
                delta_ndcg=round(ok["ndcg"] - miss["ndcg"], 4),
            )
            row["rel_delta_recall"] = (
                round((ok["recall"] - miss["recall"]) / ok["recall"], 4) if ok["recall"] > 0 else None
            )
        else:
            row.update(recall_ok=None, recall_missing=None, ndcg_ok=None, ndcg_missing=None,
                       delta_recall=None, delta_ndcg=None, rel_delta_recall=None)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- 区间


def bootstrap_delta_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """两个独立样本均值之差的 bootstrap 95% 区间。任一空组返回 None。"""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return None
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        da = a[rng.integers(0, na, na)].mean()
        db = b[rng.integers(0, nb, nb)].mean()
        deltas[i] = da - db
    return (round(float(np.percentile(deltas, 2.5)), 4), round(float(np.percentile(deltas, 97.5)), 4))


def verdict(layers: list[dict], n_min: int = 200) -> tuple[str, str]:
    """§2.7 判读：以各层 Δ 的 CI 为准（n 达标层），5% 只作初筛提示。

    返回 (结论, 依据)。"""
    judge = [
        l for l in layers
        if l.get("ci_recall") is not None and l["n_ok"] >= n_min and l["n_missing"] >= n_min
    ]
    if not judge:
        return ("样本不足", f"无达标层（每格 n≥{n_min}），不可判读，需重训或扩大验证集")
    signs = set()
    for l in judge:
        lo, hi = l["ci_recall"]
        if lo > 0:
            signs.add("pos")
        elif hi < 0:
            signs.add("neg")
        else:
            signs.add("zero")
    if signs == {"pos"}:
        return ("环境轴成立", f"{len(judge)}/{len(layers)} 层 desc 缺失显著掉点（CI 不含 0 且为正）")
    if signs == {"zero"}:
        return ("环境轴价值弱", f"{len(judge)} 层 CI 均含 0，desc 缺失不引入额外困难")
    if signs == {"neg"}:
        return ("信号混杂", "Δ 一致为负（desc 缺失反而更高），异常待查，勿静默接受")
    return ("信号混杂", f"层间符号不一致 {sorted(signs)}，逐层报告不宣称整体方向")


# ---------------------------------------------------------------- 报告


def _format_report(
    rows: list[dict], overall: dict, decision: tuple[str, str],
    n_examples: int, n_scored: int, n_skipped: int,
) -> str:
    lines = [
        "=" * 72,
        "desc 信息性诊断（§2.7）",
        "=" * 72,
        f"例数: 原始 {n_examples}  判读 {n_scored}  跳过 {n_skipped}（物品不在物品集）",
        "判读对象: 测试例的 next-item 的 desc 有无（非用户历史序列缺失环境）",
        "判读仅以 Recall 的 Δ 区间为准; NDCG 及其区间为辅助展示",
        "",
        "流行度档   n完整  n缺失  Recall完整  Recall缺失  ΔRecall(95%CI)      ΔNDCG(95%CI)",
    ]
    for r in rows:
        if r.get("ci_recall") and r.get("ci_ndcg"):
            dr = f"{r['delta_recall']:+.4f} [{r['ci_recall'][0]:.4f},{r['ci_recall'][1]:.4f}]"
            dn = f"{r['delta_ndcg']:+.4f} [{r['ci_ndcg'][0]:.4f},{r['ci_ndcg'][1]:.4f}]"
        else:
            dr, dn = "(n 不足)", "(n 不足)"
        miss = f"{r['recall_missing']:.4f}" if r["recall_missing"] is not None else "  -   "
        ok = f"{r['recall_ok']:.4f}" if r["recall_ok"] is not None else "  -   "
        lines.append(
            f"   {r['stratum']}      {r['n_ok']:>6}  {r['n_missing']:>6}  {ok}  {miss}  {dr:26s} {dn}"
        )
    lines.append(f"整体加权(按 n): ΔRecall={overall['delta_recall']:+.4f}  ΔNDCG={overall['delta_ndcg']:+.4f}")
    lines.append("-" * 72)
    lines.append(f"判读：{decision[0]}")
    lines.append(f"  依据：{decision[1]}")
    lines.append("  初筛提示（非判读依据）：各层相对掉点 " + "  ".join(
        f"{r['stratum']}={r['rel_delta_recall']}" if r.get("rel_delta_recall") is not None else f"{r['stratum']}=-"
        for r in rows
    ))
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--examples", required=True, help="每测试例 JSONL: {\"item\", \"rank\"}")
    p.add_argument("--meta-items", required=True, help="物品 JSONL: {\"parent_asin\", \"count\", \"has_desc\"}")
    p.add_argument("--k", type=int, default=10, help="Recall/NDCG 截断")
    p.add_argument("--n-strata", type=int, default=5, help="流行度档数")
    p.add_argument("--n-boot", type=int, default=2000, help="bootstrap 重采样次数")
    p.add_argument("--seed", type=int, default=20260830, help="bootstrap 种子")
    p.add_argument("--out", default="outputs/data_check/desc_informativity.json")
    args = p.parse_args()

    items, ranks = [], []
    with open(args.examples, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            assert "item" in r and "rank" in r, f"examples 行缺字段: {line.strip()}"
            items.append(r["item"])
            ranks.append(r["rank"])
    ranks = np.asarray(ranks, dtype=float)

    log_counts, stratum_of, desc_of = {}, {}, {}
    with open(args.meta_items, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            assert {"parent_asin", "count", "has_desc"} <= r.keys(), f"meta 行缺字段: {line.strip()}"
            pa = r["parent_asin"]
            log_counts[pa] = np.log1p(max(0, int(r["count"])))
            desc_of[pa] = bool(r["has_desc"])
    if log_counts:
        ids = list(log_counts.keys())
        strata = stratify(np.asarray([log_counts[i] for i in ids]), args.n_strata)
        stratum_of = {i: int(s) for i, s in zip(ids, strata)}

    hits, ndcgs = examples_from_ranks(items, ranks, args.k)
    cells = bucket_examples(items, hits, ndcgs, stratum_of, desc_of, args.n_strata)
    rows = layer_summary(cells, args.n_strata)

    # 每层 Δ 的 bootstrap CI（组内每例；Recall 参与判读，NDCG 辅助展示）
    for st in range(args.n_strata):
        a_hit, a_ndcg, b_hit, b_ndcg = [], [], [], []
        for it, hit, ndcg in zip(items, hits, ndcgs):
            st0 = stratum_of.get(it)
            if st0 is None or st0 != st:
                continue
            if desc_of.get(it, False):
                a_hit.append(float(hit))
                a_ndcg.append(float(ndcg))
            else:
                b_hit.append(float(hit))
                b_ndcg.append(float(ndcg))
        rows[st]["ci_recall"] = bootstrap_delta_ci(
            np.asarray(a_hit), np.asarray(b_hit), args.n_boot, args.seed
        )
        rows[st]["ci_ndcg"] = bootstrap_delta_ci(
            np.asarray(a_ndcg), np.asarray(b_ndcg), args.n_boot, args.seed
        )

    # 整体加权 Δ
    tot_ok = sum(c["n"] for (st, d), c in cells.items() if d)
    tot_miss = sum(c["n"] for (st, d), c in cells.items() if not d)
    w_dr = w_dn = None
    if tot_ok and tot_miss:
        r_ok = sum(c["n"] * c["recall"] for (st, d), c in cells.items() if d) / tot_ok
        r_miss = sum(c["n"] * c["recall"] for (st, d), c in cells.items() if not d) / tot_miss
        n_ok2 = sum(c["n"] * c["ndcg"] for (st, d), c in cells.items() if d) / tot_ok
        n_miss2 = sum(c["n"] * c["ndcg"] for (st, d), c in cells.items() if not d) / tot_miss
        w_dr, w_dn = round(r_ok - r_miss, 4), round(n_ok2 - n_miss2, 4)
    n_scored = sum(c["n"] for c in cells.values())
    n_skipped = len(items) - n_scored
    overall = {"delta_recall": w_dr, "delta_ndcg": w_dn}

    decision = verdict(rows)
    print(_format_report(rows, overall, decision, len(items), n_scored, n_skipped))

    out = {
        "k": args.k, "n_strata": args.n_strata,
        "n_examples": len(items), "n_scored": n_scored, "n_skipped": n_skipped,
        "layers": rows, "overall": overall, "verdict": {"label": decision[0], "reason": decision[1]},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
