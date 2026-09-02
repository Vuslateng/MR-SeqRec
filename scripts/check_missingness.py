"""模态覆盖率数据核查：量化 Amazon-2023 自然缺失分布，支撑 L_inv 环境构造与 MNAR 参数化。

回答四个决策问题（S2 开工前必须用数据回答）：
1. 物品级模态覆盖率：title/description/image 在 k-core item 集上的占比
2. 交互级文本覆盖率：有非空 review 文本的交互占比（抽样估计）
3. 交互级缺失模式分布：{text,image,desc} 的 2^3=8 种子集各占比
   -> 决定「观测缺失模式分环境」是否可行、是否必须依赖合成增广
   （desc 是唯一有实质自然缺失的通道，约 50%——必须纳入模式分布）
4. 流行度 × 缺失相关：image/desc/text 覆盖率随流行度的变化趋势
   -> MNAR 倾向模型（问题3）的种子证据
附：序列完整性分布（每用户历史中 desc 可用占比的分位数）+ 跨环境混合度（序列内是否混合
desc 可用/缺失两类环境）——评估时缺失强度的现实基准与 V-REx 环境混杂判断。

k-core 语义与 preprocess.k_core_filter 一致：迭代式、user+item 双侧、交互数 ≥ k 剪枝到收敛。

用法（服务器上，数据缺失时自动从 hf-mirror 下载；reviews 默认只取前 600MB 前缀够抽样）：
  python scripts/check_missingness.py --min-interactions 10 --sample 1000000
  # reviews 全量下载（S2 抽特征时才需要）：--reviews-prefix-mb 0

输出：控制台摘要 + {--out}/summary.json + {--out}/report.txt
      可选 --meta-items-out 额外落盘 k-core 物品集 meta-items.jsonl
      （{"parent_asin","count","has_desc"}，喂给服务器批次 ③④⑦）
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，强制 UTF-8 避免中文乱码

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "amazon"
CATEGORY = "Health_and_Household"
DEFAULT_MIRROR = "hf-mirror.com"
REPO = "McAuley-Lab/Amazon-Reviews-2023"
RESOLVE = "https://{host}/datasets/{repo}/resolve/main/{path}"

# 镜像内真实文件名均为未压缩（.csv / .jsonl）；曾误以为 .gz 导致 404
META_PATH = f"raw/meta_categories/meta_{CATEGORY}.jsonl"
REVIEWS_PATH = f"raw/review_categories/{CATEGORY}.jsonl"
RAW_BENCH = f"benchmark/0core/rating_only/{CATEGORY}.csv"  # 与 fetch_amazon 一致的原始交互集

MIN_KEEP_BYTES = 1_000_000  # 低于此视为残缺/中断文件，重新下载


def _curl(url: str, dest: Path) -> bool:
    """下载到 dest（失败时清理残留文件）。"""
    proc = subprocess.run(
        ["curl", "-L", "--fail", "--connect-timeout", "30", "-o", str(dest), url],
        text=True,
    )
    if proc.returncode == 0:
        print(f"  -> OK ({dest.stat().st_size / 1e6:.0f} MB)")
        return True
    dest.unlink(missing_ok=True)
    print(f"  -> failed (curl exit {proc.returncode})")
    return False


def _resolve_url(host: str, path: str) -> str:
    return RESOLVE.format(host=host, repo=REPO, path=path)


def ensure_download(dest: Path, remote_path: str, mirror: str, label: str) -> Path:
    """dest 缺失或残缺时从镜像下载，返回有效路径。"""
    if dest.exists() and dest.stat().st_size > MIN_KEEP_BYTES:
        print(f"skip download (already have): {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    print(f"downloading {label} from hf-mirror ...")
    url = _resolve_url(mirror, remote_path)
    print(f"  {url}")
    if not _curl(url, dest):
        raise RuntimeError(f"download failed: {url}")
    return dest


def resolve_raw(data_dir: Path, mirror: str) -> Path:
    """优先复用 fetch_amazon 已下载的 raw（.csv / .csv.gz），否则下载 .csv。"""
    for name in (CATEGORY + ".csv", CATEGORY + ".csv.gz"):
        p = data_dir / name
        if p.exists() and p.stat().st_size > MIN_KEEP_BYTES:
            print(f"reuse raw: {p.name} ({p.stat().st_size / 1e6:.0f} MB)")
            return p
    return ensure_download(data_dir / (CATEGORY + ".csv"), RAW_BENCH, mirror, "rating_only 交互集")


def ensure_prefix_download(dest: Path, remote_path: str, mirror: str, mb: int) -> Path:
    """只下载文件开头 mb MB（HTTP Range），够抽样即可，避免全量大文件下载。

    用于 reviews（全量 ~11.4GB，抽样只需开头 ~600MB）。产出文件是前缀切片，
    不是完整数据集——S2 真正抽特征时再全量下载。
    """
    want = mb * 1_000_000
    if dest.exists() and dest.stat().st_size >= int(want * 0.9):
        print(f"skip (already have prefix): {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    url = _resolve_url(mirror, remote_path)
    print(f"downloading {mb} MB prefix of reviews (Range) ...")
    print(f"  {url}  字节 0-{want - 1}")
    proc = subprocess.run(
        ["curl", "-sL", "--fail", "-r", f"0-{want - 1}", "-o", str(dest), url],
        text=True,
    )
    if proc.returncode != 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"prefix download failed: {url}")
    print(f"  -> OK ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


# ---------------------------------------------------------------- 模态解析（纯函数，可单测）

def _image_urls(field) -> list[str]:
    """从 metadata 的 images 字段提取 URL 字符串列表（兼容 list[dict] / dict[list] / list[str]）。"""
    if isinstance(field, dict):
        nested = []
        for v in field.values():
            nested.extend(v if isinstance(v, list) else [v])
        field = nested
    if not isinstance(field, list):
        return []
    urls = []
    for it in field:
        if isinstance(it, dict):
            for v in it.values():
                if isinstance(v, str) and v.strip():
                    urls.append(v.strip())
        elif isinstance(it, str) and it.strip():
            urls.append(it.strip())
    return urls


def _has_text(field) -> bool:
    """review 文本 / 标题等字符串模态是否非空。"""
    return isinstance(field, str) and len(field.strip()) > 0


def _has_desc(field) -> bool:
    """description 可为字符串或字符串列表，任一非空即算有。"""
    if isinstance(field, str):
        return len(field.strip()) > 0
    if isinstance(field, list):
        return any(isinstance(x, str) and x.strip() for x in field)
    return False


def parse_meta(rec: dict) -> dict:
    """metadata 一行 -> 模态标记。"""
    images = _image_urls(rec.get("images"))
    return {
        "has_title": _has_text(rec.get("title")),
        "has_desc": _has_desc(rec.get("description")),
        "has_image": len(images) > 0,
        "n_images": len(images),
    }


def parse_review(rec: dict) -> dict:
    """review 一行 -> 模态标记（文本模态）。"""
    return {"has_text": _has_text(rec.get("text"))}


MODALITIES = ("text", "image", "desc")


def _all_patterns() -> list[str]:
    """{text,image,desc} 的全部非空子集 + none，共 2^3=8 种缺失模式。"""
    out = []
    for r in range(1, len(MODALITIES) + 1):
        for combo in itertools.combinations(MODALITIES, r):
            out.append("+".join(combo))
    out.append("none")
    return out


def pattern_label(has_text: bool, has_image: bool, has_desc: bool = False) -> str:
    """缺失模式标签：n 个模态拼 n 个，全缺为 none。"""
    parts = []
    if has_text:
        parts.append("text")
    if has_image:
        parts.append("image")
    if has_desc:
        parts.append("desc")
    return "+".join(parts) if parts else "none"


# ---------------------------------------------------------------- k-core 与流式读取

def k_core_sets(df: pd.DataFrame, k: int) -> tuple[set, set]:
    """迭代式 k-core：user/item 双侧、交互数 ≥ k，剪枝到收敛（与 preprocess.k_core_filter 一致）。

    返回 (items, users) 的字符串 ID 集合。
    """
    prev_items = prev_users = None
    while True:
        keep_items = set(df["parent_asin"].value_counts().loc[lambda s: s >= k].index)
        keep_users = set(df["user_id"].value_counts().loc[lambda s: s >= k].index)
        if keep_items == prev_items and keep_users == prev_users:
            break
        prev_items, prev_users = keep_items, keep_users
        df = df[df["parent_asin"].isin(keep_items) & df["user_id"].isin(keep_users)]
    return prev_items, prev_users


def iter_jsonl(path: Path, sample: int = 0):
    """逐行迭代 jsonl/gz；sample>0 时最多取 sample 条。"""
    opener = gzip.open if str(path).endswith(".gz") else open
    count = 0
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)
            count += 1
            if sample and count >= sample:
                return


# ---------------------------------------------------------------- 统计（纯函数，可单测）

def compute_stats(
    item_flags: dict[str, dict],
    review_flags: dict[tuple, bool],
    counts: pd.Series,
    kcore_items: set,
    user_sequences: dict[str, list[str]],
) -> dict:
    """由模态标记与交互统计计算全部核查指标。"""
    n_items = len(kcore_items)
    n_interactions = int(counts.sum())

    def _share(pred) -> float:
        if n_items == 0:
            return 0.0
        hit = sum(1 for pa in kcore_items if pred(item_flags.get(pa, {})))
        return round(100.0 * hit / n_items, 2)

    item_cov = {
        "n_items": n_items,
        "n_interactions": n_interactions,
        "n_with_metadata": len(item_flags),
        "pct_title": _share(lambda f: f.get("has_title", False)),
        "pct_description": _share(lambda f: f.get("has_desc", False)),
        "pct_image": _share(lambda f: f.get("has_image", False)),
        "pct_title_and_image": _share(lambda f: f.get("has_title", False) and f.get("has_image", False)),
        "pct_none": _share(lambda f: not (f.get("has_title", False) or f.get("has_desc", False) or f.get("has_image", False))),
        "median_image_count": _median_image_count(item_flags),
    }

    # 交互级：缺失模式分布（text 来自 review 抽样，image/desc 来自 item 级）
    patterns = {k: 0 for k in _all_patterns()}
    for (uid, pa), has_text in review_flags.items():
        f = item_flags.get(pa, {})
        patterns[pattern_label(has_text, f.get("has_image", False), f.get("has_desc", False))] += 1
    n_pattern = sum(patterns.values())
    pattern_share = {
        k: round(100.0 * v / n_pattern, 2) if n_pattern else 0.0 for k, v in patterns.items()
    }
    texted = sum(v for k, v in patterns.items() if "text" in k)
    has_text_share = round(100.0 * texted / n_pattern, 2) if n_pattern else 0.0
    distinct = sum(1 for v in patterns.values() if v > 0)

    # 流行度 × 缺失（MNAR 种子）：物品按交互数分 10 等频桶 + log 流行度点二列相关
    ranked = counts.sort_values(ascending=False)
    pop_items = [(pa, int(c)) for pa, c in ranked.items() if pa in kcore_items]
    n_pop = len(pop_items)
    texted_items = set(pa for (uid, pa), t in review_flags.items() if t)
    buckets = []
    for b in range(10):
        seg = pop_items[n_pop * b // 10: n_pop * (b + 1) // 10]
        if not seg:
            buckets.append({"range": f"{b*10}-{b*10+9}%", "n_items": 0, "pct_image": 0.0, "pct_desc": 0.0, "pct_text": 0.0})
            continue
        img = sum(1 for pa, _ in seg if item_flags.get(pa, {}).get("has_image", False))
        desc = sum(1 for pa, _ in seg if item_flags.get(pa, {}).get("has_desc", False))
        txt = sum(1 for pa, _ in seg if pa in texted_items)
        buckets.append({
            "range": f"{b*10}-{b*10+9}%",
            "n_items": len(seg),
            "pct_image": round(100.0 * img / len(seg), 2),
            "pct_desc": round(100.0 * desc / len(seg), 2),
            "pct_text": round(100.0 * txt / len(seg), 2),
        })

    imgs = [1.0 if item_flags.get(pa, {}).get("has_image", False) else 0.0 for pa, _ in pop_items]
    descs = [1.0 if item_flags.get(pa, {}).get("has_desc", False) else 0.0 for pa, _ in pop_items]
    texts = [1.0 if pa in texted_items else 0.0 for pa, _ in pop_items]
    log_counts = [math.log1p(c) for _, c in pop_items]

    # 序列完整性 + 跨环境混合度：desc 是唯一真实缺失通道（image 恒 100% 无信息）
    # - 完整性：每用户历史中 desc 可用物品占比（评估时缺失强度的现实基准）
    # - 跨环境混合度：用户历史是否同时含 desc 可用与 desc 缺失物品——判断"缺失"
    #   在序列内是天然混合的（环境多样）还是集中在用户/物品两端（环境混杂）。
    shares = []
    n_mixed = n_any_missing = n_fully = 0
    for seq in user_sequences.values():
        if not seq:
            continue
        n = len(seq)
        avail = sum(1 for pa in seq if item_flags.get(pa, {}).get("has_desc", False))
        shares.append(avail / n)
        if 0 < avail < n:
            n_mixed += 1
        if avail < n:
            n_any_missing += 1
        if avail == n:
            n_fully += 1
    seq_stats = _percentiles(shares)
    n_users = seq_stats["n_users"]
    seq_stats["pct_users_mixed_env"] = round(100.0 * n_mixed / n_users, 1) if n_users else 0.0
    seq_stats["pct_users_any_missing"] = round(100.0 * n_any_missing / n_users, 1) if n_users else 0.0
    seq_stats["pct_users_fully_available"] = round(100.0 * n_fully / n_users, 1) if n_users else 0.0

    return {
        "item_coverage": item_cov,
        "pattern_distribution": {
            "n_sampled": n_pattern,
            "patterns": patterns,
            "shares": pattern_share,
            "has_text_share": has_text_share,
            "distinct": distinct,
        },
        "popularity_missingness": {
            "buckets": buckets,
            "point_biserial_img": round(_point_biserial(imgs, log_counts), 4),
            "point_biserial_desc": round(_point_biserial(descs, log_counts), 4),
            "point_biserial_text": round(_point_biserial(texts, log_counts), 4),
        },
        "sequence_completeness": seq_stats,
    }


def _median_image_count(item_flags: dict[str, dict]) -> int:
    vals = sorted(f.get("n_images", 0) for f in item_flags.values())
    if not vals:
        return 0
    m = len(vals) // 2
    return vals[m] if len(vals) % 2 else (vals[m - 1] + vals[m]) // 2


def _percentiles(shares: list[float]) -> dict:
    """序列完整性的 mean/p50/p90/低于50% 用户占比。"""
    if not shares:
        return {"n_users": 0, "mean": None, "p50": None, "p90": None, "pct_below_half": None}
    s = sorted(shares)
    def pct(p):
        i = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
        return round(100.0 * s[i], 1)
    return {
        "n_users": len(s),
        "mean": round(100.0 * sum(s) / len(s), 1),
        "p50": pct(0.5),
        "p90": pct(0.9),
        "pct_below_half": round(100.0 * sum(1 for x in s if x < 0.5) / len(s), 1),
    }


def _point_biserial(binary: list[float], cont: list[float]) -> float:
    """点二列相关（binary vs 连续），-1..1；空或退化输入返回 0。"""
    n = len(binary)
    if n == 0:
        return 0.0
    p = sum(binary) / n
    if p == 0.0 or p == 1.0:
        return 0.0
    y0 = [c for b, c in zip(binary, cont) if b == 0.0]
    y1 = [c for b, c in zip(binary, cont) if b == 1.0]
    if not y0 or not y1:
        return 0.0
    m1 = sum(y1) / len(y1)
    m0 = sum(y0) / len(y0)
    overall = sum(cont) / n
    s = (sum((c - overall) ** 2 for c in cont) / n) ** 0.5
    if s == 0.0:
        return 0.0
    return (m1 - m0) / s * (p * (1 - p)) ** 0.5


def _mnar_verdict(pm: dict) -> tuple[str, float, str]:
    """MNAR 倾向判定：取实际缺失通道（desc）的点二列相关；image 覆盖≈100% 时方差为 0、无信息量。

    返回 (通道, 相关, 结论)。desc 是环境构造的实际缺失通道；若 desc 为常数则回退到 text。
    """
    d, t, i = pm["point_biserial_desc"], pm["point_biserial_text"], pm["point_biserial_img"]
    if d != 0.0:
        channel, corr = "desc", d
    else:
        channel, corr = max(("desc", d), ("text", t), key=lambda kv: abs(kv[1]))
    if abs(corr) >= 0.15:
        label = "中（倾向与流行度耦合，关口2 需倾向建模 + confound 消融）"
    elif abs(corr) >= 0.05:
        label = "弱-中（存在倾向耦合，关口2 需受控合成强倾向验证）"
    else:
        label = "弱（倾向接近随机）"
    return channel, corr, label


def meta_items_rows(
    item_flags: dict[str, dict],
    counts: pd.Series,
    kcore_items: set,
) -> list[dict]:
    """k-core 物品集的 meta-items 行（⑥，喂给 desc 信息性诊断 / 倾向估计 / S2 real）。

    行 = {"parent_asin", "count", "has_desc"}，key 空间与 check_missingness 的
    k-core 物品集一致（parent_asin 字符串）；count 为 k-core 域内交互计数
    （与 S1/S2 训练人群一致）。确定性排序（parent_asin 升序）便于复现与 diff。
    """
    rows = []
    for pa in sorted(kcore_items):
        # str(pa) 强制字符串键：fetch_amazon 落盘的 items.jsonl / meta 的 parent_asin
        # 均为字符串，防 pandas 把数值型 asin 读成 numpy 标量导致 JSON 序列化失败或
        # desc_informativity/S2 字符串键查表错位
        s = str(pa)
        rows.append({
            "parent_asin": s,
            "count": int(counts.get(pa, 0)),
            "has_desc": bool(item_flags.get(s, {}).get("has_desc", False)),
        })
    return rows


# ---------------------------------------------------------------- 报告

def _format_report(stats: dict) -> str:
    ic = stats["item_coverage"]
    pd_ = stats["pattern_distribution"]
    pm = stats["popularity_missingness"]
    sc = stats["sequence_completeness"]
    lines = [
        "=" * 56,
        "模态覆盖率数据核查报告",
        "=" * 56,
        f"交互集 (k-core): N={ic['n_interactions']:,}  物品 {ic['n_items']:,}",
        f"物品级 (有 metadata {ic['n_with_metadata']:,})",
        f"  title:        {ic['pct_title']}%",
        f"  description:  {ic['pct_description']}%",
        f"  image(≥1):    {ic['pct_image']}%",
        f"  title+image:  {ic['pct_title_and_image']}%",
        f"  全无模态:      {ic['pct_none']}%",
        f"  image 数中位数: {ic['median_image_count']}",
        "",
        f"交互级缺失模式 (抽样 N={pd_['n_sampled']:,}; 不同模式 {pd_['distinct']} 种)",
        "  " + ("  ".join(f"{k}={v}%" for k, v in pd_["shares"].items() if v > 0) or "(空)"),
        f"  有文本交互合计 (问题2): {pd_['has_text_share']}%",
        "",
        f"序列完整性 (用户 N={sc['n_users']:,}; 历史 desc 占比)",
        f"  mean={sc['mean']}%  p50={sc['p50']}%  p90={sc['p90']}%  低于50%用户 {sc['pct_below_half']}%",
        f"跨环境混合度: 历史含 desc 缺失用户 {sc['pct_users_any_missing']}%  全可用 {sc['pct_users_fully_available']}%  "
        f"序列内混合两种环境 {sc['pct_users_mixed_env']}%",
        "",
        "流行度 × 缺失 (log 流行度点二列相关; image={:.3f}, desc={:.3f}, text={:.3f})".format(
            pm["point_biserial_img"], pm["point_biserial_desc"], pm["point_biserial_text"]
        ),
    ]
    for b in pm["buckets"]:
        lines.append(f"  {b['range']:>6}: n={b['n_items']:>5}  image={b['pct_image']}%  desc={b['pct_desc']}%  text={b['pct_text']}%")
    return "\n".join(lines)


def report(stats: dict) -> None:
    print(_format_report(stats))
    pd_ = stats["pattern_distribution"]
    pm = stats["popularity_missingness"]
    top_pattern = max(pd_["shares"], key=pd_["shares"].get)
    top_share = pd_["shares"][top_pattern]
    print("-" * 56)
    print("决策指引：")
    print(f"  主缺失模式 '{top_pattern}' 占 {top_share}% , 共 {pd_['distinct']} 种模式")
    print(f"  -> 观测缺失模式分环境可用性: {'依赖合成增广（自然模式过少/过偏）' if pd_['distinct'] < 4 or top_share > 90 else '观测模式可支撑环境构造'}")
    ch, corr, label = _mnar_verdict(pm)
    d, t, i = pm["point_biserial_desc"], pm["point_biserial_text"], pm["point_biserial_img"]
    print(f"  -> 流行度×缺失 点二列相关 desc={d:+.3f} text={t:+.3f} image={i:+.3f}")
    print(f"     MNAR 倾向证据（按实际缺失通道 {ch}={corr:+.3f} 判）：{label}")


# ---------------------------------------------------------------- 主流程

def main() -> None:
    parser = argparse.ArgumentParser(description="模态覆盖率数据核查（L_inv 环境构造 / MNAR 参数化前置）")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--mirror", default=DEFAULT_MIRROR)
    parser.add_argument("--min-interactions", type=int, default=10, help="k-core 阈值（与 S1 一致）")
    parser.add_argument("--sample", type=int, default=1_000_000, help="review 抽样条数（0=全部）")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "data_check")
    parser.add_argument("--raw", type=Path, default=None, help="rating_only CSV 路径（默认自动下载）")
    parser.add_argument("--meta", type=Path, default=None, help="metadata JSONL 路径（默认自动下载）")
    parser.add_argument("--reviews", type=Path, default=None, help="reviews JSONL 路径（默认自动下载，全量 ~11.4GB）")
    parser.add_argument(
        "--reviews-prefix-mb", type=int, default=600,
        help="只下载 reviews 文件开头 N MB（HTTP Range，默认 600），够抽样；传 0 则全量下载",
    )
    parser.add_argument(
        "--meta-items-out", type=Path, default=None,
        help="写 k-core 物品集 meta-items.jsonl（{\"parent_asin\",\"count\",\"has_desc\"}，"
             "与本次 k-core 域一致），供服务器批次 ③④⑦ 使用",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    raw = args.raw or resolve_raw(data_dir, args.mirror)
    meta = args.meta or ensure_download(
        data_dir / Path(META_PATH).name, META_PATH, args.mirror, "metadata"
    )
    if args.reviews:
        reviews = args.reviews
    elif args.reviews_prefix_mb > 0:
        if args.sample == 0:
            print("WARN: --reviews-prefix-mb 只覆盖文件开头，与 --sample 0（全量）矛盾，改用全量下载")
            reviews = ensure_download(data_dir / Path(REVIEWS_PATH).name, REVIEWS_PATH, args.mirror, "reviews")
        else:
            reviews = ensure_prefix_download(
                data_dir / (Path(REVIEWS_PATH).name + ".prefix"),
                REVIEWS_PATH, args.mirror, args.reviews_prefix_mb,
            )
    else:
        reviews = ensure_download(data_dir / Path(REVIEWS_PATH).name, REVIEWS_PATH, args.mirror, "reviews")

    # ---- 1. 原始交互集 -> k-core 域：流行度 / 用户历史 / item 集
    print(f"\nreading interactions: {raw.name}")
    df = pd.read_csv(raw, usecols=["user_id", "parent_asin", "timestamp"])
    df = df.dropna(subset=["user_id", "parent_asin", "timestamp"])
    kcore_items, kcore_users = k_core_sets(df, args.min_interactions)
    df_core = df[df["parent_asin"].isin(kcore_items) & df["user_id"].isin(kcore_users)]
    counts = df_core["parent_asin"].value_counts()
    user_sequences = (
        df_core.sort_values("timestamp")
        .groupby("user_id")["parent_asin"]
        .agg(list)
        .to_dict()
    )
    print(f"raw interactions: {len(df):,}  -> k-core({args.min_interactions}): {len(df_core):,}  items {len(kcore_items):,}  users {len(kcore_users):,}")

    # ---- 2. metadata：k-core item 集的模态标记
    print(f"reading metadata: {meta.name}")
    item_flags: dict[str, dict] = {}
    for rec in iter_jsonl(meta):
        pa = rec.get("parent_asin")
        if pa in kcore_items:
            item_flags[pa] = parse_meta(rec)
    print(f"items with metadata: {len(item_flags):,} / {len(kcore_items):,}")

    # ---- 3. reviews：交互级文本模态（抽样；仅记 k-core 交互域内的 (user, item)）
    print(f"reading reviews: {reviews.name} (sample={args.sample or 'all'})")
    review_flags: dict[tuple, bool] = {}
    n_reviews = 0
    n_in_universe = 0
    for rec in iter_jsonl(reviews, args.sample):
        n_reviews += 1
        pa = rec.get("parent_asin")
        uid = rec.get("user_id")
        if pa in kcore_items and uid in kcore_users:
            n_in_universe += 1
            key = (uid, pa)
            # 同一 (user, item) 多条 review：任一有文本即算有
            if parse_review(rec)["has_text"]:
                review_flags[key] = True
            else:
                review_flags.setdefault(key, False)
    print(f"reviews scanned: {n_reviews:,}  in k-core universe: {n_in_universe:,}  unique (user,item): {len(review_flags):,}")

    # ---- 4. 统计
    stats = compute_stats(item_flags, review_flags, counts, kcore_items, user_sequences)
    report(stats)

    out_json = args.out / "summary.json"
    out_txt = args.out / "report.txt"
    out_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    out_txt.write_text(_format_report(stats), encoding="utf-8")
    print(f"\nwritten: {out_json}\n         {out_txt}")

    if args.meta_items_out:
        rows = meta_items_rows(item_flags, counts, kcore_items)
        with open(args.meta_items_out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"meta-items: {args.meta_items_out} ({len(rows):,} 行，count=本次 k-core 域计数)")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
