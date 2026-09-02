"""image 可下载性核查（数据风险闸门，规格书 §3.2）。

Amazon-2023 标准版只有原始图片 URL、没有预计算特征。这些 URL 是否存活、
下载内容是否真是图片，决定"image 模态能不能建"。本脚本从 meta 抽样抽取
图片 URL，HTTP(Range) 探测存活率 + 魔数校验，输出按 host/状态/失败原因分组。

用法（服务器上，数据已就位）：
    python scripts/check_image_download.py --meta <meta.jsonl 绝对路径> \
        --sample 200 --max-urls 1500 --out outputs/data_check/image_download.json

注意：本脚本区分"HTTP 明确失败"（URL 死链，服务器响应了）与"网络层失败"
（超时/连接拒绝，可能是 GFW 屏蔽而非 URL 死链）；结果解读见 report 输出。
"""

import argparse
import concurrent.futures
import json
import os
import random
import socket
import ssl
import sys
import urllib.error
import urllib.request
from collections import Counter
from urllib.parse import urlparse

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 直接运行（python scripts/check_image_download.py）时脚本目录在 sys.path[0]，
# 需把项目根补进去才能 import scripts.*（pytest 由 pyproject 的 pythonpath 保证）。
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from scripts.check_missingness import _image_urls, iter_jsonl

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


# ---------------------------------------------------------------- URL 收集


def collect_urls(meta_rows, sample: int, max_urls: int, seed: int = 20260830):
    """从 meta 行蓄水池均匀抽 sample 个物品，收集最多 max_urls 个图片 URL。

    - 蓄水池抽样：不假设文件顺序/总长，避免"文件前缀物品"的代表性偏差；
    - **max_urls 截断是"收全量 → 洗牌 → 随机截断"而非"文件序先到先得"**
      （2026-08-30 审查修正）：旧版按物品顺序累积、达到上限即 break，前部多图商品
      会占满配额导致后部物品的 URL 完全不参与探测，且 item_stats 在 break 时少计
      物品数；新版每个抽样物品的 URL 等概率进入探测集，存活率才反映 URL 总体；
    - seed 固定保证重跑一致。
    返回 (urls, item_stats)。item_stats 含抽样物品数、有图占比、URL/物品均值、
    n_items_in_probe（URL 实际来源物品数）。
    """
    rng = random.Random(seed)
    sampled = []
    for i, row in enumerate(meta_rows):
        if i < sample:
            sampled.append(row)
        else:
            j = rng.randint(0, i)
            if j < sample:
                sampled[j] = row

    n_items = len(sampled)
    n_with_img = 0
    per_item = []
    pairs = []  # (物品 id, url)——id 用于统计探测集的实际来源物品数
    for idx, row in enumerate(sampled):
        u = _image_urls(row.get("images"))
        per_item.append(len(u))
        if u:
            n_with_img += 1
        pid = row.get("parent_asin")
        key = pid if pid is not None else f"__row{idx}"
        pairs.extend((key, url) for url in u)

    rng.shuffle(pairs)
    pairs = pairs[:max_urls]
    urls = [url for _, url in pairs]
    n_items_in_probe = len({pid for pid, _ in pairs})

    item_stats = {
        "n_items": n_items,
        "n_with_image": n_with_img,
        "pct_with_image": round(100.0 * n_with_img / n_items, 2) if n_items else 0.0,
        "urls_per_item_mean": round(sum(per_item) / len(per_item), 2) if per_item else 0.0,
        "n_items_in_probe": n_items_in_probe,
    }
    return urls, item_stats


# ---------------------------------------------------------------- HTTP 探测


def http_status(url: str, timeout: float, range_bytes: int):
    """GET Range 探测单个 URL。

    返回 (status_code, first_bytes, fail_kind)：
      - 成功/HTTP 明确失败（服务器响应了）：fail_kind="http"，code 为状态码；
      - 网络层失败：code=None，fail_kind ∈ {"timeout","conn","dns","ssl","other"}；
      - 非 http(s) scheme（畸形/标签字符串混入）：code=None，fail_kind="invalid"。
    3xx 由 urllib 自动跟随。
    """
    # 纵深防御：畸形 URL 在 Request 构造期抛 ValueError（try 块外），先按 scheme 过滤，
    # 单条坏 URL 只记一条 invalid，不让整个并发探测崩溃。
    if urlparse(url).scheme not in ("http", "https"):
        return None, b"", "invalid"
    req = urllib.request.Request(
        url, headers={"User-Agent": _UA, "Range": f"bytes=0-{range_bytes - 1}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read(range_bytes), None
    except urllib.error.HTTPError as e:
        return e.code, b"", "http"
    except urllib.error.URLError as e:
        r = e.reason
        if isinstance(r, socket.gaierror):
            return None, b"", "dns"
        if isinstance(r, ssl.SSLError):
            return None, b"", "ssl"
        if isinstance(r, (ConnectionRefusedError, ConnectionResetError, ConnectionAbortedError)):
            return None, b"", "conn"
        if isinstance(r, (socket.timeout, TimeoutError)):
            return None, b"", "timeout"
        return None, b"", "other"
    except TimeoutError:
        return None, b"", "timeout"
    except Exception:
        return None, b"", "other"


# ---------------------------------------------------------------- 汇总


def _magic_name(b: bytes):
    """按魔数识别常见图片格式；识别不出返回 None。"""
    if b[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if b[:4] == b"\x89PNG":
        return "png"
    if b[:3] == b"GIF":
        return "gif"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "webp"
    if b"ftypavif" in b[:16] or b"ftypmif1" in b[:16]:
        return "avif"
    return None


def probe_urls(urls, timeout: float, range_bytes: int, workers: int):
    """并发探测，返回 [(url, code, first_bytes, fail_kind)]，保持输入顺序。"""
    def one(u):
        code, first, fail = http_status(u, timeout, range_bytes)
        return (u, code, first, fail)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, urls))


def summarize(urls, results):
    """汇总：总体存活率 + 图片魔数率 + 按 host/状态/失败原因分组。"""
    total = len(urls)
    ok_n = 0
    host_ok = Counter()
    host_n = Counter()
    status_cnt = Counter()
    fail_cnt = Counter()
    magic_tried = 0
    magic_ok = 0
    for (url, code, first, fail) in results:
        host = urlparse(url).netloc
        host_n[host] += 1
        if code is not None and 200 <= code < 400:
            ok_n += 1
            host_ok[host] += 1
        if code is not None:
            status_cnt[str(code)] += 1
        else:
            fail_cnt[fail or "unknown"] += 1
        if first:
            magic_tried += 1
            if _magic_name(first):
                magic_ok += 1
    return {
        "n_urls": total,
        "n_ok": ok_n,
        "pct_ok": round(100.0 * ok_n / total, 2) if total else 0.0,
        "pct_valid_image": round(100.0 * magic_ok / magic_tried, 2) if magic_tried else None,
        "n_magic_tried": magic_tried,
        "by_status": dict(status_cnt),
        "by_failure": dict(fail_cnt),
        "by_host": {
            h: {"n": host_n[h], "pct_ok": round(100.0 * host_ok[h] / host_n[h], 2)}
            for h in host_n
        },
    }


def verdict(pct_ok: float) -> str:
    """存活率分级决策。"""
    if pct_ok >= 90:
        return "image 模态可建（存活率足够），风险解除；仍需 CLIP 特征实测兜底。"
    if pct_ok >= 50:
        return "存活率中等：先看死链是否集中在少数 host（可换源/换图域），普遍则走降级路线。"
    return "存活率偏低：优先复用已处理特征路线（SMD/I3-MRec/MILK），image 自建降级/暂缓。"


# ---------------------------------------------------------------- 报告


def report(out: dict) -> None:
    is_ = out["item_stats"]
    d = out["download"]
    print("=" * 56)
    print("image 可下载性核查")
    print("=" * 56)
    print(f"抽样物品 {is_['n_items']}  有图 {is_['pct_with_image']}%  URL/物品均值 {is_['urls_per_item_mean']}  "
          f"URL 实际来源物品 {is_['n_items_in_probe']}")
    mag = f"  下载内容为真图片 {d['pct_valid_image']}%" if d.get("pct_valid_image") is not None else ""
    print(f"URL 总数 {d['n_urls']}  存活 {d['n_ok']} ({d['pct_ok']}%){mag}")
    if d["by_status"]:
        print("状态码分布: " + "  ".join(f"{k}={v}" for k, v in sorted(d["by_status"].items())))
    if d["by_failure"]:
        print("网络层失败: " + "  ".join(f"{k}={v}" for k, v in sorted(d["by_failure"].items()))
              + "  (超时/连接拒绝/SSL 失败可能是网络屏蔽而非 URL 死链；conn 类含拒绝/重置/中止)")
    print("按 host：")
    for h, v in sorted(d["by_host"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {h:45s} n={v['n']:>4}  存活 {v['pct_ok']}%")
    print("-" * 56)
    print("结论：" + verdict(d["pct_ok"]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta", required=True, help="meta jsonl 绝对路径")
    p.add_argument("--sample", type=int, default=200, help="抽样物品数")
    p.add_argument("--max-urls", type=int, default=1500, help="最多探测的 URL 数（覆盖更多商品）")
    p.add_argument("--timeout", type=float, default=6.0, help="单 URL 超时秒数")
    p.add_argument("--range-bytes", type=int, default=2048, help="每 URL 读取字节数（够魔数校验）")
    p.add_argument("--workers", type=int, default=8, help="并发数（过高易触发 CDN 限流 429）")
    p.add_argument("--seed", type=int, default=20260830, help="物品蓄水池抽样种子")
    p.add_argument("--out", default="outputs/data_check/image_download.json")
    args = p.parse_args()

    urls, item_stats = collect_urls(iter_jsonl(args.meta), args.sample, args.max_urls, seed=args.seed)
    if not urls:
        print("抽样样本里没有图片 URL，退出。")
        return
    results = probe_urls(urls, args.timeout, args.range_bytes, args.workers)
    stats = summarize(urls, results)
    out = {
        "item_stats": item_stats,
        "download": stats,
        "meta": args.meta,
        "sample": args.sample,
        "max_urls": args.max_urls,
        "seed": args.seed,
        "timeout": args.timeout,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    report(out)


if __name__ == "__main__":
    main()
