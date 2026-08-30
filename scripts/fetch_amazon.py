"""获取并转换 Amazon Reviews 2023 (Health_and_Household) 为 `user item ts` 格式。

用途：S1 数据信号检验（三审 3.2）的前置——把官方基准 CSV 转成管线可读的整数 ID + 秒级时间戳。

转换要点（与管线源码一一对应）：
- load_interactions 要求 user/item 为 int64 -> 字符串 ID 映射为整数（category codes）
- item 用 parent_asin：合并同款变体，复购语义更准
- signal.py 按秒算天数（/86400）-> 毫秒时间戳除以 1000 归一为秒
- 不按 (user,item) 去重：复购是核心健康信号

下载：国内直连 huggingface.co 超时，默认走镜像 hf-mirror.com；文件名以实际目录为准，
先试常见扩展名 (.csv.gz/.csv)，仍 404 则用镜像 API 列出目录自动发现真实文件名。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "amazon"
CATEGORY = "Health_and_Household"
DEFAULT_MIRROR = "hf-mirror.com"
REPO = "McAuley-Lab/Amazon-Reviews-2023"
RESOLVE = "https://{host}/datasets/{repo}/resolve/main/{path}"
BENCH_DIR = "benchmark/{core}/rating_only/{category}"
CANDIDATE_EXTS = (".csv.gz", ".csv")


MIN_KEEP_BYTES = 100_000_000  # 低于此视为残缺文件，重新下载


def _curl(url: str, dest: Path) -> bool:
    """下载到 dest（进度实时可见）；成功返回 True（失败时清理残留文件）。"""
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


def _already_have(dest: Path) -> bool:
    return dest.exists() and dest.stat().st_size > MIN_KEEP_BYTES


def _find_remote_health_file(host: str, core: str) -> str | None:
    """镜像 API 列出 rating_only 目录，返回 Health 类目文件路径（兜底自动发现）。"""
    api = f"https://{host}/api/datasets/{REPO}/tree/main/benchmark/{core}/rating_only"
    try:
        with urllib.request.urlopen(api, timeout=60) as resp:
            entries = json.load(resp)
    except OSError as exc:
        print(f"WARN: mirror API listing failed ({exc}); falling back to direct URL")
        return None
    files = [e["path"] for e in entries if e.get("type") == "file"]
    match = [p for p in files if "Health" in p and p.endswith((".csv", ".csv.gz"))]
    if not match:
        print("WARN: no Health file found in mirror rating_only dir:")
        for p in files:
            print(f"  {p}")
        return None
    return match[0]


def convert(csv: Path, out: Path) -> None:
    print(f"reading {csv.name} ...")
    df = pd.read_csv(csv, usecols=["user_id", "parent_asin", "timestamp"])
    before = len(df)
    df = df.dropna(subset=["user_id", "parent_asin", "timestamp"])

    df["user"] = df["user_id"].astype("category").cat.codes
    parent_cat = df["parent_asin"].astype("category")
    df["item"] = parent_cat.cat.codes
    # 毫秒(13位, >1e12) -> 秒；已是秒则原样
    df["ts"] = df["timestamp"] / (1000 if df["timestamp"].max() > 1e12 else 1)

    # 持久化 parent_asin ↔ item code 映射：category codes 无法从 Health.txt 反推回原始
    # asin，S2 真实模式需用 meta（parent_asin 为字符串）对齐物品可用性，缺此映射会
    # 全部走 desc 缺失回退、静默摧毁自然 MNAR 信号（见 s2_train._build_real_data）。
    map_path = out.with_name(out.stem + ".items.jsonl")
    with open(map_path, "w", encoding="utf-8") as fh:
        for code, asin in enumerate(parent_cat.cat.categories):
            fh.write(json.dumps({"parent_asin": str(asin), "code": int(code)}, ensure_ascii=False) + "\n")

    df = df.sort_values(["user", "ts"], kind="mergesort")
    df[["user", "item", "ts"]].to_csv(out, sep=" ", header=False, index=False)

    print(f"rows: {before} -> {len(df)} (kept repurchases)")
    print(f"users: {df['user'].nunique()}, items: {df['item'].nunique()}")
    print(f"item mapping: {map_path} ({len(parent_cat.cat.categories)} items)")
    print(f"ts range: {df['ts'].min():.0f} .. {df['ts'].max():.0f} (unix seconds)")
    print(f"written: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch + convert Amazon Health_and_Household to user item ts")
    parser.add_argument("--core", default="0core", help="benchmark core level (0core/2core/4core; smaller=faster)")
    parser.add_argument("--mirror", default=DEFAULT_MIRROR, help="HF mirror host (hf-mirror.com; huggingface.co for direct)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--keep-raw", action="store_true", help="keep the downloaded csv.gz")
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    local: Path | None = None
    for ext in CANDIDATE_EXTS:
        dest = data_dir / (CATEGORY + ext)
        if _already_have(dest):
            print(f"skip download (already have): {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
            local = dest
            break
        url = RESOLVE.format(host=args.mirror, repo=REPO, path=BENCH_DIR.format(core=args.core, category=CATEGORY) + ext)
        print(f"trying {url}")
        if _curl(url, dest):
            local = dest
            break

    if local is None:
        remote = _find_remote_health_file(args.mirror, args.core)
        if remote is None:
            raise RuntimeError("cannot locate Health_and_Household rating_only file on the mirror")
        url = RESOLVE.format(host=args.mirror, repo=REPO, path=remote)
        dest = data_dir / Path(remote).name
        print(f"auto-discovered: {remote}")
        if not _curl(url, dest):
            raise RuntimeError(f"download failed: {url}")
        local = dest

    out = data_dir / "Health.txt"
    convert(local, out)
    if not args.keep_raw:
        local.unlink(missing_ok=True)

    print("\nnext step:")
    print(f'  python -m mrseqrec.cli signal-check --data "{out.as_posix()}"')


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
