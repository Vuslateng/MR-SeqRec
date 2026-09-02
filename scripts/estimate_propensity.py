"""③ 真实缺失倾向估计（关口 2 §2.6 C1）：desc 缺失 ~ log 流行度 的 IRLS 逻辑回归。

输入 meta-items.jsonl（⑥，k-core 物品集 {parent_asin, count, has_desc}）。
y = 1 记 desc 缺失（!has_desc）；X = log1p(count)。仅流行度一个预测因子——
image≈100%、text≈99.99% 自然覆盖无方差、title 亦近 100%，无常量信息可进模型
（§2.6 的 z 协变量在本数据上不成立，如实说明，不硬塞）。

判读（§2.6 定式 P(O^desc=0 | x)=σ(α+β·log count)，β̂<0 = 越冷门越缺）：
  - β̂ 显著为负（95% CI 不含 0）→ MNAR 倾向成立，支撑 mnar_select 冷门优先的
    参数化方向；
  - β̂ CI 含 0 → desc 缺失近似随机（倾向接近 MCAR），MNAR 增广价值弱，如实降级；
  - β̂ 为正 → 与数据核查（点二列 +0.118，desc 存在）矛盾，异常待查，不得静默接受。

用法（服务器，⑥ 之后）：
  python scripts/estimate_propensity.py --meta-items outputs/data_check/meta-items.jsonl \
      --out outputs/data_check/propensity_desc.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from mrseqrec.missing.propensity import fit_logistic


def load_meta_items(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读 meta-items.jsonl → (log1p(count), desc 缺失指示)。行序即文件序。"""
    logcounts: list[float] = []
    missing: list[float] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        assert {"parent_asin", "count", "has_desc"} <= r.keys(), f"meta 行缺字段: {line.strip()}"
        logcounts.append(math.log1p(max(0, int(r["count"]))))
        missing.append(0.0 if bool(r["has_desc"]) else 1.0)
    if not logcounts:
        raise ValueError(f"meta-items 为空: {path}")
    return np.asarray(logcounts, dtype=float), np.asarray(missing, dtype=float)


def _wald_p(beta: float, se: float) -> float | None:
    """Wald 双侧 p 值（正态近似）；se<=0 或无界时返回 None。"""
    if not np.isfinite(beta) or se is None or not np.isfinite(se) or se <= 0:
        return None
    z = abs(beta) / se
    return float(math.erfc(z / math.sqrt(2.0)))


def propensity_report(logcounts: np.ndarray, missing: np.ndarray, l2: float = 1e-3) -> dict:
    """IRLS 拟合 + sandwich SE + Wald 检验 + 判读。返回结构化报告。"""
    n = len(logcounts)
    X = logcounts.reshape(-1, 1)
    beta, se = fit_logistic(X, missing, l2=l2, with_se=True)
    slope, slope_se = float(beta[1]), float(se[1])
    inter, inter_se = float(beta[0]), float(se[0])
    ci = (slope - 1.96 * slope_se, slope + 1.96 * slope_se)
    p = _wald_p(slope, slope_se)

    if p is not None and ci[1] < 0:
        label = "MNAR 倾向成立（冷门更缺 desc）"
    elif p is not None and ci[0] > 0:
        label = "异常：方向与数据核查矛盾（越热门越缺？），待查勿静默接受"
    else:
        label = "倾向弱：desc 缺失近似随机（CI 含 0），MNAR 增广价值有限"
    return {
        "n_items": n,
        "n_desc_missing": int(missing.sum()),
        "pct_desc_missing": round(100.0 * float(missing.mean()), 2),
        "x_mean_log1p_count": round(float(logcounts.mean()), 4),
        "logistic": {
            "intercept": inter,
            "intercept_se": inter_se,
            "beta_log_count": slope,
            "beta_log_count_se": slope_se,
            "ci95_log_count": (round(ci[0], 5), round(ci[1], 5)),
            "wald_p": round(p, 6) if p is not None else None,
        },
        "verdict": label,
    }


def _format_report(r: dict) -> str:
    lg = r["logistic"]
    lines = [
        "=" * 64,
        "真实缺失倾向估计（desc 缺失 ~ log 流行度，§2.6 C1）",
        "=" * 64,
        f"物品 N={r['n_items']:,}  desc 缺失 {r['n_desc_missing']:,}（{r['pct_desc_missing']}%）",
        f"log1p(count) 均值 {r['x_mean_log1p_count']}",
        f"y=1 记 desc 缺失；X=[截距, log1p(count)]，IRLS + sandwich SE",
        f"  intercept      β̂ = {lg['intercept']:+.5f}  SE = {lg['intercept_se']:.5f}",
        f"  log(count) 斜率 β̂ = {lg['beta_log_count']:+.5f}  SE = {lg['beta_log_count_se']:.5f}",
        f"  95% CI = [{lg['ci95_log_count'][0]:+.5f}, {lg['ci95_log_count'][1]:+.5f}]",
        f"  Wald p = {lg['wald_p'] if lg['wald_p'] is not None else 'n/a'}",
        "-" * 64,
        f"判读：{r['verdict']}",
        "注：仅流行度一个预测因子（image/text/title 覆盖≈100% 无常量信息）；",
        "   β̂<0 方向若成立，与 mnar_select 冷门优先、数据核查点二列一致。",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-items", required=True, help="⑥ 产出的 meta-items.jsonl")
    p.add_argument("--l2", type=float, default=1e-3, help="逻辑回归 L2 正则系数")
    p.add_argument("--out", default="outputs/data_check/propensity_desc.json")
    args = p.parse_args(argv)
    logcounts, missing = load_meta_items(args.meta_items)
    report = propensity_report(logcounts, missing, l2=args.l2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(_format_report(report))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
