"""缺失鲁棒性汇总指标：retention（perf@ρ / perf@0）与曲线下面积。

S1 先落地指标机制；多模态缺失注入（MM-SASRec 阶段）后对每个 ρ 评估一次 perf 即得曲线。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RetentionReport:
    rho_values: list[float]
    perf_by_rho: dict[float, float]
    retention_by_rho: dict[float, float]
    auc: float  # retention 曲线下面积（ρ 从 0 到 max）

    def __str__(self) -> str:
        lines = ["  rho    perf      retention"]
        for rho in self.rho_values:
            lines.append(
                f"  {rho:4.2f}  {self.perf_by_rho[rho]:.4f}  {self.retention_by_rho[rho]:.4f}"
            )
        lines.append(f"  AUC(retention) = {self.auc:.4f}")
        return "\n".join(lines)


def compute_retention(perf_by_rho: dict[float, float]) -> RetentionReport:
    """perf_by_rho: {ρ: 某指标值}，retention@ρ = perf@ρ / perf@0。"""
    if 0.0 not in perf_by_rho or perf_by_rho[0.0] <= 0:
        raise ValueError("perf@0 must be present and positive")
    rho_values = sorted(perf_by_rho)
    base = perf_by_rho[0.0]
    retention = {rho: perf_by_rho[rho] / base for rho in rho_values}
    auc = float(np.trapezoid([retention[r] for r in rho_values], x=rho_values))
    return RetentionReport(
        rho_values=rho_values, perf_by_rho=perf_by_rho, retention_by_rho=retention, auc=auc
    )
