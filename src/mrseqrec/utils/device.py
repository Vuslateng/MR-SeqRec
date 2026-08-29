"""设备解析：auto 优先 CUDA → Intel XPU → CPU（Intel Arc 环境无 CUDA 时落到 CPU）。"""

from __future__ import annotations

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")
