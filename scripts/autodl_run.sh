#!/usr/bin/env bash
# MR-SeqRec AutoDL 云 GPU 一键训练脚本（预装 PyTorch+CUDA 环境，不复装 torch）
# 用法：git clone 后在项目根目录执行  bash scripts/autodl_run.sh
# 与 server_run.sh 的区别：AutoDL 镜像已预装 CUDA torch，这里直接复用、跳过 torch 下载（省 ~2.5GB）；
# Python ≥ 3.10 即可（镜像默认 3.10/3.11，代码仅用 3.10 语法）。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_FILE="$PROJECT_DIR/data/amazon/Health.txt"
CONFIG="$PROJECT_DIR/configs/s1_amazon.yaml"
SAVE_DIR="$PROJECT_DIR/outputs/s1_amazon"

echo "==> [1/5] GPU / CUDA 检查 =="
nvidia-smi || { echo "错误：未检测到 NVIDIA GPU"; exit 1; }
python - <<'PY' || { echo "错误：torch 不可用 CUDA，确认镜像为 PyTorch+CUDA 版"; exit 1; }
import torch, sys
print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()} | gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
sys.exit(0 if torch.cuda.is_available() else 1)
PY

echo "==> [2/5] 安装项目依赖（torch 已满足则跳过，不会重装）=="
pip install -e "$PROJECT_DIR"

echo "==> [3/5] 数据准备 =="
if [ ! -f "$DATA_FILE" ]; then
    echo "服务器联网下载并转换（国内镜像 hf-mirror，带宽快；约 1.1GB 下载 + pandas 转换）"
    python "$PROJECT_DIR/scripts/fetch_amazon.py"
fi
du -h "$DATA_FILE"

echo "==> [4/5] 冒烟测试（合成数据，快）=="
(cd "$PROJECT_DIR" && python -m pytest -q) || { echo "pytest 失败，先排查环境再训练"; exit 1; }

echo "==> [5/5] 启动训练（30 epochs, CUDA；日志与结果写入 outputs/s1_amazon/）=="
mkdir -p "$SAVE_DIR"
cd "$PROJECT_DIR"
nohup python -u -m mrseqrec.cli train \
    --config "$CONFIG" \
    --save-dir "$SAVE_DIR" \
    > "$SAVE_DIR/train.log" 2>&1 &
echo "已后台启动，PID=$!（建议在 tmux 会话内跑，防断连中断）"
echo "  看进度: tail -f $SAVE_DIR/train.log"
echo "  确认用 GPU: 日志首行应为 device=cuda"
echo "  完成后:     $SAVE_DIR/metrics.json  +  $SAVE_DIR/model.pt"
