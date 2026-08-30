#!/usr/bin/env bash
# MR-SeqRec GPU 服务器一键训练脚本（Linux + CUDA）
# 用法：在项目根目录执行  bash scripts/server_run.sh
# 前置：代码已传到服务器、Python ≥ 3.10、数据已就位（见下方 [3/6] 两条路径）
# 说明：脚本幂等——已建 venv / 已装依赖 / 已有数据会跳过对应步骤。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_FILE="$PROJECT_DIR/data/amazon/Health.txt"
CONFIG="$PROJECT_DIR/configs/s1_amazon.yaml"
SAVE_DIR="$PROJECT_DIR/outputs/s1_amazon"
VENV="$PROJECT_DIR/.venv"

# 训练用的 torch 版本：按服务器驱动 CUDA 版本选（见 SERVER_GUIDE.md 第 4 步）
#   cu118 驱动≥450  |  cu121 驱动≥525  |  cu124 驱动≥550
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

echo "==> [1/6] NVIDIA GPU 检查 =="
nvidia-smi || { echo "错误：未检测到 NVIDIA GPU，请在 GPU 节点上执行"; exit 1; }

echo "==> [2/6] Python 版本检查（需 ≥ 3.10）=="
python3 - <<'PY' || { echo "错误：需要 Python ≥ 3.10"; exit 1; }
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
python3 --version

echo "==> [3/6] 数据检查 =="
if [ ! -f "$DATA_FILE" ]; then
    echo "未找到 $DATA_FILE"
    echo "两条路径任选其一："
    echo "  A. 从本地 scp 上传（推荐，734MB）："
    echo "     scp data/amazon/Health.txt <user>@<server>:$(dirname "$DATA_FILE")/"
    echo "  B. 在服务器联网下载并转换（约 1.1GB，需能访问 hf-mirror.com）："
    echo "     python3 scripts/fetch_amazon.py      # 默认 0core 全量；k-core 在训练预处理时按配置做"
    exit 1
fi
du -h "$DATA_FILE"

echo "==> [4/6] 虚拟环境与依赖 =="
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    # 先装 CUDA 版 torch，再装项目（torch 已满足时 pip 会跳过）
    "$VENV/bin/pip" install torch --index-url "$TORCH_INDEX"
fi
"$VENV/bin/pip" install -e "$PROJECT_DIR"

echo "==> [5/6] 冒烟测试（合成数据，快）=="
(cd "$PROJECT_DIR" && "$VENV/bin/python" -m pytest -q) || { echo "pytest 失败，先排查环境再训练"; exit 1; }

echo "==> [6/6] 启动训练（30 epochs, CUDA；日志与结果写入 outputs/s1_amazon/）=="
mkdir -p "$SAVE_DIR"
cd "$PROJECT_DIR"
nohup "$VENV/bin/python" -u -m mrseqrec.cli train \
    --config "$CONFIG" \
    --save-dir "$SAVE_DIR" \
    > "$SAVE_DIR/train.log" 2>&1 &
echo "已后台启动，PID=$!（tmux 会话内运行可避免断开中断）"
echo "  看进度: tail -f $SAVE_DIR/train.log"
echo "  确认用 GPU: 日志首行应为 device=cuda"
echo "  完成后:     $SAVE_DIR/metrics.json  +  $SAVE_DIR/model.pt"
echo "  日志末尾即 valid/test 的 Recall@10/20、NDCG"
