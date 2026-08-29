# GPU 服务器训练指南（MR-SeqRec S1 基线）

**分工（用户既定工作流）**：本机（Intel Arc 无 CUDA）只做开发 + pytest + 冒烟；**所有模型训练在学校 GPU 服务器执行**。本文是你在服务器上照做的完整手册，配套一键脚本 `scripts/server_run.sh`。

目标：在 GPU 服务器上用真实 Amazon Health 数据训练 SASRec 基线，产出 valid/test 的 Recall@10/20、NDCG、loss 曲线。

---

## 0. 前置：需要带什么上服务器

| 内容 | 大小 | 说明 |
|---|---|---|
| 代码（本仓库，含 `src/`、`configs/`、`scripts/`、`tests/`、`pyproject.toml`） | ~1 MB | 排除 `.venv/`、`data/`、`outputs/` |
| 数据 `data/amazon/Health.txt` | **734 MB** | 已转换好的 `user item ts` 三列，无需再转 |
| （可选）NVIDIA 驱动 CUDA 版本 | — | 决定 torch 装哪个 cu 版本 |

---

## 1. 代码上服务器（二选一）

### 方案 A：git（推荐，后续同步方便）
先在本机提交所有改动并推送到远端（GitHub/Gitee），服务器上 clone：

```bash
# 本机
git add -A && git commit -m "S1: 采样CE训练器 + GPU服务器训练包"
git push origin master
```
```bash
# 服务器
git clone <你的仓库地址> mrseqrec
```

### 方案 B：tar + scp（不走 git）
本机打包（排除重目录）：
```bash
tar --exclude='.venv' --exclude='data' --exclude='outputs' --exclude='__pycache__' \
    -czf mrseqrec.tar.gz -C /d/MR_SeqRec .
scp mrseqrec.tar.gz <user>@<server>:~/
```
服务器解包：
```bash
mkdir -p mrseqrec && tar -xzf mrseqrec.tar.gz -C mrseqrec && rm mrseqrec.tar.gz
```

---

## 2. 数据上服务器（二选一）

### 方案 A：scp 上传（推荐，734MB 一次传完）
```bash
scp data/amazon/Health.txt <user>@<server>:~/mrseqrec/data/amazon/
```

### 方案 B：服务器联网下载 + 转换（约 1.1GB，需能访问 hf-mirror.com）
```bash
cd ~/mrseqrec
python3 scripts/fetch_amazon.py        # 默认 0core 全量；k-core 在训练预处理时按配置做
```

> 注意：训练配置 `configs/s1_amazon.yaml` 里 `min_interactions: 10`（kcore10），
> 这个过滤发生在训练前预处理阶段，不需要预先手动做 k-core。

---

## 3. 检查服务器环境

```bash
nvidia-smi                  # 确认 GPU 与驱动 CUDA 版本
python3 --version           # 需要 ≥ 3.12；不足则：conda create -n mrseq python=3.12
```

---

## 4. 安装依赖

用一键脚本（自动建 venv、装 CUDA torch、装项目、跑 pytest）：
```bash
cd ~/mrseqrec
bash scripts/server_run.sh
```

如果手动装，等价于：
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# 按 nvidia-smi 顶部 "CUDA Version" 选 torch 版本：
#   cu118 驱动≥450 | cu121 驱动≥525 | cu124 驱动≥550
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
python -m pytest -q          # 冒烟：合成数据，几十秒
```

---

## 5. 训练

```bash
# 建议在 tmux 会话里跑，断开 SSH 不中断：
#   tmux new -s train
cd ~/mrseqrec
nohup .venv/bin/python -u -m mrseqrec.cli train \
    --config configs/s1_amazon.yaml \
    --save-dir outputs/s1_amazon \
    > outputs/s1_amazon/train.log 2>&1 &
```

**确认训练真的用了 GPU**：日志第一行应为 `device=cuda`（`src/mrseqrec/utils/device.py` 按 CUDA→XPU→CPU 自动解析）。

看进度 / 结果：
```bash
tail -f outputs/s1_amazon/train.log          # 进度条 + epoch loss
cat outputs/s1_amazon/metrics.json           # valid/test 指标（等训练结束后）
```

训练规模参考（GPU 上应远快于本地 CPU）：136K 用户 / 63K 商品 / 30 epochs / 采样 CE 损失。

---

## 6. 结果回收

`outputs/s1_amazon/` 下有：
- `model.pt` — 训练好的模型权重
- `metrics.json` — valid/test 的 Recall@10/20、NDCG
- `train.log` — 完整日志（loss 曲线数据源）

拉回本机存档：
```bash
scp -r <user>@<server>:~/mrseqrec/outputs/s1_amazon/ ./outputs/
```

---

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| `torch.cuda.is_available()` 为 False | 驱动 CUDA 与 torch cu 版本不匹配，换 cu118 重装 torch |
| 日志 `device=cpu` | torch 装成了 CPU 版；`pip install torch --index-url .../cu121` 覆盖 |
| `data/amazon/Health.txt` 缺失报错 | 按第 2 步补数据 |
| 训练被打断、想续跑 | S1 暂无断点续训；重跑即可（30 epochs 在 GPU 上很快） |
| 服务器 Python < 3.12 | `conda create -n mrseq python=3.12 && conda activate mrseq` |

---

## 8. 说明

- **采样 CE 训练损失**（`src/mrseqrec/trainers/trainer.py`）：词表 6 万时全词表 softmax 在 CPU 上 ~11h，采样 CE（1 正 + 100 负）在 GPU 上分钟级完成，与评估协议（1 正 + 100 负）一致。
- 训练/评估脚本同时可用 `configs/s1_default.yaml`（同一数据、epochs=50、kcore5），对比调参用。
