# MR-SeqRec

缺失不变多模态序列推荐（Missingness-Invariant Multimodal Sequential Recommendation）。
双目标：中国大学生计算机设计大赛（2027届）+ 国际权威期刊。设计细节见 `PROJECT_SPEC.md`。

## 环境

```bash
python -m venv .venv            # Python ≥ 3.12
.venv/Scripts/activate          # Windows
pip install -e ".[dev]"
```

当前为 Intel Arc 130T(16GB) 无 CUDA，torch 按 CPU 安装；后续如需 Arc XPU 加速再装 Intel Extension for PyTorch。

## 使用

```bash
# 训练并评估 SASRec 基线（配置见 configs/s1_default.yaml）
python -m mrseqrec.cli train --config configs/s1_default.yaml --save-dir outputs/s1

# 数据信号检验：购买间隔分布与周期性（健康叙事可行性闸门）
python -m mrseqrec.cli signal-check --data data/amazon/Health.txt
```

## 结构

```
src/mrseqrec/
  data/       读取、k-core、leave-one-out 划分、负采样、数据集、信号检验
  models/     序列推荐模型（SASRec；MM-SASRec 待特征数据到位）
  trainers/   训练循环
  eval/       Recall/NDCG 指标、ranking 评估器、retention 指标
  utils/      配置（YAML+pydantic）、设备、随机种子、日志
  pipeline.py 训练-评估流水线
  cli.py      命令行入口
tests/        单元测试（pytest）
```

## 测试

```bash
python -m pytest
```
