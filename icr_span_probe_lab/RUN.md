# ICR Span Probe Lab Run Guide

## 1. 适用范围

这份说明面向两类场景：

- 在云端先跑一个最小闭环
- 在云端完整跑两条 span 路线和 5 类方法

当前实现全部位于 `icr_span_probe_lab/` 下，不会覆盖原项目已有实验。

## 2. 运行前提

默认假设你已经把整个仓库同步到云端，并且当前目录是项目根目录：

```bash
cd /path/to/icr_probe_modify
```

当前实验默认使用这两个现成输入文件：

- `outputs/icr_halu_eval_random_qwen2.5.jsonl`
- `data/HaluEval/qa_data.json`

如果这两个文件不在云端，先同步过去再跑。

## 3. 环境准备

推荐使用 Python 3.10 或 3.11。

### 3.1 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3.2 安装依赖

如果你只跑 span 实验，装本实验依赖即可：

```bash
pip install -r icr_span_probe_lab/requirements.txt
python -m spacy download en_core_web_sm
```

如果你后面还要顺手跑原仓库的其他脚本，可以再额外装：

```bash
pip install -r requirements.txt
```

## 3.3 一键脚本

当前已经提供一键脚本：

```bash
bash icr_span_probe_lab/scripts/run_span_lab.sh minimal
```

支持三种模式：

- `minimal`
- `full`
- `figures-only`

也支持环境变量：

- `DEVICE=cpu|cuda`
- `MAX_SAMPLES=1000`
- `SPACY_MODEL=en_core_web_sm`

## 4. 云端资源建议

### 4.1 预处理阶段

下面这些步骤主要吃 CPU 和内存，不需要 GPU：

- `prepare_span_ready_data.py`
- `build_tokenizer_windows.py`
- `build_spacy_spans.py`
- `build_silver_span_labels.py`
- `build_span_dataset.py`

### 4.2 训练阶段

下面这些脚本在 CPU 上也能跑，但 GPU 会更快：

- `train_baseline_mlp.py`
- `train_temporal_conv.py`
- `train_trajectory_encoder.py`

而这两个本质上是传统特征模型，CPU 就够：

- `train_discrepancy.py`
- `train_change_point.py`

如果你只是先验证方向，建议先跑：

- `Tokenizer Window`
- `Baseline MLP`
- `max_samples=500` 或 `1000`

## 5. 最小闭环

这条路线最适合先在云端试跑。

### 5.1 准备 span-ready 数据

```bash
python3 icr_span_probe_lab/scripts/prepare_span_ready_data.py
```

产物：

- `icr_span_probe_lab/data/intermediate/icr_halu_eval_span_ready.jsonl`
- `icr_span_probe_lab/data/intermediate/icr_halu_eval_span_ready.summary.json`

说明：

- 这一步只会重新 tokenize `response`
- 不会重新跑 7B 模型
- 默认会按 ICR 文件中的 `model_name_or_path` 下载 tokenizer

如果云端不能直接访问 HuggingFace，可以显式传本地 tokenizer 路径：

```bash
python3 icr_span_probe_lab/scripts/prepare_span_ready_data.py \
  --model_name_or_path /path/to/local/Qwen2.5-7B-Instruct
```

### 5.2 构建 tokenizer window span

```bash
python3 icr_span_probe_lab/scripts/build_tokenizer_windows.py
```

产物：

- `icr_span_probe_lab/data/span_candidates/tokenizer_windows.jsonl`
- `icr_span_probe_lab/data/span_candidates/tokenizer_windows.summary.json`

默认窗口大小是 `1,2,3,4`。

### 5.3 构建 silver span labels

```bash
python3 icr_span_probe_lab/scripts/build_silver_span_labels.py \
  --span_path icr_span_probe_lab/data/span_candidates/tokenizer_windows.jsonl
```

产物：

- `icr_span_probe_lab/data/span_labels/tokenizer_windows_silver_labels.jsonl`
- `icr_span_probe_lab/data/span_labels/tokenizer_windows_silver_labels.summary.json`

### 5.4 导出可训练数据集

```bash
python3 icr_span_probe_lab/scripts/build_span_dataset.py \
  --labeled_span_path icr_span_probe_lab/data/span_labels/tokenizer_windows_silver_labels.jsonl
```

产物：

- `icr_span_probe_lab/data/datasets/tokenizer_windows_dataset.jsonl`
- `icr_span_probe_lab/data/datasets/tokenizer_windows_dataset.summary.json`

### 5.5 训练第一个模型

```bash
python3 icr_span_probe_lab/scripts/train_baseline_mlp.py \
  --dataset_path icr_span_probe_lab/data/datasets/tokenizer_windows_dataset.jsonl \
  --device cpu
```

默认结果目录：

- `icr_span_probe_lab/results/tokenizer_windows_dataset/baseline_mlp/`

主要文件：

- `BaselineMLP.metrics.json`
- `BaselineMLP.oof_predictions.jsonl`

### 5.6 做 sample-level 聚合评估

```bash
python3 icr_span_probe_lab/scripts/evaluate_sample_aggregation.py \
  --prediction_files icr_span_probe_lab/results/tokenizer_windows_dataset/baseline_mlp/BaselineMLP.oof_predictions.jsonl
```

### 5.7 直接生成可视化结果

在部分云环境里，`matplotlib` 的默认缓存目录可能不可写。为了避免字体缓存告警，建议先设：

```bash
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp/xdg-cache
```

然后直接跑：

```bash
python3 icr_span_probe_lab/scripts/generate_default_figures.py
```

默认会生成：

- 方法总览图
- sample-level 聚合对比图
- span 长度统计图
- 1 个高分 hallucinated case 热图
- 1 个高分 false positive case 热图

说明：

- 这个脚本会自动扫描 `results/` 下所有已经跑完的 `*.metrics.json` 和 `*.oof_predictions.jsonl`
- 你当前只跑了哪些方法，它就只会画哪些方法
- 等你后面在云上把全部方法跑完，它会自动把全量方法都纳入比较图

## 6. 完整流程

如果最小闭环已经通了，可以按下面顺序跑完整实验。

### 6.1 准备两条 span 路线

```bash
python3 icr_span_probe_lab/scripts/prepare_span_ready_data.py

python3 icr_span_probe_lab/scripts/build_tokenizer_windows.py

python3 icr_span_probe_lab/scripts/build_spacy_spans.py
```

### 6.2 为两条路线构建 silver labels

```bash
python3 icr_span_probe_lab/scripts/build_silver_span_labels.py \
  --span_path icr_span_probe_lab/data/span_candidates/tokenizer_windows.jsonl

python3 icr_span_probe_lab/scripts/build_silver_span_labels.py \
  --span_path icr_span_probe_lab/data/span_candidates/spacy_spans.jsonl
```

### 6.3 为两条路线导出数据集

```bash
python3 icr_span_probe_lab/scripts/build_span_dataset.py \
  --labeled_span_path icr_span_probe_lab/data/span_labels/tokenizer_windows_silver_labels.jsonl

python3 icr_span_probe_lab/scripts/build_span_dataset.py \
  --labeled_span_path icr_span_probe_lab/data/span_labels/spacy_spans_silver_labels.jsonl
```

### 6.4 训练 5 类方法

先以 `tokenizer_windows_dataset.jsonl` 为例：

```bash
python3 icr_span_probe_lab/scripts/train_baseline_mlp.py \
  --dataset_path icr_span_probe_lab/data/datasets/tokenizer_windows_dataset.jsonl \
  --device cpu

python3 icr_span_probe_lab/scripts/train_discrepancy.py \
  --dataset_path icr_span_probe_lab/data/datasets/tokenizer_windows_dataset.jsonl

python3 icr_span_probe_lab/scripts/train_temporal_conv.py \
  --dataset_path icr_span_probe_lab/data/datasets/tokenizer_windows_dataset.jsonl \
  --device cpu

python3 icr_span_probe_lab/scripts/train_change_point.py \
  --dataset_path icr_span_probe_lab/data/datasets/tokenizer_windows_dataset.jsonl

python3 icr_span_probe_lab/scripts/train_trajectory_encoder.py \
  --dataset_path icr_span_probe_lab/data/datasets/tokenizer_windows_dataset.jsonl \
  --device cpu
```

然后把同样一组命令再对 `spacy_spans_dataset.jsonl` 跑一遍。

### 6.5 跑完整可视化

如果前面的训练结果都已经在 `results/` 下，可以直接：

```bash
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp/xdg-cache
python3 icr_span_probe_lab/scripts/generate_default_figures.py
```

或者直接：

```bash
bash icr_span_probe_lab/scripts/run_span_lab.sh figures-only
```

如果你想单独画某一类图，也可以分别跑：

```bash
python3 icr_span_probe_lab/scripts/plot_method_comparison.py

python3 icr_span_probe_lab/scripts/plot_span_statistics.py \
  --dataset_path icr_span_probe_lab/data/datasets/tokenizer_windows_dataset.jsonl \
  --prediction_file icr_span_probe_lab/results/tokenizer_windows_dataset/baseline_mlp/BaselineMLP.oof_predictions.jsonl

python3 icr_span_probe_lab/scripts/plot_case_heatmap.py \
  --dataset_path icr_span_probe_lab/data/datasets/tokenizer_windows_dataset.jsonl \
  --prediction_file icr_span_probe_lab/results/tokenizer_windows_dataset/baseline_mlp/BaselineMLP.oof_predictions.jsonl \
  --selection highest_hallucinated
```

### 6.6 一键全量运行

如果你要在云端完整跑一遍，可以直接：

```bash
DEVICE=cuda bash icr_span_probe_lab/scripts/run_span_lab.sh full
```

如果你只是想先验证链路：

```bash
MAX_SAMPLES=1000 DEVICE=cpu bash icr_span_probe_lab/scripts/run_span_lab.sh minimal
```

## 7. 小样本 smoke test

如果你不想一上来就跑全量，建议先只做 500 或 1000 条。

### 7.1 只在前处理阶段限样本

```bash
python3 icr_span_probe_lab/scripts/prepare_span_ready_data.py --max_samples 1000
python3 icr_span_probe_lab/scripts/build_tokenizer_windows.py --max_samples 1000
python3 icr_span_probe_lab/scripts/build_spacy_spans.py --max_samples 1000
```

说明：

- `prepare_span_ready_data.py --max_samples 1000` 会只导出前 1000 条 span-ready 数据
- 后续脚本如果继续读取这个产物，就天然只会在这 1000 条上跑

### 7.2 先只训练一个模型

建议 smoke test 只跑：

- `Tokenizer Window`
- `Baseline MLP`

确认整个链路可跑通之后，再扩到其余方法。

## 8. 输出目录说明

### 8.1 中间数据

- `icr_span_probe_lab/data/intermediate/`
- `icr_span_probe_lab/data/span_candidates/`
- `icr_span_probe_lab/data/span_labels/`
- `icr_span_probe_lab/data/datasets/`

### 8.2 结果

训练脚本默认把结果写到：

- `icr_span_probe_lab/results/<dataset_name>/<method_family>/`

例如：

- `icr_span_probe_lab/results/tokenizer_windows_dataset/baseline_mlp/`
- `icr_span_probe_lab/results/tokenizer_windows_dataset/discrepancy/`
- `icr_span_probe_lab/results/spacy_spans_dataset/trajectory_encoder/`

每个目录下通常会有：

- `*.metrics.json`
- `*.oof_predictions.jsonl`

### 8.3 图表

可视化脚本默认把图写到：

- `icr_span_probe_lab/figures/`

当前支持的图包括：

- `method_summary.png`
- `sample_aggregation_summary.png`
- `*_span_stats.png`
- `*_highest_hallucinated.png`
- `*_highest_false_positive.png`

## 9. 常见问题

### 9.1 `prepare_span_ready_data.py` 报 tokenizer 下载失败

通常是云端没有外网或没有 HuggingFace 权限。

处理方式：

- 给脚本传本地模型目录：`--model_name_or_path /path/to/tokenizer_dir`
- 或先在云端配置好 HuggingFace 下载环境

### 9.2 `build_spacy_spans.py` 报找不到 `en_core_web_sm`

执行：

```bash
python -m spacy download en_core_web_sm
```

### 9.3 训练脚本报缺少 `torch` 或 `scikit-learn`

重新安装：

```bash
pip install -r icr_span_probe_lab/requirements.txt
```

### 9.4 我不想让云端长时间占 GPU

那就先只跑：

- 预处理全流程
- `train_discrepancy.py`
- `train_change_point.py`
- `train_baseline_mlp.py --device cpu`

等确认 silver label 和 sample-level 聚合有价值，再把 CNN / GRU / Transformer 放到 GPU。

## 10. 建议的云端执行顺序

如果你只想先拿到第一批可读结果，建议按这个顺序：

1. 安装依赖
2. 跑 `prepare_span_ready_data.py`
3. 跑 `build_tokenizer_windows.py`
4. 跑 `build_silver_span_labels.py`
5. 跑 `build_span_dataset.py`
6. 跑 `train_baseline_mlp.py`
7. 跑 `generate_default_figures.py`
8. 跑 `evaluate_sample_aggregation.py`
9. 查看 `results/` 和 `figures/`

如果这条线结果正常，再补：

1. `spaCy Span`
2. `Discrepancy`
3. `Change Point`
4. `Temporal Conv`
5. `Trajectory Encoder`
