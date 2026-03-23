# ICR Span Probe Lab

## 1. 这个文件夹在做什么

这个文件夹用于开展一套**独立于原项目主流程**的 span-level ICR Probe 实验。

目标不是继续在 sample-level 上堆更复杂的模型，而是回答下面这个问题：

> 当我们把整条回答拆成更细粒度的 span 之后，ICR 的跨层轨迹是否能提供比 sample-level 更有区分力、更可解释的幻觉检测信号？

本实验会在项目根目录下单独进行，尽量不动原项目的数据、代码和已有实验结果。

## 2. 当前项目现状

### 2.1 现有 4 个方法的位置

当前 sample-level 的 4 个方法位于：

- `tests/trajectory_analysis/01_discrepancy`
- `tests/trajectory_analysis/02_temporal_conv`
- `tests/trajectory_analysis/03_change_point`
- `tests/trajectory_analysis/04_trajectory_encoder`

这些方法之外，还存在一个原始的 baseline MLP 思路，可视为第 5 个对照方法。

### 2.2 当前方法的核心局限

虽然原始 ICR 数据本身是 token-aware 的，单条样本的形状是：

`[27 usable layers, n_tokens]`

但当前 4 个方法在进入建模前，都会先把 token 维做均值池化，变成：

`[27]`

也就是只保留 sample-level 的层轨迹，不再保留 token/span 粒度信息。

这也是当前现象的核心原因之一：

- 结构化方法都比随机强很多，说明 ICR 轨迹里确实有信号
- 但它们很难明显拉开与原始 baseline MLP 的差距
- 因为大家最终都在吃一个被压缩后的 `[27]` 向量

### 2.3 当前数据的已知事实

基于 `outputs/icr_halu_eval_random_qwen2.5.jsonl` 的现场检查，当前数据有以下特征：

- 样本数：10000
- 标签分布：5010 hallucinated / 4990 correct
- 响应 token 数最小值：1
- 响应 token 数最大值：81
- 响应 token 数均值：9.9794
- 当前输出文件里保存了 `response`、`label`、`icr_scores` 等字段
- 当前输出文件**没有**保存 `response_ids`、`token_texts`、`offsets`

这意味着：

- 现有产物足够做 sample-level 实验
- 但如果要做 span-level，就需要先恢复“第 j 个 ICR token 对应哪段文本”的映射关系

## 3. 关于“重新分词”的说明

这里说的“重新分词”，**不是重新前向、不是重新算 ICR、也不是重跑 7B 模型**。

这里真正要做的是：

1. 使用与原导出脚本一致的 tokenizer
2. 对当前样本的 `response` 文本再次 tokenize
3. 恢复每个 token 的：
   - token id
   - token text
   - 字符级 offset
4. 将这些 token 与现有 `icr_scores[:, j]` 的列进行对齐

之所以可以这样做，是因为当前导出脚本 `scripts/compute_icr_halueval.py` 中：

- response 是单独 tokenize 的
- 使用的是 `add_special_tokens=False`
- 输出文件中还保存了 `num_response_tokens`

因此，只要 tokenizer 版本和设置一致，通常就能把 token 对齐关系恢复出来，而无需重算 ICR。

如果后续发现对齐失败率较高，再考虑重新导出一个增强版派生数据文件，把 token 元数据一并保存进去。

## 4. 本实验的总约束

### 4.1 不直接修改的内容

原则上不要直接修改以下内容：

- `src/`
- `tests/trajectory_analysis/`
- `outputs/icr_halu_eval_random_qwen2.5.jsonl`
- `data/HaluEval/qa_data.json`

### 4.2 所有派生内容都留在本实验目录

后续若实现代码、派生数据、图表和结果，应尽量都放在本目录之下，例如：

```text
icr_span_probe_lab/
  README.md
  TODO.md
  data/
  scripts/
  src/
  results/
  figures/
```

是否真的创建这些子目录，取决于实现阶段再决定；当前先建立最小文档骨架。

## 5. 本实验要做的两种 span 切法

本实验明确要同时做两条路线，并且两条路线都跑同一组方法，方便公平比较。

### 5.1 路线 A：Tokenizer Window

定义：

- 以**模型 tokenizer 切出来的 response token 序列**为基础
- 取连续 token 窗口作为候选 span
- 例如窗口长度为 3 时，候选 span 为：
  - `[t1, t2, t3]`
  - `[t2, t3, t4]`
  - `[t3, t4, t5]`
  - ...

优点：

- 与 ICR 的 token 边界完全一致
- 无需额外的字符到子词对齐步骤
- 最适合先做严格对齐的 baseline

缺点：

- 可解释性偏弱
- span 边界不一定符合自然语言短语结构

### 5.2 路线 B：spaCy Span

定义：

- 对 `response` 做 `spaCy` 分析
- 从中提取自然语言更可读的 span 候选，例如：
  - named entities
  - noun chunks
  - 其他可读的短语级片段

优点：

- span 更接近“人能理解的事实片段”
- 更适合做案例分析和可解释性展示

缺点：

- spaCy span 与 LLM tokenizer subword 边界不同
- 需要额外做字符 offset 到 tokenizer token 区间的映射

## 6. 两种切法都要跑的 5 类方法

要求：`Tokenizer Window` 和 `spaCy Span` 两种切法，都要分别跑以下 5 类方法。

### 6.1 Baseline MLP

- 输入：span 级别的 `[27]` 轨迹表示
- 作用：作为最直接的 span-level baseline

### 6.2 Discrepancy Modeling

- 沿用现有 sample-level 的 `early / middle / late` 分段思路
- 在 span 级 `[27]` 轨迹上提取 7 个特征

### 6.3 Temporal Conv Over Layers

- 继续把 `[27]` 看作层维度的一维信号
- 用 CNN 在层维度上建模局部模式

### 6.4 Change Point Detection

- 在 span 轨迹上做一阶/二阶差分
- 提取变化点数量、粗糙度、最大变化位置和幅度等特征

### 6.5 Layer Trajectory Encoder

- 在 span 级 `[27]` 轨迹上应用 GRU / Transformer / Deep1DCNN 等编码器

## 7. span 表示怎么和现有 5 类方法对齐

为了让 5 类方法之间可比，第一版实现建议采用统一的 span 表示：

对于一个覆盖 token 区间 `[start, end)` 的 span：

1. 从原始 ICR 中取出 `icr_scores[:, start:end]`
2. 在 token 维做池化，得到一个 span 级 `[27]` 表示
3. 该 `[27]` 再送入不同方法

第一版建议默认使用：

- `mean pooling`

可记录的辅助信息：

- `span_text`
- `token_start`
- `token_end`
- `char_start`
- `char_end`
- `span_len_tokens`
- `span_type`（window / entity / noun_chunk 等）

如果后续需要，再加：

- `max pooling`
- `top-k pooling`
- 保留 `[27, span_len]` 的局部 patch 版本

但第一轮目标是先把 span-level 主干流程跑通并保证可比性。

## 8. 标签问题：当前没有 gold span label

这是本实验最重要的现实约束。

当前使用的 HaluEval QA 数据和导出的 ICR 文件只有 sample-level 的标签：

- 一条回答整体是 hallucinated 还是 correct

它**没有天然的 gold hallucination span**。

因此，span-level 实验需要先解决监督来源问题。建议按下面的优先级执行：

### 8.1 第一阶段：构建 silver span labels

先从当前 HaluEval QA 数据出发，在不换数据集的前提下，构建银标 span：

- 对 hallucinated sample：
  - 找出最可能“不被 knowledge 支持”的 span，标为正例候选
- 对 correct sample：
  - 将被 knowledge 支持的 span 作为负例候选

第一版不要一开始就上复杂大模型判别器，优先从轻量、可控的规则和匹配做起。

### 8.2 第二阶段：人工抽样校验

为了避免 silver label 带偏方向，需要建立一个小规模人工核验集，例如：

- 抽样 100 到 200 条样本
- 对 tokenizer-window 和 spaCy-span 两种候选都抽查
- 记录误标类型

### 8.3 第三阶段：必要时再考虑换数据集

只有在以下情况同时出现时，才考虑换数据集：

- HaluEval QA 的回答太短，span 信号仍然很弱
- silver label 质量不稳定
- 聚合回 sample-level 后依然没有明显价值

也就是说，**换数据集是备选方案，不是第一步**。

## 9. 运行入口

本实验的运行说明见：

- `icr_span_probe_lab/RUN.md`

当前也提供了一键脚本：

- `icr_span_probe_lab/scripts/run_span_lab.sh`

如果只是先在云上做最小闭环，建议优先走：

1. `prepare_span_ready_data.py`
2. `build_tokenizer_windows.py`
3. `build_silver_span_labels.py`
4. `build_span_dataset.py`
5. `train_baseline_mlp.py`
6. `generate_default_figures.py` 或 `run_span_lab.sh figures-only`

等这个闭环跑通之后，再补 `spaCy Span` 路线和其余 4 类方法。

## 10. CPU / GPU 预期

### 10.1 可以先在 CPU 上做的事情

以下工作默认可以先在本机 CPU 上做：

- 恢复 token 对齐信息
- 构建 tokenizer-window span
- 构建 spaCy span
- 构建 silver span labels
- 训练 classical ML、小型 MLP、小型 CNN/GRU/Transformer
- 做 span-level 和 sample-level 聚合评估

原因：

- 当前总样本 10000
- 平均 response token 约 10
- span 实例规模仍然在可控范围内

### 10.2 需要 GPU 的情况

以下情况再考虑切到远程 GPU：

- 重算 ICR
- 重导出带 token 元数据的新 ICR 文件
- 使用较重的 NLI / entailment / LLM 作为自动标注器
- 使用显著更大的 span-level 模型

## 11. 建议的比较方式

本实验最终至少要形成两层比较：

### 11.1 span-level 比较

比较对象：

- 两种切法：Tokenizer Window vs spaCy Span
- 五类方法：Baseline MLP + 4 个 trajectory 方法

输出：

- AUROC
- AUPRC
- F1
- 代表性 case study

### 11.2 sample-level 回聚比较

对 span-level 预测结果做 sample-level 聚合，例如：

- max span probability
- top-k mean
- noisy-or

然后再与当前 sample-level baseline 对比，回答：

> span-level 建模除了可解释性之外，是否还能在聚合后带来检测性能收益？

## 12. 第一轮实现的建议顺序

### 12.1 先打通数据主干，不要先堆模型

优先级应该是：

1. 恢复 token 对齐
2. 生成两类 span
3. 构建 silver labels
4. 先跑 Baseline MLP
5. 再逐个补齐 4 类方法

### 12.2 第一轮先保持方法适配简单

第一轮不要做大改造，尽量沿用现有方法框架：

- 所有方法先都吃 span-pooled `[27]`
- 只改变“样本单位”从 sample 变为 span
- 这样可以把贡献集中在“span 切法”和“更细粒度监督”上

## 13. 本目录当前状态

当前已经补齐了独立实验骨架，主要包括：

- `README.md`
- `TODO.md`
- `RUN.md`
- `requirements.txt`
- `scripts/`
- `src/`
- `data/`
- `results/`
- `figures/`

其中已经实现的核心内容包括：

- token 对齐恢复脚本
- tokenizer-window span 构建脚本
- spaCy span 构建脚本
- silver span label 构建脚本
- span pooling 与数据集导出
- 5 类 span-level 方法训练脚本
- sample-level 聚合评估脚本

当前更适合直接按 `RUN.md` 的顺序在云端执行，而不是再从零搭骨架。
