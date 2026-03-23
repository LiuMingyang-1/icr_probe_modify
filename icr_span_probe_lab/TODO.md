# ICR Span Probe Lab TODO

## 总原则

- 所有新代码、派生数据、日志、图表、结果优先放在 `icr_span_probe_lab/` 下
- 不直接覆盖原项目的 `src/`、`tests/trajectory_analysis/`、`outputs/`、`data/`
- 先做能在 CPU 上验证方向的最小闭环
- 优先保证对齐正确、标签合理、评估清晰，再追求更复杂模型

## Phase 0. 建立实验骨架

- [x] 在项目根目录创建独立实验文件夹 `icr_span_probe_lab/`
- [x] 写入 `README.md`
- [x] 写入 `TODO.md`
- [ ] 后续实现时按需创建：
  - `data/`
  - `scripts/`
  - `src/`
  - `results/`
  - `figures/`

## Phase 1. 复原 token 对齐信息

目标：在不重算 ICR 的前提下，恢复每个 response token 的文本和位置。

- [ ] 新建脚本：`prepare_span_ready_data.py`
- [ ] 读取 `outputs/icr_halu_eval_random_qwen2.5.jsonl`
- [ ] 使用与原导出一致的 tokenizer 对 `response` 重新 tokenize
- [ ] 恢复并保存：
  - `response_token_ids`
  - `response_token_texts`
  - `response_offsets`
  - `num_response_tokens_retokenized`
- [ ] 校验以下字段是否一致：
  - `num_response_tokens`
  - `len(icr_scores[0])`
  - `num_response_tokens_retokenized`
- [ ] 输出对齐统计：
  - 总样本数
  - 完全一致样本数
  - 不一致样本数
  - 不一致样本示例
- [ ] 若不一致率很低，则继续使用该派生数据
- [ ] 若不一致率不可接受，则转入备选方案：
  - 复制并改造 `scripts/compute_icr_halueval.py`
  - 新导出带 token 元数据的增强版 ICR 文件

## Phase 2. 构建两类 span 候选

目标：同时准备 `Tokenizer Window` 和 `spaCy Span` 两条路线。

### 2.1 Tokenizer Window

- [ ] 新建脚本或模块：`build_tokenizer_windows.py`
- [ ] 以 tokenizer token 为基础生成连续窗口
- [ ] 第一轮窗口长度建议：
  - `k = 1`
  - `k = 2`
  - `k = 3`
  - `k = 4`
- [ ] 记录每个窗口的：
  - `sample_id`
  - `token_start`
  - `token_end`
  - `span_text`
  - `window_size`

### 2.2 spaCy Span

- [ ] 新建脚本或模块：`build_spacy_spans.py`
- [ ] 安装并使用适合英文的 spaCy 模型
- [ ] 第一轮优先提取：
  - named entities
  - noun chunks
- [ ] 可选提取：
  - 规则化短语片段
  - 句法相关短片段
- [ ] 将 spaCy span 映射回 tokenizer token 区间
- [ ] 去重与过滤：
  - 空 span
  - 重复 span
  - 超长 span
  - 无法映射的 span

## Phase 3. 构建 silver span labels

目标：在没有 gold span label 的前提下，先建立一个可用的银标体系。

- [ ] 新建脚本：`build_silver_span_labels.py`
- [ ] 读取：
  - 原始 `knowledge`
  - `question`
  - `response`
  - span 候选
- [ ] 设计第一版轻量规则：
  - 数字/日期是否被 knowledge 支持
  - 实体名称是否在 knowledge 中出现
  - 关键名词短语是否有足够 lexical support
- [ ] 对 hallucinated sample 标出“高疑似 unsupported span”
- [ ] 对 correct sample 标出“supported span / non-hallucinated span”
- [ ] 产出银标数据文件
- [ ] 保存每条银标的来源规则与置信度，便于后续排错

## Phase 4. 建立人工核验小集合

目标：避免 silver label 完全失控。

- [ ] 抽样 100 到 200 条样本，构建人工核验集
- [ ] 同时覆盖两种切法：
  - Tokenizer Window
  - spaCy Span
- [ ] 记录以下错误类型：
  - span 边界不合理
  - knowledge 明明支持却被判 unsupported
  - 多个 span 都可算幻觉但漏标
  - 标点或子词导致的错对齐
- [ ] 根据人工核验结果修正规则

## Phase 5. 统一 span 表示

目标：保证后续 5 类方法吃到的基础输入一致、可比。

- [ ] 新建公共数据处理模块
- [ ] 对任意 span 从 `icr_scores[:, start:end]` 取出局部 ICR
- [ ] 第一轮统一用 `mean pooling` 生成 span 级 `[27]`
- [ ] 同时保留元信息：
  - `span_text`
  - `span_type`
  - `span_len_tokens`
  - `char_start`
  - `char_end`
- [ ] 预留后续可选扩展：
  - `max pooling`
  - `top-k pooling`
  - 保留 `[27, span_len]` patch

## Phase 6. 为两种切法分别跑 5 类方法

目标：方法矩阵完整，比较公平。

### 6.1 共同要求

- [ ] 两种切法都单独形成数据集：
  - `tokenizer_window`
  - `spacy_span`
- [ ] 每种切法都分别跑：
  - Baseline MLP
  - Discrepancy Modeling
  - Temporal Conv
  - Change Point Detection
  - Layer Trajectory Encoder

### 6.2 Baseline MLP

- [ ] 新建 span-level Baseline MLP 训练脚本
- [ ] 输入为 span-pooled `[27]`
- [ ] 输出 span-level 概率

### 6.3 Discrepancy Modeling

- [ ] 复用现有 early/middle/late 思路
- [ ] 在 span-level `[27]` 上提取 7 个特征
- [ ] 训练 LR / RF

### 6.4 Temporal Conv

- [ ] 复用 sample-level 的层维度卷积思路
- [ ] 输入仍是 span-level `[27]`
- [ ] 与 Baseline MLP 做同条件比较

### 6.5 Change Point Detection

- [ ] 在 span-level `[27]` 上提取变化点和 roughness 特征
- [ ] 训练 LR / RF

### 6.6 Layer Trajectory Encoder

- [ ] 复用 GRU / Transformer / Deep1DCNN 的 span-level 版本
- [ ] 控制模型规模，优先 CPU 可跑

## Phase 7. 评估方式

目标：同时评估 span-level 和回聚后的 sample-level。

### 7.1 Span-level 评估

- [ ] 指标：
  - AUROC
  - AUPRC
  - F1
- [ ] 区分：
  - silver-label 结果
  - 人工核验子集结果

### 7.2 Sample-level 回聚评估

- [ ] 设计至少 3 种聚合方式：
  - `max`
  - `top-k mean`
  - `noisy-or`
- [ ] 比较聚合后 sample 预测与当前 sample-level baseline
- [ ] 回答两个核心问题：
  - span-level 是否更可解释
  - span-level 回聚后是否保住甚至提升 sample-level 性能

## Phase 8. 可视化与分析

目标：不要只留数字，要能看出 span-level 到底在抓什么。

- [ ] 画出样本级 span 热图
- [ ] 对比两种切法的典型案例
- [ ] 统计：
  - 哪类 span 最常被判为 hallucinated
  - 不同 span 长度的表现
  - tokenizer-window 与 spaCy-span 的优劣
- [ ] 分析 5 类方法在两种切法下的排序变化

## Phase 9. 决策门槛

目标：避免无休止扩展，明确何时继续、何时转向。

- [ ] 若 token 对齐恢复成功且 silver label 基本可用，则继续当前路线
- [ ] 若 token 对齐问题严重，则转为“增强版 ICR 派生导出”
- [ ] 若 HaluEval QA 的短回答仍明显限制 span 信号，则再评估是否换数据集
- [ ] 若两种切法里只有一种明显有效，后续资源集中到那一条

## Phase 10. 后续新对话的启动提示

后续如果在新对话中继续实现，建议先做下面几件事：

- [ ] 先读取本目录的 `README.md`
- [ ] 再读取本目录的 `TODO.md`
- [ ] 优先执行 `Phase 1` 和 `Phase 2`
- [ ] 在真正写代码前，先确认：
  - 继续使用现有 `outputs/icr_halu_eval_random_qwen2.5.jsonl`
  - 是否先只做 CPU 版最小闭环
  - 是否把 Baseline MLP 作为第一批唯一训练目标，再逐步补齐其余 4 类方法
