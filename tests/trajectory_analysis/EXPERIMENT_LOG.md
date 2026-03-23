# ICR 跨层轨迹异常分析 — 实验记录

## 实验背景

- **目标**：验证"跨层轨迹异常"思路对幻觉检测的可行性，为后续 span-level 扩展提供依据
- **数据**：HaluEval QA 数据集，10,000 样本（5010 幻觉 / 4990 正确），模型 Qwen2.5-7B-Instruct
- **ICR 数据格式**：每个样本有 `[28层 × N个token]` 的 ICR 分数矩阵（layer 27 全为 0，实际用 0-26 共 27 层）
- **标签粒度**：sample-level（整条回复标为幻觉/正确），非 token-level
- **评估方式**：5-fold stratified CV，指标为 AUROC / AUPRC / F1
- **数据表示**：所有方法均先对 token 维度取平均，得到 `[n_samples, 27]` 的 sample-level 轨迹

---

## Method 1: Discrepancy Modeling（阶段差异建模）

### 思路

将 27 层分为三个阶段，计算阶段间的统计差异作为特征，检验幻觉样本在不同抽象阶段是否存在不一致。

### 实现

- 分段：early(0-8) / middle(9-17) / late(18-26)，每段 9 层
- 提取 7 个手工特征：
  1. `mean_early` — 浅层均值
  2. `mean_mid` — 中层均值
  3. `mean_late` — 深层均值
  4. `diff_mid_early` = mid - early
  5. `diff_late_mid` = late - mid
  6. `diff_late_early` = late - early
  7. `slope` — 27 层线性回归斜率
- 分类器：Logistic Regression / Random Forest

### 结果

| 模型 | AUROC | AUPRC | F1 |
|------|-------|-------|-----|
| 单特征 diff_late_mid | 0.8962 | — | — |
| 单特征 mean_mid | 0.8162 | — | — |
| LR (7 特征) | 0.9213±0.0074 | 0.9148±0.0098 | 0.8547±0.0103 |
| RF (7 特征) | 0.9237±0.0070 | 0.9147±0.0080 | 0.8612±0.0070 |

### 发现

- `diff_late_mid`（late 和 middle 的差值）单特征就有 0.896 的 AUROC，是最强的单一指标
- 说明幻觉样本在 middle→late 层段的 ICR 演化模式确实和正确样本不同
- 但 7 个特征组合（0.92）远低于直接用 27 维向量的 baseline（0.97），说明粗粒度分段丢失了较多信息

### 生成文件

- `01_discrepancy/figures/trajectory_comparison.png` — 幻觉 vs 正确的均值±std轨迹对比
- `01_discrepancy/figures/feature_violins.png` — 7 个特征的 violin plot
- `01_discrepancy/figures/roc_comparison.png` — ROC 曲线对比
- `01_discrepancy/figures/scatter_discrepancy.png` — diff_mid_early vs diff_late_mid 散点图

---

## Method 2: Temporal Conv Over Layers（层维度时序卷积）

### 思路

将每个样本的 27 层 ICR 向量当作一维信号，用不同 kernel size 的 Conv1d 捕捉局部层段模式（例如"连续几层突然升高"）。

### 实现

三个 PyTorch 模型：

1. **BaselineMLP**：`[27] → Linear(64) → Linear(32) → Linear(1)`，LeakyReLU + BN + Dropout(0.3)
2. **TemporalCNN**：三路并行 Conv1d（kernel=3/5/7，各 16 filters）→ concat → GlobalAvgPool → Linear(1)
3. **MultiScaleCNN**：两层 Conv1d（kernel=3/5）+ 残差连接 → GlobalAvgPool → Linear(1)

训练参数：BCELoss, Adam lr=1e-3, weight_decay=1e-4, 100 epochs, early stopping patience=10

### 结果

| 模型 | AUROC | AUPRC | F1 |
|------|-------|-------|-----|
| BaselineMLP | **0.9862±0.0033** | **0.9875±0.0037** | **0.9545±0.0041** |
| MultiScaleCNN | 0.9836±0.0033 | 0.9848±0.0040 | 0.9486±0.0021 |
| TemporalCNN | 0.9717±0.0031 | 0.9729±0.0037 | 0.9302±0.0056 |

### 发现

- BaselineMLP 是所有方法中最强的（AUROC=0.986）
- 卷积模型不仅没有提升，反而略低于 MLP
- 原因分析：sample-level 下 27 维向量的全连接已经足够学到任意层组合，卷积的局部性约束反而限制了表达力
- 结论：在当前 sample-level 任务上，1D 卷积没有优势

### 生成文件

- `02_temporal_conv/figures/training_curves.png` — 训练曲线
- `02_temporal_conv/figures/roc_comparison.png` — ROC 对比
- `02_temporal_conv/figures/model_comparison.png` — 三模型柱状图对比

---

## Method 3: Change Point Detection（变化点检测）

### 思路

检测每个样本轨迹中的突变点——如果幻觉样本的跨层轨迹在某些层出现不自然的突变，那么变化点的数量、位置、幅度等统计量应该和正确样本不同。

### 实现

- 计算一阶差分 `d1[i] = traj[i+1] - traj[i]` 和二阶差分 `d2`
- 阈值法检测变化点：`|d1[i]| > mean(|d1|) + 2×std(|d1|)`
- 提取 8 个特征：
  1. `n_cp_d1` — 一阶变化点数量
  2. `n_cp_d2` — 二阶变化点数量
  3. `first_cp_loc` — 第一个变化点位置
  4. `max_change_loc` — 最大变化发生的层位置
  5. `max_change_mag` — 最大变化幅度
  6. `mean_change_mag` — 平均变化幅度
  7. `roughness` — 一阶差分方差（轨迹粗糙度）
  8. `max_sharpness` — 最大二阶差分（最尖锐的转折）
- 分类器：Logistic Regression / Random Forest

### 结果

| 模型 | AUROC | AUPRC | F1 |
|------|-------|-------|-----|
| 单特征 roughness | 0.7158 | — | — |
| 单特征 max_change_mag | 0.7018 | — | — |
| LR (8 特征) | 0.8114±0.0131 | 0.7497±0.0174 | 0.7765±0.0060 |
| RF (8 特征) | 0.8984±0.0072 | 0.8843±0.0100 | 0.8369±0.0071 |

### 发现

- 变化点特征整体区分能力最弱（RF: AUROC=0.90），但仍显著高于随机
- `roughness`（轨迹粗糙度）是最好的单特征，说明幻觉样本的逐层变化确实更剧烈
- 变化点数量本身（n_cp_d1=0.52）几乎无区分力，说明幻觉不是简单的"变化点更多"
- RF 比 LR 高很多（0.90 vs 0.81），说明特征间存在非线性交互
- 这个方法的价值更多在**解释性**：可以说"幻觉在第 X 层开始出现异常"

### 生成文件

- `03_change_point/figures/example_trajectories.png` — 示例轨迹 + 标注变化点
- `03_change_point/figures/cp_frequency_by_layer.png` — 各层位置的变化点频率（幻觉 vs 正确）
- `03_change_point/figures/d1_distribution.png` — 一阶差分分布对比
- `03_change_point/figures/feature_violins.png` — 8 特征 violin plot
- `03_change_point/figures/roc_comparison.png` — ROC 对比

---

## Method 4: Layer Trajectory Encoder（轨迹编码器）

### 思路

用学习型序列编码器对 27 层 ICR 轨迹建模，让模型自动学习"什么样的跨层演化模式像幻觉"。

### 实现

三个 PyTorch 编码器，均将 `[27]` 维轨迹当作 seq_len=27, feature=1 的序列：

1. **GRU**：BiGRU(hidden=32, layers=2, dropout=0.3) → 拼接正反向最后隐状态 [64] → Linear(1)
2. **SmallTransformer**：Linear(1→32) + positional encoding → TransformerEncoder(d=32, heads=4, layers=2) → mean pool → Linear(1)
3. **Deep1DCNN**：Conv1d(1→32, k=3) → Conv1d(32→64, k=3) → Conv1d(64→64, k=3) → AdaptiveAvgPool → Linear(1)，全部 BN + LeakyReLU

训练参数：BCELoss, Adam lr=1e-3, weight_decay=1e-4, 50 epochs, early stopping patience=8

### 结果

| 模型 | AUROC | AUPRC | F1 |
|------|-------|-------|-----|
| Deep1DCNN | 0.9810±0.0040 | 0.9827±0.0047 | 0.9440±0.0038 |
| Transformer | 0.9764±0.0056 | 0.9769±0.0061 | 0.9316±0.0085 |
| GRU | 0.9696±0.0043 | 0.9692±0.0064 | 0.9245±0.0054 |

### 发现

- Deep1DCNN 最好（0.981），但仍不如 Method 2 的 BaselineMLP（0.986）
- Transformer 和 GRU 更弱，可能因为 27 维序列太短、数据量不够大，序列模型的优势发挥不出来
- 结论：在 sample-level + 10k 数据的条件下，复杂编码器没有优势

### 生成文件

- `04_trajectory_encoder/figures/training_curves.png` — 训练曲线
- `04_trajectory_encoder/figures/roc_comparison.png` — ROC 对比
- `04_trajectory_encoder/figures/tsne_transformer.png` — Transformer 隐表示的 t-SNE 可视化

---

## 总排名

| 排名 | 方法 | AUROC | F1 |
|------|------|-------|-----|
| 1 | BaselineMLP (Method 2) | **0.9862** | **0.9545** |
| 2 | MultiScaleCNN (Method 2) | 0.9836 | 0.9486 |
| 3 | Deep1DCNN (Method 4) | 0.9810 | 0.9440 |
| 4 | Transformer (Method 4) | 0.9764 | 0.9316 |
| 5 | TemporalCNN (Method 2) | 0.9717 | 0.9302 |
| 6 | Raw 27-dim LR (Baseline) | 0.9697 | 0.9230 |
| 7 | GRU (Method 4) | 0.9696 | 0.9245 |
| 8 | Discrepancy RF (Method 1) | 0.9237 | 0.8612 |
| 9 | Discrepancy LR (Method 1) | 0.9213 | 0.8547 |
| 10 | ChangePoint RF (Method 3) | 0.8984 | 0.8369 |
| 11 | ChangePoint LR (Method 3) | 0.8114 | 0.7765 |

---

## 总结

1. **所有方法都显著高于随机（0.5）**，说明 ICR 跨层轨迹确实包含幻觉判别信息
2. **没有任何轨迹方法超过 BaselineMLP**（直接 27 维 → MLP），说明在 sample-level 任务上，原始层向量已经足够，结构化建模反而约束了信息利用
3. **手工特征方法（Method 1, 3）明显弱于学习方法（Method 2, 4）**，信息压缩过狠
4. **当前实验的局限**：标签是 sample-level 的，无法测试这些方法在 token-level / span-level 定位上的真正潜力
5. **有价值的发现**：
   - `diff_late_mid` 单特征 AUROC=0.896，说明幻觉在 middle→late 层段有明确的异常模式
   - 轨迹 roughness 可区分幻觉（0.716），幻觉样本的逐层变化更剧烈
   - 变化点数量无区分力，说明幻觉不是"突变更多"，而是"变化模式不同"

## 下一步方向

- 获取 token-level / span-level 标注，在细粒度任务上重新评估
- 将轨迹特征作为可解释性工具，分析"幻觉在哪些层段形成"
- 考虑将轨迹特征与原始 27 维向量拼接，而非替代
