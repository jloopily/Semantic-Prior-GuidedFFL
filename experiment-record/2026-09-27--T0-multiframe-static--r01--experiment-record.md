---
type: experiment-log
status: completed
experiment_id: T0-A-20260927-001
date: 2026-09-27
experiment_line: CIR-LID-to-static-multiframe
round: 1
purpose: exploratory-negative-result
stage: validation-only
source_artifacts:
  - /Users/phyl/Documents/CIR-LID/server_reports/t0_review_bundle
  - /Users/phyl/.codex/attachments/e93317c6-78c3-4630-8571-3ab214a523e9/已粘贴的文本.txt
related_documents:
  - /Users/phyl/Documents/寻找方向/T0_多帧静态聚合验证_实验方案与AgentPrompt.md
  - /Users/phyl/Documents/寻找方向/CIR-LID_T0后续研究图景与论文标准.md
  - /Users/phyl/Documents/寻找方向/CIR-LID_GateB失败复盘与方向转向_v6.md
server_experiment_namespace: results/t0_multiframe_v1
official_test_read: false
original_synthetic_gate_b: FAIL
---

# T0 多帧静态聚合实验记录

## 1. 实验摘要

本实验是一次独立的探索性验证，不是原 CIR-LID Synthetic Gate B 的补考，也不用于解锁 official test。

实验只回答一个问题：

> 在不改变现有 backbone、真实图库、特征提取、kNN 评分和阈值纪律的前提下，把每个 FF++ validation 视频的采样帧数从 1/3 帧增加到 5/9 帧，是否能够改善静态 patch-level face forgery localization？

### 主要结论

在当前冻结的静态 pooled evaluation 流程下，增加采样帧数没有带来提升。

- c23 上，C2 和 CIR-LID 的方向完全一致；
- T5 相比 T3 的 Pixel AP 稳定下降；
- T9 相比 T3 的 Pixel AP 也稳定下降；
- c40 上点估计方向仍不支持增加帧数，但置信区间包含 0；
- 因此，本实验不支持继续以“增加静态采样帧数”作为性能提升路线；
- 由于没有可靠的跨帧 token correspondence，T0-B 未执行，因此本实验不能判断 temporal mean、temporal variance、temporal MAD 或动态残差是否有效。

当前最稳妥的静态结论是：

> 在本实验协议下，T3 是比 T5/T9 更合理的静态帧预算；增加到 5 帧或 9 帧没有改善定位结果，并且在 c23 上造成了具有 bootstrap 支持的下降。

---

## 2. 实验身份与决策背景

### 2.1 前序研究状态

原 CIR-LID 方向的 Synthetic Gate B 已失败。服务器端修正结果为：

- `Q_CIR = 0.7267`
- `Q_C2 = 0.4945`
- 要求的相对降低：至少 `+10%`
- 实际相对降低：`-46.95%`

因此，本轮 T0 不得被解释为：

- CIR-LID 原主张重新成立；
- 原 Synthetic Gate B 通过；
- CIR-LID 相对 C2 具有已证实的 nuisance 稳定性优势；
- official test 可以解锁。

### 2.2 本轮实验的决策问题

本轮原计划是判断多帧观测是否能够为后续动态真实参考图库提供基础。如果增加帧数本身没有收益，则需要谨慎区分：

1. 多帧静态 pooled evaluation 本身是否不适合当前定位流程；
2. 该负结果是否主要由 crop jitter、空间未对齐或帧质量混杂造成；
3. 在严格对齐之前，是否有理由进入动态残差图库实验。

---

## 3. 实验假设

### 3.1 原始待检验假设

> 相比单帧或三帧观测，增加到五帧或九帧可以通过更多观测降低估计方差，从而提升视频级伪造定位表现。

### 3.2 预先规定的解释边界

即使 T5/T9 有提升，也不能仅凭 pooled evaluation 宣称发现了 temporal inconsistency，因为 pooled evaluation 可能只反映：

- 更多观测像素；
- 更多伪造区域被采样；
- 估计方差变化；
- 不同帧之间的分数分布变化。

本实验实际得到负结果，因此目前只能对静态多帧 pooled 路线作出结论，不能对真正的时序不一致信号作出正面或负面判断。

---

## 4. 实验协议

### 4.1 评分器

本轮使用两个评分器：

1. `arithmetic_multi_neighbor / C2`：作为当前主要静态评分器；
2. `CIR-LID`：作为历史主方法诊断评分器，不作为 Gate B 通过证据。

### 4.2 数据与划分

使用 FF++ validation 数据：

- validation real c23：用于阈值校准和真实像素误报诊断；
- validation fake c23：主评估条件；
- validation real c40：压缩诊断；
- validation fake c40：压缩诊断；
- manipulation：Deepfakes、Face2Face、FaceSwap、NeuralTextures。

服务器审阅结果显示：

```text
train_validation_overlap = []
test_split_read = false
test_videos_read = false
```

因此未发现 train/validation 身份重叠，official test 未读取。

### 4.3 固定帧预算

归一化位置按照以下规则映射到实际帧：

```python
frame_index = round(position * (num_frames - 1))
```

固定配置如下：

| 配置 | 采样位置 | 目标 |
|---|---|---|
| `T1_center` | `[0.50]` | 单帧中心基线 |
| `T3_frozen` | `[0.25, 0.50, 0.75]` | 当前冻结静态基线 |
| `T5_uniform` | `[0.10, 0.30, 0.50, 0.70, 0.90]` | 五帧均匀采样 |
| `T9_uniform` | `[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]` | 九帧均匀采样 |

采样审计结果：

```text
requested_videos = 700
valid_videos = 700
duplicate_frames = 0
invalid_frames = 0
```

每个视频均按预定帧预算完成采样，没有发现重复帧或无效帧。

### 4.4 T0-A 评估方式

T0-A 对每一帧独立计算静态 28×28 score map，并按照既有流程上采样到评估分辨率。不同帧之间没有在未确认 correspondence 的情况下直接进行 token-wise mean/std 计算。

视频级评价按照以下逻辑执行：

1. 保存每个视频、每个采样帧的 score 和 mask 信息；
2. 将同一视频所选帧的预测像素和 mask 像素进行 pooled evaluation；
3. 计算 Pixel AP、Pixel AUROC、固定阈值 F1、固定阈值 IoU 等指标；
4. 先在 manipulation 内进行视频级宏平均；
5. 再对四种 manipulation 做等权 macro average。

因此，报告中的总体 Pixel AP 应理解为：

```text
manipulation-macro Pixel AP
```

而不是把全部视频和帧直接混合后的全局像素池结果。

### 4.5 阈值

阈值流程符合冻结协议：

- 阈值来自 validation real c23；
- 以 `T3_frozen` 的校准帧为基础；
- 校准帧数为 `420`；
- 使用 `99th percentile`；
- 同一评分器阈值用于所有帧预算和两个压缩等级；
- 未使用 fake mask 调阈值；
- 未使用逐视频最优阈值。

阈值为：

```text
CIR-LID                    = 32.41762924194336
arithmetic_multi_neighbor = 0.9685385823249817
```

---

## 5. 主要结果

### 5.1 c23：C2 / arithmetic_multi_neighbor

| 比较 | Pixel AP 差值 | 95% bootstrap CI | 解释 |
|---|---:|---:|---|
| T3 − T1 | -0.002782 | [-0.008324, 0.002814] | 点估计下降，但 CI 跨 0 |
| T5 − T3 | **-0.005103** | **[-0.009021, -0.000887]** | CI 完全低于 0，稳定下降 |
| T9 − T3 | **-0.004181** | **[-0.007873, -0.000437]** | CI 完全低于 0，稳定下降 |

C2 在 c23 上的主要趋势为：

```text
T1 > T3 > T5
T1 > T3 > T9
```

这里的 `>` 表示对应的 Pixel AP 点估计更高，不代表每一个单独视频都满足该顺序。

### 5.2 c23：CIR-LID

| 比较 | Pixel AP 差值 | 95% bootstrap CI | 解释 |
|---|---:|---:|---|
| T3 − T1 | -0.003168 | [-0.008623, 0.002416] | 点估计下降，但 CI 跨 0 |
| T5 − T3 | **-0.004931** | **[-0.008893, -0.000732]** | CI 完全低于 0，稳定下降 |
| T9 − T3 | **-0.004063** | **[-0.007722, -0.000295]** | CI 完全低于 0，稳定下降 |

CIR-LID 与 C2 的方向完全一致。因此，本实验没有发现“CIR-LID 对增加帧数更有利”的迹象。

### 5.3 c40：压缩诊断

c40 上总体方向仍不支持增加帧数，但统计证据弱于 c23。

C2：

| 比较 | Pixel AP 差值 | 95% bootstrap CI |
|---|---:|---:|
| T5 − T3 | -0.004014 | [-0.008293, 0.000146] |
| T9 − T3 | -0.001227 | [-0.005081, 0.002540] |

CIR-LID：

| 比较 | Pixel AP 差值 | 95% bootstrap CI |
|---|---:|---:|
| T5 − T3 | -0.003757 | [-0.007980, 0.000389] |
| T9 − T3 | -0.001053 | [-0.004856, 0.002686] |

严谨表述为：

- c23：T5/T9 相比 T3 的下降具有 paired bootstrap 支持；
- c40：点估计方向仍偏向下降，但置信区间包含 0；
- 不能写成“两个压缩等级上都显著下降”。

---

## 6. 数据完整性与执行质量审计

### 6.1 执行状态

服务器任务各阶段正常完成：

```text
prepare  : completed
gallery  : completed
crops    : completed
features : completed
scores   : completed
evaluate : completed
EXIT_CODE: 0
```

日志中未发现 `Traceback`、`Error`、`Exception` 或任务失败记录。

### 6.2 Crop 缓存

```text
rows          = 15400
cached_images = 15400
cached_masks  = 12320
failed_rows   = 0
excluded_rows = 0
```

没有发现因裁剪失败导致的样本丢失。

### 6.3 共同 Pixel AP cohort

总体 Pixel AP 的共同 cohort 为 `554` 个视频，而不是最初的 `560` 个视频。

排除原因是 `single_class_mask`，涉及：

- Face2Face：`115_939`、`492_325`、`824_419`；
- NeuralTextures：`115_939`、`492_325`、`824_419`。

这些视频在部分或全部预算下没有同时存在正负 mask 类别，Pixel AP/AUROC 无法定义。处理方式为：

- 不强行填入 0；
- 不伪造 AP；
- 所有帧预算和压缩条件使用共同定义 cohort；
- 最终 Pixel AP 使用 554 个共同视频。

各 manipulation 的共同视频数为：

```text
Deepfakes       = 140
Face2Face       = 137
FaceSwap        = 140
NeuralTextures  = 137
```

### 6.4 Real FPR 的解释边界

c23 T3 的阈值本身由 validation real c23 T3 校准，因此 c23/T3/real FPR 约为 1% 是协议设计下的预期现象，不是独立测试集泛化证据。

c40 的 real-pixel FPR 约为 1.7%，更适合解释为 compression shift 的诊断信号，不能直接归因于 T5/T9。

---

## 7. T0-B 状态与未解决问题

T0-B 未执行。服务器记录的原因是：

```text
Temporal mean/variance maps were not evaluated because token-wise temporal correspondence was not guaranteed.
```

因此，本实验没有测试：

- temporal mean map；
- temporal std map；
- temporal MAD map；
- dynamic residual gallery；
- temporal frequency；
- token trajectory statistics。

这意味着当前结果不能支持以下结论：

- temporal inconsistency 不存在；
- 动态真实参考图库一定无效；
- 时序差分一定不能改善定位；
- T1 动态残差方向已经被证伪。

### 7.1 可能的混杂因素

当前结果中，下列因素尚未进行因果归因：

- crop jitter；
- 帧间人脸空间错位；
- 不同帧的模糊程度；
- 多帧 pooled 后引入的边界或背景响应；
- 伪造区域在不同帧之间的大小和位置变化；
- 高分区域是否集中在嘴、眼、发际线等天然运动区域。

服务器诊断对此记录为：

```text
crop_jitter_causal_attribution = not_evaluated
blur_interpretation = descriptive only
anatomical_high_score_regions = not_evaluated
```

因此，这些只能作为待验证解释，不能写成当前实验已经证明的原因。

---

## 8. 证据等级与结论审计

### 8.1 已被本实验支持的结论

1. 当前实验执行完成且未发现执行错误；
2. official test 未读取；
3. 原 Synthetic Gate B 仍为 FAIL；
4. 在当前静态 pooled 流程下，T5/T9 没有改善 c23 上的 Pixel AP；
5. c23 上 C2 与 CIR-LID 的方向一致；
6. c23 上 T5-T3 和 T9-T3 的下降具有 95% paired bootstrap 支持；
7. T0-B 因缺少可靠 token correspondence 未执行；
8. 当前证据不足以支持“更多静态采样能够改善定位”。

### 8.2 当前不能声称的结论

1. 时序信息本身没有价值；
2. temporal mean/std/MAD 无效；
3. 动态残差图库无效；
4. T1 动态路线已经被证伪；
5. crop jitter 是造成下降的确定原因；
6. c40 上也存在统计显著下降；
7. CIR-LID 在所有真实人脸伪造任务上均失败。

### 8.3 探索性解释

当前最合理但尚未证实的解释是：

> 静态 score map 在不同帧之间缺乏可靠的空间/token correspondence，多帧 pooled evaluation 可能把 crop jitter、帧质量差异和非稳定边界响应一并纳入评估，从而抵消了增加观测数量可能带来的收益。

这只是下一轮诊断假设，不是本实验的正式结论。

---

## 9. 研究决策记录

### 当前决定

- `T3_frozen` 暂定为当前静态方法的默认帧预算；
- `T5_uniform` 和 `T9_uniform` 不作为性能提升配置；
- CIR-LID 不恢复为主方法；
- C2 保留为主要静态评分器；
- T0 不解锁 official test；
- 暂停 T1 动态残差图库、频域分支和 3D/复杂时序扩展；
- 不重复运行 T0 以追求更好的数字；
- 保留完整 raw outputs 和 bootstrap 结果。

### 暂不做最终方向裁决

由于 T0-B 未执行，本记录不把整个“动态真实参考”方向判定为失败。是否继续，只能在后续一次明确的 alignment/crop-jitter 诊断之后决定。

---

## 10. 建议的待办事项

当前只保留以下一个可选后续任务：

### T0.1：跨帧对齐与质量混杂诊断

目的不是追求更高 Pixel AP，而是判断 T0 的下降是否可能由 pipeline 混杂造成。

建议记录：

- 相邻帧 face crop 中心、宽高和 IoU；
- landmark 或相似变换的变化；
- 人脸检测置信度；
- 清晰度指标；
- 人脸框大小和姿态变化；
- T1/T3/T5/T9 各配置的有效质量分布；
- 如果具备可靠的 canonical alignment，再进行一次固定对齐的验证性比较。

T0.1 必须遵守：

- 不使用 fake mask 选择帧；
- 不按 validation fake 结果删除帧；
- 不修改原 T0 结果；
- 不把诊断结果写成 T0 主结果；
- 不在没有确认 correspondence 前计算 token-wise temporal statistics。

### T0.1 后决策

- 若对齐/质量混杂很小且下降仍存在：终止多帧/动态扩展路线；
- 若对齐后下降明显减弱：只允许设计一个小规模、real-only、video-disjoint 的 T1 pilot；
- 若只发现 pooled gain 而没有独立 dynamic gain：只能保留“多帧静态观测”的保守结论；
- 无论哪种情况，均不得因此恢复原 CIR-LID Gate B 或解锁原 official test。

---

## 11. 可复现性与产物索引

服务器端主要结果目录：

```text
/Users/phyl/Documents/CIR-LID/server_reports/t0_review_bundle
```

预期/相关结果文件包括：

```text
prepare
 gallery
crops
features
scores
evaluate
frame manifest
threshold summary
per-video metrics
aggregate metrics
bootstrap differences
diagnostics
summary
```

本地相关方案文档：

- `/Users/phyl/Documents/寻找方向/T0_多帧静态聚合验证_实验方案与AgentPrompt.md`
- `/Users/phyl/Documents/寻找方向/CIR-LID_T0后续研究图景与论文标准.md`
- `/Users/phyl/Documents/寻找方向/CIR-LID_GateB失败复盘与方向转向_v6.md`

### 记录限制

本记录依据用户提供的服务器端完整结果审阅摘要整理。当前工作区没有直接读取服务器目录中的原始 JSONL、日志和 bootstrap 文件，因此本记录不替代服务器原始产物；数值和状态均按提供的审阅结果转录。

---

## 12. 一句话归档结论

> T0-A 执行完整且协议无明显违规；增加到 5/9 帧没有改善当前静态 pooled localization，并在 c23 上稳定下降。该结果否定了“更多静态采样即可提升性能”的假设，但由于缺少可靠跨帧 correspondence，尚未否定真正的动态异常信号；下一步应先保存阴性结果，暂缓方法堆叠，再决定是否进行一次对齐与 crop-jitter 诊断。
