---
type: experiment-log
status: completed-with-reporting-caveat
experiment_id: M1-A-20260928-001
date: 2026-09-28
review_date: 2026-09-29
experiment_line: CIR-LID-limited-static-forgery-localization
round: 1
purpose: preregistered-validation-pilot
stage: validation-only
frozen_decision: PRIMARY_NOT_SUPPORTED_SECONDARY_SIGNAL
source_artifacts:
  - /Users/phyl/Documents/CIR-LID/server_reports/m1_review_bundle/results/m1_static_geometry_pilot_v1
  - /Users/phyl/Documents/CIR-LID/server_reports/m1_review_bundle/logs
  - /Users/phyl/Documents/CIR-LID/server_reports/m1_review_bundle/m1_review_bundle_20260928.tar.gz
related_documents:
  - /Users/phyl/Documents/寻找方向/M1_受限静态伪造定位验证_AgentPrompt_v1.md
  - /Users/phyl/Documents/CIR-LID/experiment-record/2026-09-27--M0-static-real-manifold-audit--r01--experiment-record.md
  - /Users/phyl/Documents/CIR-LID/experiment-record/2026-09-27--T0-multiframe-static--r01--experiment-record.md
server_result_namespace: results/m1_static_geometry_pilot_v1
official_test_read: false
official_test_locked: true
original_synthetic_gate_b: FAIL
---

# M1 受限静态伪造定位验证：实验记录与结果审阅

> 记录性质：对服务器 M1 结果包的只读审阅。文中的统计数值来自冻结实验产物及逐视频表；未在本地重跑模型、重新抽样 bootstrap 或改变冻结决策。`PRIMARY_NOT_SUPPORTED_SECONDARY_SIGNAL` 是服务器冻结规则产生的状态，不是主方法通过。

## 1. 决策摘要

本轮在 FF++ validation 的冻结 T3 帧预算、真实图库和固定特征/近邻计算下，依次检验：fake 对应真实图像的分数是否变化，以及这种变化是否确实落在 annotation-defined fake mask 内。**冻结主评分器 `r_k` 的静态定位证据为负**：c23 的面积校正增益 U 和配对真实增益 D 均小于零，且 source-group bootstrap 区间不跨零；区域差分 R 也小于零。

预登记次级评分器 `C2` 的 c23 配对增益 D 为正，达到冻结规则的候选触发条件，因此服务器状态为 **`PRIMARY_NOT_SUPPORTED_SECONDARY_SIGNAL`**，并按预登记方案执行小样本混杂诊断。但是 `C2` 的面积校正增益 U 的 95% 和 97.5% bootstrap 区间均跨零；真实图像空间模板的 AP 也高于 `C2` 热图 AP，空间部位/位置混杂尚未排除。这只能记作**有限的次级候选信号**，不能写成主分析支持定位、CIR-LID 恢复有效或跨身份泛化成立。

**执行边界：**原 Synthetic Gate B 继续为 `FAIL`；M0 仍是暂定的真实结构审计；本轮不是 untouched validation 或 official test。保持 official test 锁定，不自动启动下一轮实验。

## 2. 实验协议和样本单位

- 实验命名空间：`m1_static_geometry_pilot_v1`；版本 `M1-limited-v1.0`。事前冻结配置见结果目录中的 `preregistered_config.yaml`；`freeze_receipt.json` 标记 `status=frozen`、`freeze_before_fake_metrics=true`。
- 真实 gallery：train real c23，64 视频、50,176 token；DINOv3 ViT-B/16 第 6 block，28×28、768 维特征；每 query token 取 20 个精确欧氏近邻；主评分 `r_k=r_20`，固定次级为 `mean_r`、`C2`，`CIR_LID` 仅为历史诊断。方向均按冻结定义，不因结果而翻转。
- score map：28×28 双线性放大至 448×448，无后处理。主分析为 validation c23；c40 仅用同一 c23 gallery 和固定阈值做转移诊断。四类 manipulation 均保留。
- 每个 fake 视频汇总 T3 帧像素后计算 Pixel AP，再按四种 manipulation 等权 macro。主要量：`U=AP_fake−mask_positive_fraction`、`D=AP_fake−AP_paired_target_real_with_fake_virtual_mask`。辅助量 `R=mask 内平均(fake−real)分数−mask 外平均(fake−real)分数`；R 不是部署评分器，且不同 scorer 的数值量纲不能直接横比。
- 每个 scorer 的阈值在 validation real c23 上按 score-map 像素第 99 百分位冻结，`method=linear`，预测为 `score > threshold`。real c23 约 1% FPR 是**同样本校准检查**，并非独立泛化验证。
- 结果包含 4,200 条 query frame（840 real、3,360 fake）；70 个已核验的 target/source 关联组。统计不把三帧、共享原视频或不同 manipulation 的结果当作独立个体。关联组 bootstrap 为 2,000 次；历史 video-stratified bootstrap 仅辅助参考。

## 3. 冻结主分析结果

以下为 c23 四类 manipulation 等权 macro；区间为 70 个 source group 的 2,000 次 bootstrap 百分位区间。主 U/D 的 97.5% 区间用于冻结方案的双比较覆盖控制。

| 评分器和指标 | 点估计 | 95% CI | 97.5% CI | 解释 |
|---|---:|---:|---:|---|
| `r_k`：U | −0.1676 | [−0.1703, −0.1643] | [−0.1706, −0.1638] | 主方法低于各视频自身 mask 占比基线 |
| `r_k`：D | −0.0047 | [−0.0072, −0.0023] | [−0.0075, −0.0019] | 不优于对应真实帧的虚拟 mask 对照 |
| `r_k`：R | −0.2589 | [−0.3903, −0.1507] | [−0.4124, −0.1381] | 内外区域差分方向不支持定位 |
| `C2`：U | +0.0110 | [−0.0009, +0.0238] | [−0.0021, +0.0251] | 正点估计，但区间跨零，且低于主方法预设的 +0.02 实用效应下限 |
| `C2`：D | +0.0253 | [+0.0201, +0.0301] | [+0.0195, +0.0308] | 符合固定次级候选的 D 触发条件 |
| `C2`：R | +0.001686 | [+0.001383, +0.001968] | [+0.001343, +0.002001] | 同评分器量纲下的探索性区域差分 |

`r_k` c23 Pixel AP 为 0.3082，固定阈值 F1 为 0.0010、IoU 为 0.0005；四种 manipulation 的 U 均为负，D 仅 Deepfakes 的点估计略为正（+0.0030），其余三类为负。`mean_r` c23 U 为 −0.1653，D 为 −0.0042，方向与主方法一致。`C2` c23 Pixel AP 为 0.4868、配对真实虚拟 mask AP 为 0.4615、固定阈值 F1 为 0.0250、IoU 为 0.0133；四类的 D 点估计分别为 Deepfakes +0.0238、Face2Face +0.0302、FaceSwap +0.0375、NeuralTextures +0.0098。这些子组点估计不构成各子组显著性证明。

`r_k` 的全图 fake−real 平均分数位移为正（+0.4172），但 R 为负：**全局分数变化不能替代区域定位证据**。`CIR_LID` 即使有某些正的次级指标，也不能提升为本轮主评分器，更不能改变原 Synthetic Gate B 的失败结果。

## 4. 压缩转移与条件混杂诊断

### 4.1 c40 固定阈值转移

- `r_k` c40 U = −0.1640、D = −0.0034，未改变主结论；真实 c40 FPR 为 **0.6337%**，相对 c23 校准检查约 1.00% 下降。
- `C2` c40 U = +0.0429、D = +0.0254，Pixel AP = 0.5187；真实 c40 FPR 为 **1.7851%**，高于 c23 校准检查约 1.00%。这是同内容/同阈值的转移诊断，不是 untouched identity 泛化。不能为补救结果在 c40 重新校准阈值。

### 4.2 预登记的 16 关联组诊断

服务器决策触发后，按预先选择的 16 个 source group 执行了范数、gallery 规模与空间对照。每个变体/压缩/scorer 为 128 条视频指标记录。这些是**探索性的固定小样本描述量**，不是主分析的替代结果，也不具备此处主分析的组级 CI。

| c23 `C2` 诊断变体 | U macro | D macro |
|---|---:|---:|
| gallery 前 16 视频 | −0.0264 | +0.0118 |
| gallery 前 32 视频 | −0.0039 | +0.0159 |
| gallery 前 48 视频 | +0.0022 | +0.0153 |
| gallery 前 64 视频 | +0.0087 | +0.0151 |
| matched-count unrestricted 空间对照 | −0.0813 | −0.0072 |
| L2 normalized features | +0.0180 | +0.0168 |
| same-token-index 空间对照 | +0.0544 | +0.0120 |

16 组子集的原始 64-gallery D（+0.0151）低于全样本 D（+0.0253）；不能把小样本对照的点估计当成全样本机制证明。尤其 matched-count unrestricted 改变 D 方向，提示评分依赖图库/空间匹配方式；normalized 特征又改变了特征定义。原始 `r_k` 在各 gallery 规模下的 U 约 −0.17、D 约 −0.007 至 −0.008，未出现对主方法有利的稳健翻转。

### 4.3 空间模板限制

c23 上，`C2` 的真实图像平均空间模板用于 fake mask 的 AP 为 **0.5152**，比原始 `C2` fake Pixel AP **0.4868** 还高；c40 相应为 0.5152 与 0.5187。两者尚无专门的配对差值 CI，不能据此宣称“空间偏置完全解释了 D”，但确实**阻止**把 D>0 直接归因于伪造区域识别。嘴/眼等正常面部部位、中心位置以及 mask 模板关联依然是实质混杂。

## 5. 审阅发现的报告口径问题

服务器 `evaluation_summary.json` 的 `fake_video_metric_rows=4480` 是**已生成的 fake×scorer 记录数**，不是 4,480 条 AP 均有定义的记录；`exclusions=0` 指配对/输入级排除记录为零。逐视频原始表中，**4,448 条 `eligible`、32 条 `LOCALIZATION_UNDEFINED`**。后者来自 Face2Face 与 NeuralTextures 的 `115_939`、`824_419` 两个 video ID，在 c23/c40 上均为全零 mask（每条件 4 scorers）。因此每个 compression、每个 scorer 的 AP 有定义视频数为 **140/138/140/138**（依 manipulation 顺序），合计 556，而非 560。

生成的 `analysis-report.md` 将 4,480 写作 “Eligible fake scorer rows”，此处措辞不准确，引用时应改为“4,480 条生成的 fake×scorer 行，其中 32 条定位 AP 无定义”。生成报告还将 AP、mask 占比和 U 各自的 macro 并列表述，但**mask 占比的汇总包含全零 mask 行，AP/U 汇总不包含**；因此读者不能简单拿表中 AP macro 减去表中 mask 占比 macro 来核对 U。F1/IoU 汇总也包含这类全零 mask 视频，需标注支持域差异。这是**报告与分母口径的限制**，不是许可事后删改样本、重算或改变已冻结决策。

## 6. 完整性、运行经过与可复核范围

- 结果中的 `freeze_receipt.json` 标记在 fake 指标前冻结；`diagnostics.json` 记录 source grouping 与 pairing 均为 `verified`、数值 NaN/Inf 计数 0、fake mask 未用于拟合/调参、未读 official test、official test 仍锁定。`score_diagnostics.json` 记录 4,200 score 行及 0 个 zero-distance/neighbor-duplicate/numerical-floor 事件。
- `prepare` 最终成功；`crops` 复用了 4,200 个 crop；`features` 记录新提取 4,200 行、复用 0 行；`scores`、`calibrate`、修复后 `evaluate`、`bootstrap`、修复后 `diagnostics`、`finalize` 均有成功产物。早期 evaluate 的 `root` 未定义和 diagnostics 的 cache-key 问题见失败日志；后续重试成功。不能将早期失败隐去为从未出现异常，也不能将其当成最终实验失败。
- 下载包 `m1_review_bundle_20260928.tar.gz` 本地 SHA-256 为 `4f66c7498ec01f6093d28976a2804fb9d63467a3d6443df3fa8dc1ad8e44ab95`，与同包 `.sha256` 文件相符。冻结回执中 `git_state.available=false`，无 commit ID；因此虽然产物和打包的代码可审阅，仍不能仅凭冻结回执独立证明服务器运行时的完整 Git 版本。包中不含完整上游数据与 workspace，本审阅**不是原始视频/特征的端到端复现**。
- validation fake 曾用于 T0，故本轮即使先冻结，也不是未见过的数据上的独立确认；gallery/validation 的已核验隔离不等同于 identity-disjoint 泛化证明。

## 7. 最终可用表述与停止条件

**允许表述：**“在本次 validation-only、T3、固定真实图库的受限 M1 pilot 中，主评分器 `r_k` 未获得静态伪造定位支持。固定次级 `C2` 在 c23 的 fake−paired-real 虚拟 mask AP 上有正的候选差异，但面积基线不稳健、空间模板及图库/位置混杂未排除。”

**禁止表述：**“M1 通过”“CIR-LID 定位有效”“已证明真实流形的法向伪造偏离”“`C2` 是无混杂的定位器”“已达到 official test/投稿门槛”。本轮不自动改变 scorer、阈值、方向、mask 或范围；不解锁 official test；不自动启动新实验。

## 8. 原始产物索引

结果根目录：`/Users/phyl/Documents/CIR-LID/server_reports/m1_review_bundle/results/m1_static_geometry_pilot_v1`

- 协议与来源：`preregistered_config.yaml`、`freeze_receipt.json`、`input_provenance.json`、`pairing_manifest.jsonl`、`source_groups.json`、`scorer_registry.json`。
- 主结果与推断：`per_video_metrics.jsonl`、`per_frame_diagnostics.jsonl`、`paired_region_statistics.jsonl`、`aggregate_metrics.json`、`bootstrap_differences.json`、`bootstrap_cluster_results.npz`、`decision.json`。
- 诊断与转移：`threshold_summary.json`、`score_diagnostics.json`、`diagnostics.json`、`confound_diagnostics/aggregate.json`、`confound_diagnostics/feature_norm_summary.json`、`confound_diagnostics/*.jsonl`。
- 自动报告与图片：`README.md`、`analysis-report.md`、`stats-appendix.md`、`summary.md`、`figure-catalog.md`、`figures/`。以上自动报告应与第 5 节的口径修正共同引用。
- 运行日志与打包：`/Users/phyl/Documents/CIR-LID/server_reports/m1_review_bundle/logs`，以及其上级目录中的 `.tar.gz` 与 `.sha256`。此处没有修改这些原始产物。
