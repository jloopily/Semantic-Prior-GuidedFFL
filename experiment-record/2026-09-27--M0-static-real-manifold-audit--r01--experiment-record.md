---
type: experiment-log
status: completed
experiment_id: M0-A-20260927-001
date: 2026-09-27
experiment_line: CIR-LID-static-real-manifold-audit
round: 1
purpose: structural-audit
stage: real-only-validation
provisional_category: B
source_artifacts:
  - /Users/phyl/Documents/CIR-LID/server_reports/m0_manifold_audit_v1
related_documents:
  - /Users/phyl/Documents/寻找方向/M0_静态真实流形结构审计_实验方案与AgentPrompt.md
  - /Users/phyl/Documents/CIR-LID/experiment-record/2026-09-27--T0-multiframe-static--r01--experiment-record.md
  - /Users/phyl/Documents/寻找方向/CIR-LID_GateB失败复盘与方向转向_v6.md
local_code:
  - /Users/phyl/Documents/CIR-LID/configs/m0_manifold_audit_v1.yaml
  - /Users/phyl/Documents/CIR-LID/src/cirlid/m0.py
  - /Users/phyl/Documents/CIR-LID/src/run_m0.py
server_result_namespace: results/m0_manifold_audit_v1
official_test_read: false
fake_data_read: false
original_synthetic_gate_b: FAIL
m1_unlock: false
---

# M0 静态真实流形结构审计实验记录

## 1. 实验摘要

M0-v1 是一次 **real-only 的静态特征空间结构审计**，用于回答：

> 在冻结 DINOv3 特征、真实 gallery、视频级划分和 kNN 局部几何计算的前提下，真实人脸特征是否呈现出明显区别于“保持范数不变的各向同性随机方向”null 的局部结构？

本实验不是 fake detection 性能实验，不是原 Synthetic Gate B 的补考，也不用于解锁 official test 或直接证明 CIR-LID 有效。

### 最终结论

实验完整完成，协议和 provenance 检查通过。真实特征相对于 matched-radial null 呈现出强烈的非各向同性局部结构；但 gallery 重采样稳定性在部分指标上只有中等水平，且 c23/c40 压缩变化会系统性改变若干几何量。因此结果暂定为：

> **Category B：存在可信的真实局部结构信号，但仍需进一步审查稳定性与混杂因素。**

这不等于 M0 失败，也不等于已经证明了严格数学意义上的真实流形，更不等于 CIR-LID 已恢复有效。

---

## 2. 前序状态与实验边界

原 CIR-LID Synthetic Gate B 已失败。本实验不改变该结论：

- 原 Synthetic Gate B：`FAIL`；
- M0 不读取 fake 数据、fake mask、fake manifest 或 fake score；
- M0 不读取 official test；
- M0 不修改 Gate B、T0 或 v5.5 的既有结果；
- M0 不自动解锁 M1 或 official test；
- M0 不包含动态、频域、3D 或学习式 fusion。

M0 的作用仅限于审计 real feature geometry，不能单独回答 fake 是否偏离真实结构，也不能单独证明 CIR-LID 对伪造定位有效。

---

## 3. 冻结协议

### 3.1 特征与距离

- Backbone：DINOv3 ViT-B/16，LVD-1689M；
- block：第 6 层，`api_n=[5]`；
- 特征网格：28×28；
- 特征维度：768；
- 距离：float32 Euclidean L2；
- FAISS：`IndexFlatL2` 精确搜索，取平方距离平方根；
- 未添加额外 L2 normalization；
- k 主值：20；
- 敏感性分析：k=10、20、40。

### 3.2 输入和裁剪

- 输入尺寸：448；
- 颜色：RGB；
- ImageNet mean/std；
- test-time augmentation：关闭；
- face crop：dlib HOG；
- detector source：`original_target_c23_only`；
- primary upsample：1；retry upsample：2；
- square scale：1.3；
- padding：reflection；
- image resize：bilinear antialias；
- mask resize：nearest；
- mask binarization：nonzero。

### 3.3 数据与视频划分

- Gallery：FF++ train real c23，中间帧，64 个视频；
- Validation：video-disjoint real，140 个 c23 视频以及相同来源的 140 个 c40 视频；
- 每个视频固定抽取 128 个空间 token；
- gallery leave-video-out 时排除 query 来源视频的全部 784 个 token；
- validation 视频不进入 gallery；
- validation query 使用 `T1_center=0.50` 中心帧。

### 3.4 M0 阶段

执行阶段为：

```text
prepare → audit → crops/features（仅在无法复用精确缓存时执行）
→ baseline → resampling → compression → null → finalize
```

本次 `audit` 成功复用了 T0 已核验的 280 个 validation center-frame feature artifacts，因此没有重复运行 crops/features。

---

## 4. 执行完整性审计

结果目录：

```text
/Users/phyl/Documents/CIR-LID/server_reports/m0_manifold_audit_v1
```

关键状态：

```json
{
  "status": "completed",
  "official_test_locked": true,
  "fake_manifest_images_masks_scores_read": false,
  "test_split_or_videos_read": false,
  "feature_config_mismatch": [],
  "null_control_skipped": false
}
```

三个条件下均未发现以下问题：

- zero-distance query：0；
- k=20 内 zero distance：0；
- duplicate neighbor query：0；
- degenerate LID query：0；
- degenerate PCA query：0；
- negative PCA eigenvalue query：0；
- same-video neighbor fraction：0。

`feature_config_audit.json` 状态为 `verified`，特征维度、backbone、block、网格、预处理和权重 provenance 均匹配冻结配置。

主要输出文件均存在并可解析，包括：

```text
summary.md
protocol.json
diagnostics.json
feature_config_audit.json
input_manifest.json
split_audit.json
baseline.provenance.json
resampling_summary.json
compression_stability.json
null_control_summary.json
per_query_geometry.jsonl
per_video_geometry.jsonl
null_per_query_geometry.jsonl
null_per_video_geometry.jsonl
neighbor_stability.jsonl
```

### 输出规模复核

- `per_query_geometry.jsonl`：132,096 行；
- `per_video_geometry.jsonl`：1,032 行；
- `null_per_query_geometry.jsonl`：40,960 行；
- `null_per_video_geometry.jsonl`：320 行；
- `neighbor_stability.jsonl`：17,920 行。

从运行、provenance 和产物完整性看，本次 M0 没有发现执行层面的异常。

---

## 5. Real baseline 结果

k=20，统计单位为“每个视频的 token-level median，再对视频取均值”，置信区间为 video-level bootstrap 95% CI。

| Condition | Metric | Estimate | 95% CI |
|---|---|---:|---|
| gallery c23 leave-video-out | local anisotropy | 0.22275 | [0.22026, 0.22529] |
| gallery c23 leave-video-out | participation ratio | 8.43820 | [8.34677, 8.52235] |
| gallery c23 leave-video-out | effective dim 90% | 10.72656 | [10.61719, 10.82813] |
| gallery c23 leave-video-out | effective dim 95% | 13.33594 | [13.21875, 13.45313] |
| gallery c23 leave-video-out | LID | 17.61790 | [17.27456, 17.95723] |
| gallery c23 leave-video-out | median radius | 27.36275 | [26.87662, 27.85015] |
| validation c23 | local anisotropy | 0.21937 | [0.21787, 0.22083] |
| validation c23 | participation ratio | 8.55929 | [8.50261, 8.61567] |
| validation c23 | effective dim 90% | 10.86071 | [10.80000, 10.91429] |
| validation c23 | effective dim 95% | 13.35357 | [13.27500, 13.43214] |
| validation c23 | LID | 17.47843 | [17.28762, 17.67472] |
| validation c23 | median radius | 26.59225 | [26.32114, 26.85114] |
| validation c40 | local anisotropy | 0.22305 | [0.22154, 0.22466] |
| validation c40 | participation ratio | 8.44139 | [8.38403, 8.49853] |
| validation c40 | effective dim 90% | 10.78571 | [10.71786, 10.85000] |
| validation c40 | effective dim 95% | 13.25000 | [13.17857, 13.32143] |
| validation c40 | LID | 18.79626 | [18.57830, 19.01660] |
| validation c40 | median radius | 27.57768 | [27.35676, 27.79486] |

这些数值说明当前真实 DINOv3 token 特征具有稳定的局部几何统计结构。但 `participation ratio` 或 `effective dimension` 不能直接解释为真实人脸流形的严格数学维度。

---

## 6. Matched-radial null control

### 6.1 Null 构造

Null 使用：

- 与原始 token 相同的特征范数；
- 相同的 token 数量；
- 相同的 784-token source-video grouping；
- 各向同性随机方向；
- 独立生成的 null query 和 null gallery；
- 相同的 leave-video-out 逻辑；
- 5 个 seeds：101、103、107、109、113。

因此该 null 主要检验：真实局部结构是否仅仅由 feature norm 和样本组织方式造成。

### 6.2 Real − Null 结果

| Metric | Real − mean Null | 95% CI |
|---|---:|---|
| local anisotropy | +0.15256 | [0.15020, 0.15504] |
| participation ratio | −9.99491 | [−10.08436, −9.90550] |
| effective dim 90% | −6.27344 | [−6.39063, −6.16406] |
| effective dim 95% | −4.66406 | [−4.78125, −4.53906] |
| LID | −117.74222 | [−118.54271, −116.93437] |
| median radius | −58.19193 | [−58.78614, −57.61827] |
| neighbor concentration | +0.08977 | [0.08164, 0.09883] |
| feature norm | −2.15×10⁻⁷ | [−7.39×10⁻⁷, 2.98×10⁻⁷] |

### 6.3 解释

最关键的现象是：

```text
feature norm difference ≈ 0
```

但 anisotropy、participation ratio、effective dimension、LID 和邻域半径均出现置信区间明显不跨 0 的差异。

因此可以支持一个保守结论：

> 真实特征中的局部几何结构不能仅由 token feature norm 差异解释；DINOv3 特征方向本身存在明显的非各向同性局部结构。

这不是严格流形证明，也不是 fake detection 证明。

另外，effective dimension 是离散统计量，部分 null replicate 的 `replicate_std=0` 主要反映该统计量的量化/离散性，不应被解释为 null 模型不存在随机波动。

---

## 7. Gallery resampling 稳定性

每次保留 80% gallery source videos，5 个 seeds：11、23、37、41、53。

总体邻居集合稳定性：

```text
neighbor token Jaccard       = 0.68877
source-video Jaccard         = 0.71260
```

video-level rank Spearman 范围：

| Metric | Video-level Spearman 范围 |
|---|---:|
| median radius | 0.9801–0.9889 |
| r_k | 0.9825–0.9893 |
| LID | 0.8353–0.9109 |
| participation ratio | 0.8912–0.9247 |
| unique neighbor videos | 0.7291–0.8224 |
| effective dim 90% | 0.6737–0.8038 |

### 解释

- 半径类指标排名最稳定；
- LID 和 participation ratio 稳定性较好；
- effective dim 90% 与 unique neighbor videos 的 video-level 稳定性明显弱一些；
- 因此不能把所有局部几何指标视为同等可靠。

这部分结果支持“存在可重复的结构”，但还不足以支持 A 类的强稳定性结论。

---

## 8. c23/c40 压缩敏感性

c23/c40 使用同一批 140 个 validation source videos 的 exact center frame 配对，配对方式为：

```text
source_video_and_exact_center_frame
```

c40 相对 c23 的 video-level median change：

| Metric | c40 − c23 | 约相对变化 |
|---|---:|---:|
| feature norm | +1.96065 | +2.65% |
| LID | +1.31782 | +7.67% |
| mean radius | +1.02406 | +4.00% |
| median radius | +0.98543 | +3.81% |
| r_k | +0.94957 | +3.51% |
| local anisotropy | +0.00368 | +1.71% |
| participation ratio | −0.11790 | −1.35% |
| effective dim 90% | −0.07500 | −0.69% |
| effective dim 95% | −0.10357 | −0.78% |

c23/c40 的视频级排名相关性为：

- `r_k`：0.8590；
- `median_r`：0.8464；
- `local_anisotropy`：0.7156；
- `participation_ratio`：0.7050；
- `LID`：0.6426；
- `effective_dim_90`：0.6089；
- `effective_dim_95`：0.4815。

### 解释

压缩没有完全破坏邻域排序，但会改变特征尺度和部分几何统计量，尤其是 LID 和邻域半径。因此后续如果混合使用 c23/c40，必须明确区分压缩条件变化与真实结构变化。

---

## 9. 主要混杂因素与限制

当前结果仍不能排除：

1. face crop 中心、尺度和边界 padding 的影响；
2. crop jitter；
3. 视频清晰度、模糊和压缩程度的影响；
4. 特征范数之外的图像质量因素；
5. 空间 token 所属解剖区域的影响；
6. 64 个 gallery 视频规模对邻居结构的影响；
7. 不同 k 和 gallery 规模下的外推稳定性。

报告中的 feature norm correlation 只是描述性检查，不能证明不存在 norm、crop/blur 或 anatomical-region confound。

M0 也没有建立可靠的跨帧 token correspondence，因此：

- 不建议依据 M0 直接开展动态 token-level 统计；
- M0 不判断 temporal mean/std/MAD 是否有效；
- M0 不改变 T0-B 未执行的事实；
- M0 不构成对动态方向的正面或负面证明。

---

## 10. 证据等级审计

### 本实验支持的结论

1. M0-v1 执行完整，输出可复核；
2. provenance、split 和 feature configuration 检查通过；
3. fake 数据和 official test 均未读取；
4. 真实特征具有明显的非各向同性局部结构；
5. 该结构不能仅由 feature norm 解释；
6. 主要半径、LID 和 participation ratio 对 gallery resampling 具有一定稳定性；
7. c23/c40 压缩会系统性改变若干局部几何量。

### 本实验不支持的结论

1. 真实特征已被严格证明为低维数学流形；
2. fake 一定偏离该结构；
3. CIR-LID 已经优于 C2 或原版 GLID；
4. 原 Synthetic Gate B 已被修复或重新通过；
5. official test 可以解锁；
6. M1 real-vs-fake 已经自动获得执行许可；
7. 动态、频域或 3D 扩展已经被验证有效。

---

## 11. 研究决策记录

### 当前决定

- M0 结果归档为 **Category B（provisional）**；
- 不重复运行 M0 以追求更好的数值；
- 不把本结果写成“严格真实流形证明”；
- 不恢复 CIR-LID 为已验证的主方法；
- 不解锁 official test；
- 暂不直接进入大规模 M1 real-vs-fake 结论；
- 保留完整 raw outputs、bootstrap 结果、null 结果和 provenance 文件。

### 建议的最小后续工作

如果继续推进，应优先进行一次针对混杂因素的定向诊断，而不是立即增加新模型或新 fusion：

1. crop jitter / face crop 稳定性；
2. blur 与压缩强度敏感性；
3. 不同空间 token 区域的敏感性；
4. feature norm 校正前后的结果；
5. k=10/20/40 以及 gallery 规模变化的一致性。

如果这些审查后结构信号仍然存在，再考虑一个小规模、real-only、video-disjoint 的 M1 pilot。即使 M1 pilot 成功，也不能自动恢复原 Gate B 或解锁原 official test。

---

## 12. 结果文件索引

本记录对应的完整服务器结果目录：

```text
/Users/phyl/Documents/CIR-LID/server_reports/m0_manifold_audit_v1
```

重点文件：

```text
/Users/phyl/Documents/CIR-LID/server_reports/m0_manifold_audit_v1/summary.md
/Users/phyl/Documents/CIR-LID/server_reports/m0_manifold_audit_v1/diagnostics.json
/Users/phyl/Documents/CIR-LID/server_reports/m0_manifold_audit_v1/protocol.json
/Users/phyl/Documents/CIR-LID/server_reports/m0_manifold_audit_v1/feature_config_audit.json
/Users/phyl/Documents/CIR-LID/server_reports/m0_manifold_audit_v1/resampling_summary.json
/Users/phyl/Documents/CIR-LID/server_reports/m0_manifold_audit_v1/compression_stability.json
/Users/phyl/Documents/CIR-LID/server_reports/m0_manifold_audit_v1/null_control_summary.json
```

## 13. 一句话归档结论

> M0-v1 在严格 real-only、video-disjoint 和 matched-radial null 协议下，发现了不能由 feature norm 单独解释的真实局部几何结构；但 gallery 稳定性和压缩/裁剪混杂仍未完全解决，因此当前只能归档为 Category B，不能据此宣称严格流形、CIR-LID 成功或解锁 official test。
