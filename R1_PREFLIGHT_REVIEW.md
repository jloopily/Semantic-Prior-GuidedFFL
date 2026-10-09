# R1 Non-Oracle Semantic Relation Localization Pilot — Preflight Review

审阅日期：2026-10-02（Asia/Shanghai）  
状态：**PREFLIGHT_NEEDS_PROTOCOL_CONFIRMATION；尚未进入 R1 实现/冻结/运行阶段。**

本报告只依据本地文件静态阅读，不运行 Python、项目代码、测试、编译检查或实验；不连接服务器、不安装依赖。R0 结果数值来自用户背景陈述，本地未发现对应 R0 结果/冻结回执，因而本报告不独立认证该数值或其实际运行版本。

研究诚信：保留全部真实结果、异常、未定义指标和失败；可以改善图表排版，不修改数据、不隐藏负面结果、不更换条件追求通过。

## 1. 当前项目目录结构

工作区根目录：`/Users/phyl/Documents/CIR-LID`。

以下是审阅时已有的项目结构（忽略 `.DS_Store` 和历史报告包内部文件）：

```text
CIR-LID/
├── .workbuddy/{memory,skills}/
├── CIR-LID_v5.5_实验说明.md
├── CIR-LID_服务器现状与下一步_2026-09-25.md
├── R0_LOCAL_AUDIT.md
├── T0_本地适配与服务器运行说明.md
├── configs/
│   ├── frozen_v5_1.yaml
│   ├── t0_multiframe_v1.yaml
│   ├── m0_manifold_audit_v1.yaml
│   ├── m1_static_geometry_pilot_v1.yaml
│   ├── m1_1_c2_deconfounding_v1.yaml
│   ├── r0_semantic_relation_v1.yaml
│   ├── r0_semantic_relation_pairfix.yaml
│   ├── r0_semantic_relation_devicefix.yaml
│   ├── r0_semantic_relation_geometryfix_v2.yaml
│   └── r0_semantic_relation_geometryfix_v2_sensitivity50.yaml
├── src/
│   ├── run_r0.py
│   ├── r0_semantic_relation/
│   │   ├── __init__.py
│   │   ├── protocol.py
│   │   ├── geometry.py
│   │   ├── mediapipe_adapter.py
│   │   ├── scoring.py
│   │   ├── metrics.py
│   │   ├── bootstrap.py
│   │   └── README.md
│   ├── cirlid/
│   │   ├── config.py / io.py / types.py / manifest.py / ffpp.py
│   │   ├── crops.py / dinov3.py / features.py / gallery.py / knn.py
│   │   ├── geometry.py / scoring.py / heatmap.py / metrics.py
│   │   ├── thresholds.py / bootstrap.py / gates.py / synthetic.py
│   │   ├── t0.py / m0.py / m1.py / m1_1.py
│   │   └── __init__.py / cli.py / commands.py
│   └── 既有数据审计、缓存、建库、评分和其他实验入口
├── tests/
│   ├── test_r0_semantic_relation.py
│   ├── test_r0_repair.py
│   └── test_ffpp.py / test_geometry.py / test_manifest.py / test_t0.py
├── tools/                         # 服务器/下载/依赖工具；未读取凭据文件
├── experiment-record/             # T0、M0、M1、M1.1 历史记录
└── server_reports/                # T0、M0、M1、M1.1 历史报告包
```

本地未发现 `.git`、原始 FF++ 数据、官方 split JSON、DINOv3 权重、MediaPipe task 资产、R0 结果目录或 R0 feature cache。不能由此断言服务器也缺少这些内容。旧报告包不能代替 R0 实际 freeze receipt。

本次仅新增静态审阅材料：本文件和 `planning/r1_preflight/` 工作记录/源文件哈希；没有新增 R1 配置或实验代码。

## 2. R0 的精确评分器定义

代码证据：`/Users/phyl/Documents/CIR-LID/src/r0_semantic_relation/scoring.py:16–117`。

令每帧 token 特征为 `X ∈ R^(784×768)`，`u(x)=x/max(||x||₂, 10^-12)`。区域来自 **target-real landmarks**，同一组区域权重用于 fake 和 target-real。

左右面颊上下文：

```text
c_L(X;R) = Σ_i w_L,i X_i / Σ_i w_L,i
c_R(X;R) = Σ_i w_R,i X_i / Σ_i w_R,i
v_i(X;R) = [u(X_i)·u(c_L), u(X_i)·u(c_R)]
S(R) = {i: (mouth_fraction_i + ring_fraction_i)>0 且 valid_fraction_i>0}
```

上下文权重为像素覆盖比例，不是全脸均值或邻居距离。上下文总权重为零、非有限或上下文范数 ≤10^-12 时关系无效。零 query 特征有计数审计，不自动等同于上下文无效。

### 四种评分

| 方法 | 精确定义 | 空间支持 |
|---|---|---|
| relation | `||v_i(fake;R) − v_i(target-real;R)||₂` | S(R) 内有效；之外先置零 |
| local_difference | `||u(fake_i) − u(target-real_i)||₂` | 全脸 valid token；不限制在嘴/口周 |
| position_template | `1[(mouth_fraction_i + ring_fraction_i)>0]` | 返回全脸 valid token domain |
| broken_context | query 位置不变，分别以 `fake[π]` / `real[π]` 计算左右面颊 context，再比较两通道关系的 L2 差异 | 同 relation |

`π` 是 `/Users/phyl/Documents/CIR-LID/src/run_r0.py:360–367` 用 `np.random.default_rng(protocol.seed)` 反复生成、直到无固定点的 784-token 排列；现有 R0 配置 seed 是 20260930。fake 和 real 使用同一个 π，不是只打乱 real，也不是语义区域标签置乱。

**重要区别**：DINO 特征缓存没有额外 L2 normalize，但 scorer 内部确实使用 token/context 单位化计算 cosine。`backbone.norm: true` 是模型接口的归一化选项，不能混称为额外 token L2 normalize。

### 几何与评价

- `/Users/phyl/Documents/CIR-LID/src/r0_semantic_relation/geometry.py:115–150`：外唇有序多边形与面部轮廓；口周环是半径 12 的方形核膨胀，去掉嘴核心并裁到脸内；左右面颊为固定相对位置四边形，排除眼睛、嘴和环带。`valid=face`。眼睛不是 R0 的独立评分区域。
- 448×448 mask 按 16×16 patch 的覆盖比例映射到 28×28 token。
- `/Users/phyl/Documents/CIR-LID/src/r0_semantic_relation/metrics.py:48–61`：通常 bilinear、`align_corners=False` 放大到 448×448；缺少 torch 时存在 nearest-repeat fallback。服务器应固定实际插值分支，不把两个分支视为等价。
- `/Users/phyl/Documents/CIR-LID/src/run_r0.py:589–605`：四个 scorer 在**相同全脸 valid 像素域**评价；relation/template/broken 的零支持之外仍在 AP 分母中，双线性插值可软化边界。local 的 score 支持不同是原始对照的一部分，不应为匹配 token 数而偷偷改为 mouth-only。
- AP 对相同分数组合阈值计算。全零、全一或空 mask AP 为 `None`，不是 0。
- 视频 AP 是满足全部预设帧覆盖时，拼接其 valid 像素后算 AP，**不是逐帧 AP 的平均**。单帧空 mask 可在合法 pooled-video AP 中贡献负像素。
- bootstrap 2,000 次，95% percentile CI，实际调用 `group_key="component_id"`；估计量为**等权 source-connected-component 内视频差异均值的平均**，不是把 frame/token 当独立样本。报告的普通视频 mean AP 和 bootstrap 点估计加权不同，R1 需明确区分。
- R0 `statistics.score_threshold: none`；没有 R0 固定阈值 F1/FPR 的既有规则。
- R0 没有独立给 paired-real 生成 non-oracle score map；R0 relation(real,real) 退化为零，不能拿来冒充 R1 的 clean-real 对照。

## 3. R0 oracle reference 的具体来源

### 特征 oracle

`/Users/phyl/Documents/CIR-LID/src/cirlid/ffpp.py:93–107` 的当前命名规则：fake `A_B.mp4` 的 target 是 B，source 是 A。这里按项目现有代码解释，未重新核验服务器数据版本。

`/Users/phyl/Documents/CIR-LID/src/run_r0.py:179–192` 按 `(compression, target_id, frame_fraction)` 找 target-real，即：

```text
fake: original FF++ train 对应 manipulated_sequences/Face2Face/c23/videos/A_B.mp4
reference: original_sequences/youtube/c23/videos/B.mp4
frames: [0.25, 0.50, 0.75] 各自映射到实际帧号，并审计 strict_frame_pair
```

这是同一 target 视频的配对真实帧，不是 source A、任意 real 视频或真实图库。

### 几何 oracle（不能遗漏）

1. `/Users/phyl/Documents/CIR-LID/src/cirlid/ffpp.py:279–285,345–346` 把 fake 的 `crop_source_video` 指向 target-real c23。
2. `/Users/phyl/Documents/CIR-LID/src/cirlid/crops.py:186–220` 从该 target-real 解码图像并检测 dlib bbox，再映射/复用到 fake。
3. `/Users/phyl/Documents/CIR-LID/src/run_r0.py:529–558` 在 paired real 上建立主语义区域；fake landmark 主要用于诊断 IoU，不决定主评分几何。
4. 三个差分 scorer 使用配对真实特征；position_template 本身不使用 real features，但仍使用 real landmarks。

因此只替换差分中的 real features，**不能称为完全去除 R0 oracle**。

补充：R0 source-component holdout 来自 train，并不是 official validation/test。按相同 frame fraction / frame index 采样也不等于外部证实视频时间戳严格同步。

## 4. R1 real-only reference 的可行实现

### 可以保留不变的数学核

保留 DINOv3 block-6、448 输入、特征规范、嘴/环/面颊区域构造公式、两个 cosine-to-cheek 通道、L2 关系差异、local 单位 token L2、二值模板、固定 context derangement、AP/视频 pooling/component bootstrap。

### 必须声明而不能伪装成“完全原封不动”的部分

R0 没有图库接口、K、邻居过滤、跨视频 token 对应或多参考聚合。新增这些是 R1 reference-setting 的设计，并非已有冻结规则。且原 `relation_score(fake, real, regions)` 只接受一份共享几何，不能直接表达两帧各自独立的语义区域。

推荐审批候选方案（**未冻结、不实施**）：

1. **数据角色**：复用 R0 seed=20260930、fraction=0.30 的 train connected-component 划分；holdout 为 evaluation，exploration 的 original-only 视频再以固定组件划分为 gallery 和 clean-real calibration。必须先做角色划分再枚举/解码，不能先全 train 提特征再过滤。
2. **无配对几何**：fake、evaluation real、gallery real、calibration real 都从自己的图像独立裁剪和检测 landmarks。不读取 paired real 决定 fake bbox/landmarks，也不在 fake landmark 失败时用 paired real 补救。配对真实帧独立预处理只供最终 evaluation。
3. **descriptor**：分别用每张图自己的区域提取原始两通道 relation descriptor。按冻结规则比较同一 28×28 crop token 位置；这一选择保留绝对位置混杂，必须如实声明，不能声称已做 semantic alignment。
4. **参考聚合推荐**：保持原始 pairwise 距离核，先对同一 gallery 视频的有效采样帧平均距离，再对 gallery 视频等权平均；不做目标相关的最优 reference 选择。relation 和 local 使用同一候选 frame catalog；broken 使用同一 catalog 和 R0 derangement；模板仅使用 query 自身几何。此处“平均距离”不是“距平均 descriptor 的距离”，两者不等价。
5. **gallery 规模/seed 候选**：固定 64 real 视频、3 个 frame fractions，重采样 seeds `[11,23,37]`（与旧 M0 中的部分既有 seed 值一致，但不是 R0 的既有图库协议）；gallery/calibration 组件互斥，所有 gallery 同时与全部 evaluation source/target IDs 不相交。若预算或有效几何不足，记录失败，不事后降低规模或换 seed。规模稳健性不自动新增：R0 无此接口。
6. **压缩**：同一身份/帧比例 catalog 的 c23 gallery 评价 c23，c40 gallery 评价 c40；不能压缩池混用。c23 clean-real 阈值固定后转移到 c40；阈值迁移与 reference-compression 变化分别记录。
7. **阈值候选**：各 scorer/seed 的 gallery-disjoint clean calibration c23 全脸 pooled 像素 0.99 quantile，linear，`score>threshold`；不见 fake mask、不见 evaluation real 来拟合阈值。统一的是规则/校准行，不是强行给不同量纲 scorer 同一数值阈值。二值模板/大量同分可能产生退化预测，照实报告实际 FPR/F1，不针对模板特殊优化。
8. **real 对照**：evaluation real 从同一 source-disjoint gallery 独立评分；最终把 fake 的 GT mask 通过已记录 crop transform 投影到共同原始坐标，比较 fake 与 paired-real map。需冻结共同评价域/投影插值和无覆盖像素处理；主 fake AP 仍在 fake 自身全脸 valid 域，paired-real 比较属于独立 secondary 子集并报告分母。
9. **统计**：seed 不当作独立视频；各压缩先逐 seed 再 source-component 内汇总，seed-averaged 主估计不能使用“最好 seed”。跨压缩比较用两条件共同可评价 source/video catalog；每条件原始结果另报，不能只报告交集并隐藏缺失。
10. **scope**：仅 Face2Face，不能为了“不同伪造方法”指标额外读 Deepfakes/FaceSwap/NeuralTextures；多方法稳健性明确 `not_in_scope`。去绝对位置、region shuffle、eyes 新区域仅可列为后续 R1.1，不能混入冻结 R1 主方法。

如果用户希望 kNN，需另外预先确定 K、检索 descriptor、位置/语义候选过滤、同距离 tie-break、每视频帧数权重和重复邻居规则；不能静默采用旧 CIR-LID 的 k=20。

本地没有真实 split/crop/feature 文件，source-disjoint 的**工程可实现性成立**，但图库是否足够、身份映射是否可用、质量是否达标，只能服务器 prepare 审计确认。FF++ video ID disjoint 不自动等于跨视频人物 identity disjoint；若要求 person-level disjoint，需要可信 identity map，不能编造 identity 标签。

## 5. 潜在泄漏点与协议风险

| 风险 | 当前证据/防护要求 |
|---|---|
| 成对特征 oracle | R0 直接使用 target-real；R1 score 输入禁止 evaluation-real 参考路径 |
| 成对 bbox/landmark oracle | 旧 ffpp/crops 加载 paired target-real；R1 必须使用独立 preprocessing |
| 同 source/target 或反向 fake 泄漏 | gallery 同时排除两个视频 ID 及其 connected component；不能只排除 target |
| calibration self-match | calibration/gallery/evaluation 组件三方互斥，绝不能校准样本检索自己 |
| 假图库“real-only” | 原 enumerate_records 会列 fake/mask；gallery 应独立 original-only 枚举，不能先读 fake 再过滤 |
| fake mask 提前读取 | 原 ffpp 会 probe mask、crops 会缓存 mask，早于 evaluate；R1 应把 mask 解码/采样有效性判断移到独立 evaluation 阶段，score/catalog 不接受 mask |
| 图库读取仅声明不执行 | audit 必须由实际打开文件/输入角色日志与 manifest 导出，不只写布尔常量 |
| R0 cache/score map 混用 | R0 crop 和 region 带 oracle，R1 独立 namespace/cache，绑定原图/crop/model/config/role hash |
| source-disjoint vs identity-disjoint | 视频 ID/组件审计可做，人物身份无本地映射；不能混称 |
| seed 选择 | seed 和候选 catalog 冻结，逐 seed 全报；不以最佳 seed 下结论 |
| test/scope 泄漏 | 只能读批准的 train metadata；test 路径可做字符串 provenance，不能 open test JSON/视频；不通过 glob 扫全数据 |
| landmark 选择偏差 | 每角色/压缩记录 attempt/single-face/no-face/multi-face/empty region；不能将 fake 失败悄悄回退到 real |
| 共同 evaluation rows 不一致 | 四方法强制同一主 catalog/像素域；paired-real、阈值迁移等 secondary 子集另列分母 |
| 空/满 mask 与异常 | AP=null；generated、eligible、undefined、failed 分开；异常不是负面科学证据 |
| duplicate / zero-distance | manifest 样本和内容重复审计；无 kNN 时 duplicate-neighbor 应注明不适用，不能伪造通过；零距离计数不静默删除 |
| 非严格冻结 | R0 回执验配置/部分源代码/model/asset，但不哈希整个实际数据 manifest 和外部 DINO repo；R1 扩充实际输入与外部代码 provenance |
| interpolation fallback | torch 缺失时 R0 插值会变，R1 明确生产分支；不把环境变化看成相同方法 |
| R0 基准版本不明确 | 多份配置、本地无实际回执；不要默认 geometryfix 或 sensitivity50 是已报告主结果 |

特别提醒：`r0_semantic_relation_geometryfix_v2_sensitivity50.yaml` 的 `min_evaluable_pair_fraction=0.50`，而本地 `src/r0_semantic_relation/protocol.py` 固定要求 0.80。按当前代码会拒绝该配置；这不是已运行验证，也不应在 R1 中偷偷放宽。当前审阅不修复 R0。

## 6. 需要新增或修改的文件

协议确认后的建议新增清单：

```text
/Users/phyl/Documents/CIR-LID/configs/r1_non_oracle_semantic_relation_v1.yaml
/Users/phyl/Documents/CIR-LID/src/run_r1.py
/Users/phyl/Documents/CIR-LID/src/r1_non_oracle_semantic_relation/__init__.py
/Users/phyl/Documents/CIR-LID/src/r1_non_oracle_semantic_relation/protocol.py
/Users/phyl/Documents/CIR-LID/src/r1_non_oracle_semantic_relation/data.py
/Users/phyl/Documents/CIR-LID/src/r1_non_oracle_semantic_relation/preprocessing.py
/Users/phyl/Documents/CIR-LID/src/r1_non_oracle_semantic_relation/gallery.py
/Users/phyl/Documents/CIR-LID/src/r1_non_oracle_semantic_relation/scoring.py
/Users/phyl/Documents/CIR-LID/src/r1_non_oracle_semantic_relation/evaluation.py
/Users/phyl/Documents/CIR-LID/src/r1_non_oracle_semantic_relation/reporting.py
/Users/phyl/Documents/CIR-LID/src/r1_non_oracle_semantic_relation/README.md
/Users/phyl/Documents/CIR-LID/tests/test_r1_protocol.py
/Users/phyl/Documents/CIR-LID/tests/test_r1_non_oracle.py
/Users/phyl/Documents/CIR-LID/R1_SERVER_RUNBOOK.md
```

只新增，不修改 R0/旧配置/旧结果/共享冻结模块。几何、MediaPipe adapter、DINO、AP/bootstrap 可以只读 import；需 hash 这些实际依赖。独立重写 R1 数据角色枚举和 mask-free preprocessing 适配层，复用不带 oracle 的底层功能。必要时在 R1 中实现单独 descriptor/aggregation wrapper，并明确和原函数的差别。

单元测试计划：oracle 路径拒绝、两侧 ID/组件 overlap、gallery-only 无 mask/fake I/O、test sentinel、mask 不影响 score、独立几何、empty/zero/nonfinite、同分 AP、共同行、视频 pooling、component bootstrap、cross-compression pairing、seed averaging、冻结篡改拒绝。**不在本地运行测试。**

## 7. 预计输出文件

服务器真实准备/执行阶段写入：

```text
/Users/phyl/Documents/CIR-LID/results/r1_non_oracle_semantic_relation_v1/
  freeze_config.yaml
  freeze_receipt.json
  code_manifest.json
  data_manifest.json
  feature_config_audit.json
  mediapipe_asset_hash.json
  dinov3_weight_hash.json
  environment.json
  protocol.json
  gallery_manifest.jsonl
  calibration_manifest.jsonl
  evaluation_manifest.jsonl
  pairing_manifest.jsonl
  input_access_log.jsonl
  preprocess_audit.json
  score_index.jsonl
  reference_aggregation_audit.json
  per_query_metrics.jsonl
  per_video_metrics.jsonl
  aggregate_metrics.json
  bootstrap_differences.json
  condition_diagnostics.json
  gallery_audit.json
  semantic_region_summary.json
  threshold_summary.json
  analysis_report.md
  decision.json
  dry_run/                  # 单独运行标识/输出，不覆盖正式结果
```

服务器绝对路径以部署目录为准，上面是本地计划路径；YAML 路径相对配置文件解析。缓存分别在项目根 `data/crops/r1_…`、`data/masks/r1_…`、`features/r1_…`，不得覆盖 R0。

freeze_receipt 的有效 `frozen` 状态必须由服务器核验实际权重、资产、split、代码、manifest、环境后生成。本地不伪造 asset/hash、Git commit、环境、gallery ID 或结果。当前静态 source hash 文件不等于实验 freeze receipt。

report 至少区分：工程状态、协议有效性、统计支持、空间/面积混杂、是否进入 R2。没有运行时不得创建含 PASS/FAIL 科学判定的 decision.json；未定义 AP 是 null，不补 0。最后摘要/结果表只能在有真实结果后填写。

## 8. 是否可以在不修改 R0 的情况下实现 R1

**可以，不修改 R0 源文件或其结果。**独立 R1 package、runner/config/cache/report 可实现。

但**不能同时满足“评分函数及几何调用逐字不变”和“彻底无 oracle”**：原函数共享一份 paired-real 几何、只接收一对 frame，不具备图库聚合。合理的保留范围是原始关系/距离数学核不变；独立几何与图库聚合作为明确、预注册的 R1 reference-setting 变更。

当前 R0 protocol 校验明确要求 oracle=true、c23-only、train-only，不能直接拿 R1 配置调用 R0 runner。R1 不能仅改一个 oracle 布尔值就绕过校验。

## 9. 需要用户确认的协议歧义

### 必须先确认，否则不开始修改代码

1. **“不修改 R0 relation”的边界**：是否接受保留数学核，但每帧独立几何、source-disjoint gallery 和预注册 reference 聚合？推荐接受；否则与彻底去 oracle 逻辑冲突。
2. **reference 规则**：同位置 pairwise 距离的两级平均，还是 kNN/单参考？R0 无此定义。推荐上述等视频权平均，禁止结果驱动 selection；图像/token correspondence 和 candidate eligibility 必须一起冻结。
3. **evaluation/gallery/calibration split**：R0 train exploration→gallery/calibration、train holdout→evaluation，还是旧 train gallery→official validation evaluation？推荐前者保持 R0 pilot scope；这仍是已研究开发数据，不是全新盲测。
4. **paired-real secondary 比较**：独立 crop 后不再共享像素坐标。推荐共同原始坐标的记录式投影和共同可见域评价，不能直接比较两张 448 数组的同下标。该子集分母与主 fake AP 分开。

### 建议一次确认的冻结细节

5. 64 gallery 视频、frames `[.25,.50,.75]`、seeds `[11,23,37]`；不足时中止/标记无效，不另选 seed。calibration 组件划分的比例/seed 要补充；不得从 holdout real 拟合。
6. c23/c40 使用各自匹配压缩的 gallery；c23 阈值转移到 c40。若要固定 c23 gallery 跨条件，应明确作为另一协议而非执行时切换。
7. fixed-F1/FPR 采用 clean-calibration 99% quantile、linear、严格大于，模板退化也不优化；这是新增 R1 规则，不是已有 R0 阈值。
8. 用户文本中“relation 与 template 持平仍可 R2”与“持平进入 R1.1”有冲突。推荐保守规则：P1 正向且 P2/P3 未证实、c40/seed 不稳→`PARTIAL_PASS_DIAGNOSTIC`；只满足预注册全部通过条件才 `PASS_CONTINUE`。
9. “明显下降”/“可接受失败率”/“少数视频驱动”未量化。沿用 R0 最少 5 独立 holdout 组件、至少 80% 可评价、异常不超过 5% 作为候选质量门；fake/real single-face 成功率与不对称差异需新增冻结数值，不能看结果后设门。独立预处理质量不足应判协议/证据不足，不判 oracle-dependent。
10. `FAIL_ORACLE_DEPENDENT` 是用户要求的标签；仅 P1 不显著不证明 oracle 因果依赖，负面报告必须说“当前 non-oracle 能力未获支持”。要量化 R0→R1 下降，需 R0 实际 receipt/results 与匹配 rows，不能凭背景数字算 CI。
11. “identity”若仅指 FF++ source/target video IDs，按 video-connected-component 审计；若指人物身份，需额外可信映射。缺少映射不能声称 identity-disjoint。
12. Face2Face-only 与“不同伪造方法”/“去绝对位置结果”要求范围不一致。推荐后者明确 out-of-scope/not-implemented，保留至 R1.1，不自动增设实验。

### 审查结论

**Preflight 静态审阅已完成，但存在会改变科学含义的协议歧义，因此实现门尚未通过。**

工程上可实现独立 R1；现在不应生成冻结配置、运行代码或把真实图库/结果填成已验证。下一步是用户确认上述核心 reference/geometry/split/secondary 规则，再在本地只写代码，不本地运行。服务器 dry run 和正式实验另需明确授权并提供实际冻结资产/路径。

## 2026-10-02 服务器只读补充（更新，不覆盖前述静态来源说明）

用户随后授权探查服务器 `/root/autodl-tmp/cirlid55`。发现真实 R0 geometryfix_v2 结果和回执，已只读复制六份小型证据并校验哈希；对应源文件/本地版本一致。

新增关键事实：

- R0 decision 是 `INCONCLUSIVE_QUALITY_OR_INDEPENDENCE`，holdout frame 可评价 69.60%，未达冻结 80%。
- relation−local 为 +9.7685 AP 百分点；relation−template / broken 为 −0.2968 / −0.2380，后两项 CI 均在零以下，不能称统计持平。
- fake single-face 成功 1,103/2,160，real 为 2,153/2,160；独立 fake 几何是重要风险，但本次未重新检测。
- 旧 T0 64-video gallery 与 R0 holdout 重叠 18 个 video IDs，不能直接继承。
- train c23/c40 original/Face2Face/mask 路径存在性齐全；未解码。

详细范围、限制和判断见 `/Users/phyl/Documents/CIR-LID/R1_SERVER_READONLY_AUDIT_2026-10-02.md`。仍未启动 R1 实现/冻结/正式实验。
