# R1 本地代码交付说明

日期：2026-10-02。

**状态：实现代码、配置、测试用例已经写入本地文件；未运行 Python、语法检查、测试、特征提取或实验。没有连接、上传或修改服务器。本文不是实验结果，也不表示代码已经通过执行验证。**

本次仅实现 R1：不使用目标配对真实帧作为评分参考时，训练真实图库上的语义关系异常能否定位 Face2Face 伪造。未修改 R0、未复用 R0/T0 缓存、未新增训练模型。既有 preflight 和服务器审计文档是历史记录；本次实现的具体选择以本文件和新配置为准。

## 1. 交付文件

| 文件（相对本项目根目录） | 内容 |
|---|---|
| `src/run_r1.py` | 独立入口、九阶段执行、断点续跑、低资源保护 |
| `configs/r1_non_oracle_semantic_relation_v1.yaml` | R1 默认条件、路径、50% 覆盖门槛 |
| `src/r1_non_oracle_semantic_relation/protocol.py` | 配置约束、源码/环境/输入摘要、阶段及样本完整性 |
| `src/r1_non_oracle_semantic_relation/dataset.py` | train-only 组件划分、图库/校准/评估角色、采样 |
| `src/r1_non_oracle_semantic_relation/preprocessing.py` | 每个输入分别裁剪、landmark、语义区域、DINOv3 特征 |
| `src/r1_non_oracle_semantic_relation/scoring.py` | 四方法、参考视频等权距离聚合 |
| `src/r1_non_oracle_semantic_relation/reference.py` | real-only 图库、clean-real 阈值、无 mask 查询评分 |
| `src/r1_non_oracle_semantic_relation/metrics.py` | 确定性 token→pixel、AP/F1/FPR、独立 crop 投影 |
| `src/r1_non_oracle_semantic_relation/evaluation.py` | 后置 mask 评估、完整视频 pooling、paired-real 次级诊断 |
| `src/r1_non_oracle_semantic_relation/reporting.py` | 全 seed 报告、组件 bootstrap、审计、结论及十二项回答 |
| `tests/test_r1_non_oracle_semantic_relation.py` | 评分数学、角色隔离、投影、断点完整性等测试用例（未执行） |

## 2. 已实现的数据流

```text
prepare → crops → geometry → features → gallery → calibrate → score → evaluate → finalize
```

- **prepare**：只读取指定 `train.json`；按真实视频配对图连通组件进行 R0 同算法 holdout，默认 30%、seed 20260930。图库、校准、评估组件互斥。探测所需视频元数据，不遍历其他 split。记录完整采样目录、缺失文件以及模型资产和官方 DINOv3 源码摘要。
- **crops**：每个 real/fake 输入独立使用 dlib HOG，upsample 1→2，最大脸、方形扩展 1.3、448 输入。fake 不读取 paired-real bbox。
- **geometry**：每个 crop 分别使用官方 MediaPipe Tasks；多脸/无脸/空语义区域记为失败，不借用真实帧几何，不用全脸替换失败区域。
- **features**：官方 DINOv3 ViT-B/16 block 6、`n=[5]`、norm=True、float32 `[784,768]`；不额外 L2 标准化磁盘特征。模型冻结、没有训练。
- **gallery**：每个 seed 预选 64 个 original train 视频，seeds 11/23/37；每视频 25/50/75% 三帧。c23/c40 都有三张不同有效帧才保留同一视频，使两个压缩条件实际图库 ID 相同；失败不替补。最少有效图库视频 32。
- **calibrate**：探索组件中独立预选 32 个真实校准视频（整个组件排除出图库）；最少有效完整视频 16。每方法、每 seed 在 c23 clean-real 全脸有效像素池计算 exact linear 99% quantile。严格 `score > threshold`；原值迁移到 c40，不用 fake/评估集拟合阈值。
- **score**：四方法同图库、同查询目录。先提交所有 score map，再允许 mask 访问。AP 不依赖阈值；校准失败不会把已经可计算的 AP 丢弃。
- **evaluate**：只有这里读取 Face2Face mask，严格同数字帧号且 mask/video 总帧数一致，不做时序 remapping。四方法主评价域都是 query 全脸 valid 像素；逐区域指标是次级结果。
- **finalize**：每 seed/压缩单独报告；主表只平均同视频全部已登记 seed，不把 seed 当独立样本。2,000 次连通组件 bootstrap，明确普通视频均值与组件等权差异是两个不同 estimand。

### 独立参考几何及聚合

R1 保留 R0 的 pairwise 数学核，但不复制 R0 的 oracle 数据流：

- relation：token 与本图左右脸颊上下文的两个 cosine 通道；query 与真实参考的通道 L2 距离，query 嘴/环以外归零。
- local_difference：单位 token 特征 L2 距离，全脸有效域评价。
- position_template：query 自身嘴/环二值模板，不偷用 target-real 区域。
- broken_context：R0 固定 seed 的 rejection-sampled 784-token derangement；同参考目录、同聚合。

参考和查询各用自己的几何。同一 normalized crop-grid token 索引对应，**没有声称 landmark 配准**。对每个真实参考视频，先平均其三帧的 pairwise 距离，再对视频等权平均；不是距均值 descriptor 的距离，不是 nearest/best-reference。

### paired-real 仅为后置诊断

独立 evaluation_real 按真实视频自身采样，供 clean-real FPR 使用。如 fake 的诊断帧号不同，新增 `paired_real` 行，避免覆盖独立真实采样或出现同 ID 不同帧号。它们从同一个 disjoint gallery 独立评分，不参与 fake 分数生成。

后置诊断将独立 real score 从 real crop 经过归一化原图坐标投影到 fake crop，只在共同有效域计算 fake 和 real-on-fake-mask AP。该差异与主 full-face AP 的分母不同，单独报告。

## 3. 覆盖及结论口径

- 默认 **50%**，具体分母是每 seed/压缩条件生成的 fake 查询帧；分子是四方法主 full-face AP 均定义的帧。
- paired-real 后置诊断的 available/defined 帧数另列，不拿它决定主评分 AP 是否有效。
- 主视频 pooling 要求三个不同采样帧全部成功。空/全阳性 mask 的单帧 AP 为 null，但完整视频 pooling **仍包含这些帧的全部有效像素**，不挑帧。
- 至少五个可评估独立组件，包括全 seed 完整病例主表的组件数保护。
- 50/60/70/80% sensitivity 只是相同数据的门槛敏感性，不重新选数据或重新计算另一套实验。
- `practical_margin_ap: 0.01` 表示 **1 AP 百分点**描述性参考线；不是统一最低改进门槛，也不是已预注册的等效/非劣检验。
- 自动结论保留 `PASS_CONTINUE` / `PARTIAL_PASS_DIAGNOSTIC` / `FAIL_ORACLE_DEPENDENT`。主要比较有支持但模板、上下文或压缩/seed 证据不足时，仍完整报告信号，标为部分通过而不是“没有研究价值”。PASS 的控制比较要求 CI 下界为正；不能把“未显著”写成“确定只是误差”。
- `FAIL_ORACLE_DEPENDENT` 只是试验分类标签，不足以证明 oracle 因果依赖。资源/依赖故障不生成科学失败结论；协议安全停止的入口状态为 `INVALID_PROTOCOL`，不覆盖既有 decision。

## 4. 在合适实验机器上的使用

**以下只是供后续实验机器使用的命令，本地和当前 0.5 CPU / 2 GiB / 无 GPU 服务器均未执行。**

准备好 Python 3.10+、NumPy 1.22+、PyYAML、兼容的 PyTorch/torchvision、OpenCV、dlib、MediaPipe Tasks、官方 DINOv3 仓库及权重、官方 face_landmarker.task。测试另需 pytest。没有自动安装依赖，也未验证版本组合。

配置内所有相对路径基于配置文件所在目录，而不是 shell 当前目录。应在第一次 prepare 前确认数据、train split、模型、资产和新输出路径。之后源码、配置、包版本固定；如需改变，使用新的 `r1_non_oracle_*` 输出目录，不复用旧 run。

在实验机器项目根目录执行：

```bash
# 首次正式运行前，在实验环境执行新测试用例
python -m pytest tests/test_r1_non_oracle_semantic_relation.py -q

# 全流程：资源检查先于 prepare；需要明确授权重负载
python -m src.run_r1 --config configs/r1_non_oracle_semantic_relation_v1.yaml --stage all --allow-heavy

# 中断后验证已完成阶段并续跑
python -m src.run_r1 --config configs/r1_non_oracle_semantic_relation_v1.yaml --stage all --resume --allow-heavy

# 也可以逐阶段执行，例如只生成 manifest，之后再单独提取特征
python -m src.run_r1 --config configs/r1_non_oracle_semantic_relation_v1.yaml --stage prepare
python -m src.run_r1 --config configs/r1_non_oracle_semantic_relation_v1.yaml --stage crops --allow-heavy
```

逐阶段顺序必须与数据流一致。除 `prepare` / `finalize` 外入口要求 `--allow-heavy`；`prepare` 也会探测视频和读取大权重文件算哈希，**不要在现有低配服务器执行**。全流程最低资源保护为 2 CPU 核、4 GiB 内存；这是拒绝低配任务的底线，不保证该配置足够快。features 默认要求 CUDA。GPU 显存需求未实测，不提供虚构用量/耗时。

### 磁盘和资源注意

单帧 float32 特征约 2.30 MiB，RGB crop 约 0.57 MiB；另有 reference descriptor、regions、分数和 mask。实际帧数受图库并集/数据可用性影响，因此总磁盘不可在未 prepare 时假定为固定值。reference 评分按单参考帧流式处理，不把所有 DINO 特征载入内存；代价是 repeated descriptor I/O。校准池使用磁盘 memmap；精确分位数会原地排序/partition **派生分数池**，不修改原图/特征/结果分数。完整性核对会重新读取上游缓存，属于实际 I/O 成本。

### 断点和完整性

- 样本记录写入 `stage/records/<sha256-id>.json`；已提交失败记录也保留，不在 resume 中隐式重试或替补。
- 每阶段结束写 `rows.jsonl`、`complete.json`；恢复时递归验证依赖产物的哈希，而非只信上游 receipt 文件。
- 校准样本 score 文件也纳入 checkpoint，防止中断后使用改动过的 score 配合旧阈值。
- 改变源码/配置/train split/包版本会拒绝继续；模型权重、MediaPipe asset 和 DINO 源码在使用前核对 prepare 哈希。
- 原始视频使用 size/mtime 指纹，**不是完整内容哈希，也不是防恶意篡改保证**。
- `active.lock` 只防同一 stage 并发写入。异常退出可能留锁，应检查无活跃进程后再人工处理；没有自动删除结果。
- prepare 尚未提交完成即被中断时，使用新的输出目录重做，不清理旧目录。项目没有 Git 元数据时 commit 记为 unavailable，不伪造版本号。

## 5. 输出

默认新目录：项目下 `results/r1_non_oracle_semantic_relation_v1/`。

```text
freeze_receipt.json
prepare/roles.json
prepare/probe_audit.json
<each stage>/records/*.json
<each stage>/rows.jsonl
<each stage>/complete.json
gallery/index.json
calibrate/thresholds.json
score/maps/*.npz
evaluate/per_query_metrics.jsonl
evaluate/per_video_metrics.jsonl
finalize/summary.json
finalize/summary.md
finalize/analysis_report.md
finalize/decision.json
finalize/quality_audit.json
finalize/protocol_audit.json
finalize/bootstrap_differences.json
finalize/condition_diagnostics.json
finalize/gallery_audit.json
finalize/semantic_region_summary.json
finalize/threshold_summary.json
finalize/all_seed_video_metrics.csv
finalize/source_group_metrics.csv
```

`analysis_report.md` 将逐项回答 oracle/source-disjoint、三项控制比较、c23/c40、seed 稳定性、空间/面积混杂、real/fake 几何失败、mask/test 访问与最终决策。图表排版不是本次依赖；目前输出表格/CSV/JSON/Markdown，不生成假示意实验图。

## 6. 本次验证边界

完成文件阅读、共享 API 签名静态核对、数据流审阅和旧文件 SHA-256 核对。旧 R0 runner、R0 模块、审阅基线覆盖的 shared 文件和 R0 配置全部与静态基线一致。

**未运行代码、未做 Python 编译/语法检查、未执行测试，因此不能声称测试通过或实际端到端成功。** 新代码应先在合适实验环境中执行测试/小范围验证，再正式运行。尚无 R1 AP、CI、decision 或 IEEE SPL 录用结论。
