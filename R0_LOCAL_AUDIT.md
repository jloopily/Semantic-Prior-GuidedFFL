# R0 本地盘点与修订说明

日期：2026-09-30

## 可复用接口

- `src/cirlid/ffpp.py`：官方 FF++ split 解析、Face2Face 伪造视频／mask 路径、target-real 采样记录和 mask 帧索引映射。
- `src/cirlid/crops.py`：只在 target-real c23 帧上检测 dlib HOG 人脸、448×448 reflection crop、nearest mask crop；R0 使用独立缓存目录。
- `src/cirlid/dinov3.py` 与 `src/cirlid/features.py`：官方 DINOv3 ViT-B/16、block 6、`reshape=False`、28×28×768 输出及带 provenance 的特征缓存。
- `src/cirlid.io`、`manifest.py`：JSON/JSONL、哈希和清单写入。

R0 没有调用旧的 gallery、kNN、CIR-LID、C2、T0、M0、M1 或 M1.1 评分器；关系评分、语义几何、R0 bootstrap 和汇总均位于 `src/r0_semantic_relation/` 与 `src/run_r0.py`。

## 本轮修订

- 关系评分改为 fake 自己的嘴部 token／面颊 context 与 target-real 自己的嘴部 token／面颊 context 分别构造关系，再计算关系变化；不再把 real cheek context 同时用于 fake 和 real。
- broken-context 对照只置乱 cheek context，保持 fake/real 嘴部 query token 的空间位置不变。
- Pixel AP 对唯一分数阈值成组计算，消除同分像素的输入顺序依赖；全零和全一 mask 仍报告 AP 未定义。
- bootstrap 改为以训练 split 的 source-video connected component 有放回重采样；反向视频共享规范化的无向 target/source group，组件级重采样进一步避免共享视频身份造成伪独立，并保存 group size、group key 与 estimand。
- prepare 增加 source-component exploration/holdout 划分，保存 freeze receipt；后续每阶段核对配置哈希，禁止 prepare 覆盖已冻结 manifest。finalize 只用 holdout 做正式 decision，同时保留两类分区报告。
- pairing manifest 明确记录 fake/real/mask/crop-source 实际帧号、匹配布尔值和 `strict_frame_pair`；不匹配记录保留并在 evaluate 中不可评价。
- evaluate 写出异常类型、帧配对失败、crop 几何不一致、真假关键点状态、空/满有效 mask、mask 覆盖上限和异常比例；所有 scorer 使用相同的全脸有效像素域，意外异常超过冻结阈值时中止，而不是伪装成普通不可评价样本。
- finalize 分开审计全局及 holdout 的质量；预先声明最少 5 个独立 holdout 组件、严格配对和单脸成功率至少 90%、生成配对的可评价比例至少 80%、意外异常不超过 5%。样本独立性或质量不足给出 inconclusive，不据此作负面的科学结论。帧索引与裁剪坐标相同仍不等于对独立编码视频的时间戳同步已经获得外部证明。
- 测试文件补上生产 `build_semantic_regions()` 调用、AP tie regression、零区域/零向量、真假面颊变化控制、严格帧配对、关系与模板差异、上下文置乱控制和组件级 bootstrap 检查。

## 新增隔离内容

- `src/r0_semantic_relation/geometry.py`：官方嘴唇外轮廓有序多边形、固定 12 像素口周环带、左右面颊确定性区域、448→28 覆盖映射。
- `src/r0_semantic_relation/mediapipe_adapter.py`：MediaPipe Tasks Face Landmarker IMAGE 模式适配；只在 target-real 上生成主区域，另记录 fake-only 稳定性诊断。
- `src/r0_semantic_relation/scoring.py`：真假各自 context 的关系变化、局部特征差异、位置模板、仅破坏 context 的对照。
- `src/r0_semantic_relation/metrics.py`、`bootstrap.py`：全脸有效域 Pixel AP、AP−mask prevalence、每视频聚合、source-video-component paired bootstrap。
- `configs/r0_semantic_relation_v1.yaml`、`src/run_r0.py`、`tests/test_r0_semantic_relation.py`、`src/r0_semantic_relation/README.md`。

## 本地检查与限制

本机当前没有可运行本实验的 NumPy、OpenCV、PyTorch、PyYAML、MediaPipe、pytest 环境，也没有 FF++ 原始视频、DINOv3 repo/权重或 Face Landmarker `.task` 资产。因此没有运行真实数据实验，也没有读取 official test。已完成修改文件的 AST/`py_compile` 静态语法检查；pytest 数值回归因本机缺少 `pytest` 与 `numpy` 未执行。不能把这次本地检查表述为真实数据或完整测试通过。

服务器首次运行前必须把官方 `face_landmarker.task` 放到配置的
`assets/mediapipe/` 路径，并把其 SHA-256 写入
`mediapipe.asset_sha256`，不能保留 `REQUIRED_ON_SERVER`。R0 的 `prepare`
和 `evaluate` 会校验这个哈希。

## 纪律

- 只打开 `train.json` 并只枚举 Face2Face/c23；validation/test 路径仅保留 provenance，不读取。
- 不读取或解锁 FF++ official test。
- 不用 mask 设计、调整或选择区域；mask 只用于有效性审计和最终评价。
- 结果明确标记为 paired target-real oracle，不是可部署检测器。
- R0 不自动更改旧实验文件，也不自动启动远程实验；本次不上传任何代码。
