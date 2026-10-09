# CIR-LID v5.5 实验说明
## 从环境配置、数据获取、模型下载到 Gate 实验与完整评估

> 编写日期：2026-09-19  
> 配套方案：`首推方案_CIR-LID_SPL最小论文版_v5.5.md`  
> 目标：以 IEEE Signal Processing Letters（SPL）所需的最小但充分证据，验证 CIR-LID 是否值得进入完整主实验。  
> 协议状态：**预注册/冻结**。在看到 fake validation pilot 或 test 结果后，不再修改主设置、主终点和 Gate 阈值。

---

## 0. 先看结论：正确执行顺序

本项目不是“环境装好后立刻跑完整 FF++ test”。正确顺序是：

1. 获取 FF++ 与 DINOv3 权限；
2. 固化软件、代码、权重和数据清单；
3. 通过数值单元测试和 1–2 个视频的 smoke test；
4. 构建固定 c23 real gallery 和 real-only threshold；
5. 先运行 Gate A-α、Gate A-β；
6. 运行合成 anomaly × nuisance 与 validation fake 上的 c23→c40 Gate B pilot；
7. **仅当 Gate A/B 全部通过时**，解锁官方 test split，运行 Gate D；
8. Gate D 通过后才撰写以 CIR-LID 为主方法的 SPL 正文。

停止规则：

- Gate A 未通过：停止“同图 LID reference pool 失效”的主叙事；
- Gate B 未通过：停止以 CIR-LID/对数聚合作为主贡献，可转向更一般的 normalized multi-neighbor scoring；
- Gate D 未通过：停止应用型 SPL 投稿，不用定性热图替代定量结果。

---

## 1. 单一事实源：必须与 v5.1 冻结表一致

实现时以配套 v5.5 文档第 6 节“v5.1 最终冻结表”为唯一协议源。以下是运行所需的核心配置摘要：

```yaml
experiment:
  name: cirlid_v5_5
  seed: 20260919

input:
  size: 448
  color: RGB
  imagenet_mean: [0.485, 0.456, 0.406]
  imagenet_std: [0.229, 0.224, 0.225]
  test_time_augmentation: false

face_crop:
  detector: dlib_hog
  detector_source: original_target_c23_only
  upsample_primary: 1
  upsample_retry: 2
  square_scale: 1.3
  padding: reflection
  image_resize: bilinear_antialias
  mask_resize: nearest

backbone:
  implementation: official_facebookresearch_dinov3
  architecture: dinov3_vitb16
  weights: LVD-1689M
  frozen: true
  block_one_based: 6
  api_n: [5]
  norm: true
  reshape: false
  extra_l2_normalization: false
  feature_dtype_on_disk: float32

features:
  expected_patch_grid: [28, 28]
  expected_tokens_per_frame: 784
  expected_dimension: 768

gallery:
  split: train
  source: original_sequences/youtube/c23
  videos: 64
  frame_per_video: middle
  tokens_per_frame: 784
  expected_tokens: 50176
  include_all_patch_tokens: true

knn:
  engine: faiss.IndexFlatL2
  exact_search: true
  k: 20
  faiss_output_is_squared_l2: true
  take_square_root_before_scoring: true
  distance_dtype: float32

scores:
  distance_floor: 1.0e-12
  cir_denominator_floor: 1.0e-6
  higher_means_more_suspicious: true
  primary: [SP_LID, CIR_LID, arithmetic_multi_neighbor]
  secondary: [r1, mean_knn, r1_over_rk, LOF, local_PCA_PR]

sampling:
  validation_real_positions: [0.25, 0.50, 0.75]
  validation_fake_positions: [0.25, 0.50, 0.75]
  test_positions: [0.25, 0.50, 0.75]

threshold:
  source: validation_real_c23_pooled_pixels
  quantile: 0.99
  numpy_method: linear
  method_specific: true
  target_clean_pixel_fpr: 0.01
  frozen_for: [validation_fake, test_real, test_fake, c40]

statistics:
  unit: video
  bootstrap: manipulation_stratified_video_cluster
  bootstrap_replicates: 2000
  confidence_level: 0.95

sensitivity_only:
  k: [10, 20, 40]
  layer_one_based: [3, 6, 9, 12]
  gallery_frames: [16, 32, 64, 128]
```

若代码、配置文件和本文冲突，先停止运行并修正冲突；不得默默采用代码默认值。

---

## 2. 硬件、系统与存储建议

### 2.1 推荐环境

最终实验建议在 Linux + NVIDIA GPU 上完成：

- 操作系统：Ubuntu 22.04/24.04 或等价 Linux；
- Python：3.11；
- GPU：24 GB 显存优先；12–16 GB 可用，但特征提取 batch size 需降至 1–2；
- RAM：64 GB 推荐，32 GB 可运行但应使用 memory map/分块；
- 磁盘：至少预留 100–150 GB；
- CPU：建议 16 核以上，视频解码、裁剪和 CPU FAISS 会明显受益。

FF++ 官方说明中，全量 c23 压缩视频约 10 GB，c40 约 2 GB；但 mask、缓存裁剪、float32 token、四层 sensitivity 特征和结果副本会显著增加占用。不要下载全量 PNG 帧；只解码清单中需要的帧。

### 2.2 macOS 的用途

Apple Silicon/macOS 可用于：

- 编写和测试 manifest；
- CPU/MPS smoke test；
- 验证 `[B, 784, 768]` 输出；
- 小规模单元测试。

最终计时和完整结果不要混用 macOS/MPS 与 Linux/CUDA 产物。若设备不同，分别记录环境，并从 checkpoint、manifest 和配置重新运行。

### 2.3 FAISS 选择

冻结的算法是精确 `IndexFlatL2`，不是 IVF/PQ/HNSW 近似搜索。为减少环境冲突，建议先用 `faiss-cpu` 完成正确性验证；若速度不足，再在独立、经过验证的 CUDA 环境中切换 GPU FAISS。无论 CPU/GPU，必须：

- 使用 `IndexFlatL2`；
- 保持 float32；
- 验证前 100 个 query 的邻居索引和距离与 CPU 版一致；
- 对 FAISS 返回值开平方后再计算比值。

---

## 3. 项目目录

建议新建独立实验目录，原始数据只读：

```text
cirlid55/
├── README.md
├── configs/
│   └── frozen_v5_1.yaml
├── env/
│   ├── pip-freeze.txt
│   ├── system.txt
│   ├── dinov3_commit.txt
│   └── checksums.txt
├── third_party/
│   ├── dinov3/
│   └── FaceForensics/
├── weights/
│   └── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
├── data/
│   ├── raw/ffpp/                  # 原始下载，只读
│   ├── manifests/
│   ├── crops/
│   ├── masks/
│   └── nuisance_banks/
├── features/
│   ├── gallery/
│   ├── validation/
│   ├── pilot/
│   └── test/
├── indices/
├── results/
│   ├── smoke/
│   ├── gate_a_alpha/
│   ├── gate_a_beta/
│   ├── synthetic/
│   ├── gate_b_pilot/
│   └── gate_d_test/
├── src/
└── tests/
```

原始 FF++ 视频不得被裁剪脚本覆盖。每个派生产物都必须能追溯到：视频路径、帧号、crop box、代码 commit、权重哈希和配置哈希。

---

## 4. 环境配置

### 4.1 建立 Conda 环境

以下是建议的冻结参考环境。PyTorch/CUDA 命令仍应先用 PyTorch 官方安装选择器确认其与本机驱动兼容。

```bash
conda create -n cirlid55 python=3.11 -y
conda activate cirlid55
python -m pip install --upgrade pip setuptools wheel
```

Linux + NVIDIA 的参考组合：

```bash
pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

无 NVIDIA CUDA 的 smoke-test 环境：

```bash
pip install torch==2.7.1 torchvision==0.22.1
```

其余依赖：

```bash
pip install \
  numpy==1.26.4 \
  scipy==1.14.1 \
  scikit-learn==1.5.2 \
  pandas==2.2.3 \
  opencv-python-headless==4.10.0.84 \
  pillow==11.0.0 \
  pyyaml==6.0.2 \
  tqdm==4.67.1 \
  matplotlib==3.9.2 \
  seaborn==0.13.2 \
  pytest==8.3.4 \
  faiss-cpu==1.11.0

conda install -c conda-forge ffmpeg dlib -y
```

说明：

- 上述版本是项目参考冻结，不代表对所有驱动都唯一可用；
- 若 PyTorch 官方安装选择器要求不同 CUDA wheel，只改 PyTorch/torchvision/CUDA 组合，并保存实际环境；
- 不要同时安装 `opencv-python` 与 `opencv-python-headless`；
- 不要为加载 DINOv3 而另装 Transformers 实现，本项目冻结为官方 DINOv3 repository + 官方 checkpoint。

### 4.2 环境自检

```bash
python - <<'PY'
import platform
import numpy as np
import torch, torchvision
import cv2, faiss
print("python:", platform.python_version())
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("numpy:", np.__version__)
print("opencv:", cv2.__version__)
print("faiss:", faiss.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("cuda runtime:", torch.version.cuda)
PY
```

Linux 上再记录：

```bash
nvidia-smi
ffmpeg -version | head
pip freeze > env/pip-freeze.txt
uname -a > env/system.txt
```

在最终结果中不得只写“PyTorch + CUDA”，必须保存精确版本、GPU 型号和驱动版本。

---

## 5. 获取 FaceForensics++

### 5.1 申请权限

FF++ 不是无需授权的公开直链数据集。按官方仓库说明：

1. 阅读并接受 FaceForensics Terms of Use；
2. 填写官方申请表；
3. 获批后取得官方 download script；
4. 不把下载脚本、临时 URL 或数据转发给未获授权者。

官方入口：

- https://github.com/ondyari/FaceForensics
- https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md

在等待权限期间，可以完成环境、DINOv3 模型、合成实验与纯数值单元测试。

### 5.2 取得官方 split

```bash
git clone https://github.com/ondyari/FaceForensics.git third_party/FaceForensics
git -C third_party/FaceForensics rev-parse HEAD > env/faceforensics_commit.txt
```

复制或只读引用官方 `train.json`、`val.json`、`test.json`。官方划分为 720 train、140 validation、140 test original videos。不要重新随机划分身份。

### 5.3 下载范围

本项目仅需要：

- original c23、c40 videos；
- Deepfakes、Face2Face、FaceSwap、NeuralTextures 的 c23、c40 videos；
- 四种 manipulation 的 masks；
- 官方 split JSON。

不需要：

- c0/raw 全量视频；
- 全量 PNG frames；
- DeepFakeDetection、FaceShifter；
- 官方生成模型；
- 训练分割模型所需的额外标注。

拿到官方脚本后先执行：

```bash
python download-FaceForensics.py -h
```

然后按脚本实际文件名和服务器参数运行。推荐显式逐项下载，避免 `-d all` 把本项目不需要的扩展数据也下载下来：

```bash
FFPP=/absolute/path/to/cirlid55/data/raw/ffpp

python download-FaceForensics.py "$FFPP" -d original -c c23 -t videos
python download-FaceForensics.py "$FFPP" -d original -c c40 -t videos

for METHOD in Deepfakes Face2Face FaceSwap NeuralTextures; do
  python download-FaceForensics.py "$FFPP" -d "$METHOD" -c c23 -t videos
  python download-FaceForensics.py "$FFPP" -d "$METHOD" -c c40 -t videos
  python download-FaceForensics.py "$FFPP" -d "$METHOD" -t masks
done
```

若官方脚本当前要求 `--server EU`/`EU2`，按 `-h` 输出追加；不要修改脚本以绕过许可确认。

### 5.4 完整性检查

至少检查：

- 四个 manipulation 文件夹均存在；
- c23/c40 的同名视频能配对；
- mask 与 fake 视频名称可配对；
- 视频可由 OpenCV/FFmpeg 解码；
- `CAP_PROP_FRAME_COUNT` 不为 0；
- split 中的 original ID 能映射到实际视频；
- 每种 manipulation 的 `<target>_<source>.mp4` 与反向名称在存在时均被枚举。

对每个原始文件保存相对路径、字节数；不要求对所有大视频计算 SHA-256，但必须对 split、manifest、权重、配置和代码 commit 计算哈希。

---

## 6. 下载并固定 DINOv3

### 6.1 官方仓库

```bash
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
git -C third_party/dinov3 rev-parse HEAD > env/dinov3_commit.txt
```

不要在实验中途执行 `git pull`。如需升级，建立新环境和新结果目录，不覆盖冻结结果。

### 6.2 权重

申请/下载官方 **DINOv3 ViT-B/16 LVD-1689M** checkpoint。官方 hub 定义对应：

- architecture：`dinov3_vitb16`；
- embedding dimension：768；
- depth：12；
- patch size：16；
- checkpoint hash 后缀：`73cec8be`。

可以从官方 DINOv3 权重入口或官方 Hugging Face 模型页获得，但只把它当作同一官方 checkpoint 的下载渠道；**模型执行仍使用官方 DINOv3 repository，不混用 Transformers 的 token 定义或 hidden-state 索引。**

官方 Hugging Face 页面：

- https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m

权重可能需要登录和接受授权。下载后保存哈希：

```bash
sha256sum weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  >> env/checksums.txt
```

macOS 使用：

```bash
shasum -a 256 weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  >> env/checksums.txt
```

### 6.3 加载方式

```python
from pathlib import Path
import torch

repo = Path("third_party/dinov3").resolve()
weights = Path("weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth").resolve()

model = torch.hub.load(
    str(repo),
    "dinov3_vitb16",
    source="local",
    weights=str(weights),
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
```

### 6.4 输入变换

LVD-1689M 使用官方 README 给出的标准 ImageNet normalization：

```python
import torch
from torchvision.transforms import v2

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((448, 448), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])
```

注意：这里的 `Resize(448,448)` 应作用于已经按协议裁出的方形人脸 crop，而不是直接把完整视频帧拉伸为正方形。

### 6.5 第 6 block token 提取

```python
with torch.inference_mode():
    tokens = model.get_intermediate_layers(
        images,
        n=[5],          # zero-based API index = 第 6 block（one-based）
        reshape=False,
        norm=True,
    )[0]

tokens = tokens.float().cpu()   # 落盘 float32
assert tokens.ndim == 3
assert tokens.shape[1:] == (784, 768)
```

只使用返回的 patch token。不要把 CLS 或 4 个 storage/register token 手工拼回；不要再做 L2 normalization。

### 6.6 首次模型 smoke test

在任何 FF++ 批处理前，使用一张普通 RGB 图片完成：

1. 输出 shape 为 `[1, 784, 768]`；
2. 所有值有限；
3. 两次前向的最大绝对误差在当前设备上接近 0；
4. 保存/重载 float32 `.npy` 后完全一致；
5. 448/16=28，token 按 row-major 恢复为 28×28 时空间顺序正确。

若 shape 不是 `[B,784,768]`，停止，不要通过截断 token “修复”。优先检查输入尺寸、API 索引和是否使用了另一个实现。

---

## 7. manifest：先冻结样本，再看结果

### 7.1 manifest 必需字段

建议保存为 Parquet + CSV 摘要，至少包含：

```text
sample_id
split
compression
manipulation
is_real
target_id
source_id
video_name
video_path
mask_path
frame_count
frame_fraction
frame_index
mask_frame_index
crop_source_video
crop_source_frame
crop_left
crop_top
crop_right
crop_bottom
crop_scale
exclusion_reason
seed
```

额外保存：

- manifest SHA-256；
- 生成脚本 commit；
- 原始 split JSON SHA-256；
- gallery 64 个 real IDs；
- Gate B validation fake 子集；
- Gate β reference/query 子集。

### 7.2 帧号规则

全部使用 0-based frame index。对总帧数 `N`：

- gallery 中间帧：`floor((N - 1) / 2)`；
- 25/50/75%：`floor(p * (N - 1) + 0.5)`，`p∈{0.25,0.50,0.75}`。

不要使用 Python 内置 `round`，避免 `.5` 时的 bankers rounding。c23 与 c40 同名视频使用相同的归一化时间位置，再分别按各自 `N` 计算帧号。

mask 原则上应与 fake 视频帧数一致。若不一致：

- 使用 `floor(i*(N_mask-1)/(N_video-1)+0.5)` 做归一化位置映射；
- 在 manifest 标注 `mask_frame_remapped=true`；
- 报告发生数量；
- 若视频或 mask 无法可靠解码，则按预注册规则排除该 paired sample。

### 7.3 gallery 抽样

```python
import numpy as np

ids = sorted(train_real_ids)
rng = np.random.default_rng(20260919)
selected = sorted(rng.choice(ids, size=64, replace=False).tolist())
```

把 `selected` 写入不可变文件。不要因某个 gallery 结果“不好看”而重新抽样。仅当视频损坏/无法解码时：

1. 记录排除原因；
2. 从同一个初始随机排列中取下一个未使用 ID；
3. 不重新调用随机数生成器；
4. 保留原始与替补清单。

### 7.4 validation fake 的预先拆分

为保证 test split 在 Gate D 前不被查看：

1. 收集 validation split 中四种 manipulation 共同存在的 directed fake video names；
2. 排序后用 `default_rng(20260919)` 生成一次排列；
3. 前 64 个 directed names 作为 **Gate B c23→c40 pilot query**；
4. 剩余 names 作为 **Gate A-β fake-reference bank**；
5. 四种 manipulation 使用相同 directed names；
6. 两部分视频名称严格不重叠。

Gate A-β 的 query 使用 Gate B pilot 集；污染 token 来自剩余 validation fake videos，因此 reference/query 视频不重叠。官方 test 始终封存到 Gate A/B 通过。

---

## 8. 人脸检测、裁剪和 mask

### 8.1 检测源

每个目标 original 视频和时间位置：

1. **只在 original target c23 frame 上做人脸检测**；
2. dlib HOG detector 首先使用 `upsample_num_times=1`；
3. 若失败，用同一 detector、`upsample_num_times=2` 重试；
4. 多脸时选面积最大框；
5. 若仍失败，排除该时间位置并记录；不切换到另一个 detector；
6. 同一 bbox 复用于：original c23、original c40、四种 fake c23/c40 和对应 mask；
7. 读取 OpenCV BGR frame 后必须显式转为 RGB，再进入 torchvision transform；
8. 先断言 c23/c40/fake frame 的宽高与 crop-source frame 一致；若极少数样本尺寸不同，按宽高比例映射同一 bbox，并在 manifest 记录，不重新检测。

这样可避免 c23/c40 或 real/fake 的 detector 差异被误当作方法差异。

### 8.2 方形框

若 detector 框为 `(l,t,r,b)`：

```text
cx = (l + r) / 2
cy = (t + b) / 2
side = 1.3 * max(r - l, b - t)
left   = floor(cx - side/2)
top    = floor(cy - side/2)
right  = ceil(cx + side/2)
bottom = ceil(cy + side/2)
```

越界部分使用 reflection padding。图像 resize 到 448×448 时使用 bilinear + antialias；mask 使用 nearest neighbor，之后以 `>=0.5` 二值化。

禁止：

- 用 fake 或 mask 决定 crop；
- 对不同方法重新检测；
- 手工修正“看起来不居中”的个别样本而不记录；
- c40 单独检测。

### 8.3 token mask

对 448×448 binary mask 使用 16×16 non-overlapping average pooling，得到 28×28 coverage。Gate A-β 的 token 正标签为：

```text
patch coverage >= 0.25
```

Pixel AP 仍使用 448×448 pixel mask，不使用 token 二值标签替代。

### 8.4 排除规则

在任何评分结果产生前冻结：

- 视频不可解码；
- original target frame 无法检测人脸；
- 对应 fake/mask 文件缺失；
- crop 后出现非有限值；
- fake video 三帧拼接后 mask 正像素总数为 0。

排除发生时，对同一个视频/帧的所有评分器同步排除，保证 paired comparison。最终报告每类排除的数量和分方法分布。

---

## 9. 特征提取与缓存

### 9.1 精度

为减少实现差异，主结果：

- model forward 不开启随机增强；
- 输入 float32；
- 输出转为 float32；
- 落盘 float32；
- 距离计算 float32；
- 不额外 L2 normalize。

可以调 batch size，不得改变数值定义。建议关闭 TF32 以提高跨设备可比性：

```python
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
```

### 9.2 缓存结构

每帧至少保存：

```text
features: float32 [784, 768]
video_name
frame_index
compression
manipulation
crop_box
weight_sha256
dinov3_commit
config_sha256
```

推荐 `.npz`/HDF5/Zarr/Parquet metadata + `.npy` features。若使用单文件容器，必须支持断点重启和写后校验。

### 9.3 gallery 构建

按冻结的 64 个 train real videos：

1. 取中间帧；
2. 按裁剪协议得到 448×448；
3. 提取第 6 block 的 784 tokens；
4. 按 `video_id` 升序、patch row-major 拼接；
5. 得到 `[50176,768]` float32 数组；
6. 保存 gallery array SHA-256；
7. 建立 `faiss.IndexFlatL2(768)`；
8. `index.ntotal` 必须等于 50,176。

gallery 包含全部 crop patch，不按 skin/face/mask 筛选。否则已不再是 v5.1 冻结协议。

---

## 10. 分数实现

### 10.1 FAISS 距离

```python
D2, I = index.search(query.astype("float32"), k)
r = np.sqrt(np.maximum(D2, 0.0))
r = np.maximum(r, 1e-12)
assert np.all(r[:, 1:] >= r[:, :-1] - 1e-7)
```

`D2` 是 squared L2。直接把 `D2` 代入 log-ratio 会使 CIR 分数产生固定倍数变化，并使冻结阈值与其他实现不一致；必须先开平方。

### 10.2 CIR-LID

对每个 query token：

```python
rk = r[:, -1:]
A = np.mean(np.log(rk / r[:, :-1]), axis=1)
S_cir = 1.0 / np.maximum(A, 1e-6)
```

记录：

- `r_j` 被 `1e-12` 截断的 token 比例；
- `A` 被 `1e-6` 截断的 token 比例；
- 非有限分数数量；
- 分数最小/中位/最大值。

### 10.3 C2：归一化算术多近邻均值

```python
S_arith = np.mean(r[:, :-1] / r[:, -1:], axis=1)
```

这是真正的主要公平对手。不得在结果出来后把 C2 换成单一 `r1/rk`。

### 10.4 其他分数

```python
S_r1 = r[:, 0]
S_mean_knn = np.mean(r, axis=1)
S_r1_rk = r[:, 0] / r[:, -1]
```

所有分数统一为“越高越可疑”。

Secondary baselines 固定为：

- **LOF**：`LocalOutlierFactor(n_neighbors=20, metric="euclidean", algorithm="brute", novelty=True)` 在固定 real gallery 上拟合，query score 使用 `-score_samples(query)`；
- **local PCA participation ratio**：对 query 的 20 个 gallery neighbors 居中，计算协方差非负特征值 `λ`，`PR=(Σλ)^2/(Σλ²+1e-12)`，直接以较高 PR 为较可疑；若其方向在 real-only validation 上与定义不符，不允许查看 fake 后翻转，最多作为失败的 secondary baseline 如实报告。

### 10.5 SP-LID

每张图的 784 个 token 构成同图 pool。对每个 query：

- 排除同一 token 自身；
- 从剩余 783 token 取最近 20 个；
- 使用与 CIR 相同的开平方、floor 和公式；
- 不额外 L2 normalize。

为保证 own-token exclusion，可对 784 个 query 搜索全部 784 个 pool items，再按 index 删除自身后取前 20；不要仅假设第一个零距离项一定是自身。

---

## 11. 必须先通过的单元测试

在任何 Gate 实验前，`pytest` 至少覆盖：

1. **shape**：448 输入输出 `[B,784,768]`；
2. **距离定义**：FAISS squared L2 开平方后与 NumPy 欧氏距离一致；
3. **排序**：每个 query 的 `r1≤…≤rk`；
4. **own exclusion**：SP-LID 不包含自身 index；
5. **有限性**：所有主分数为 finite；
6. **方向**：人工构造等距壳层时 CIR/arith 分数高于距离跨度大的邻域；
7. **单调等价**：CIR 与几何均值 `G=(∏_{j<k} r_j/r_k)^(1/(k-1))` 的排序 Spearman 误差为 0（无 ties 时）；
8. **H1**：对每个 query 随机乘 `c(q)>0`，CIR、arith、`r1/rk` 最大绝对误差 `<1e-5`；
9. **绝对距离对照**：`r1` 应随 `c(q)` 缩放；
10. **确定性**：相同输入、配置和权重生成相同 gallery hash；
11. **mask alignment**：人工方块 mask 经 crop/resize 后与 score map 空间一致；
12. **阈值方向**：大于 99th percentile 的像素被判为 suspicious；
13. **frame rule**：帧索引对奇数/偶数帧数均符合定义；
14. **no leakage**：gallery 只有 train real；threshold 只有 validation real；test 在 Gate A/B 前为空。

H1 只作为代码正确性测试，不进入主论文图表。

---

## 12. real-only threshold 校准

对所有 validation real c23 videos 的 25%、50%、75% 三帧：

1. 提取各评分器的 28×28 score map；
2. 双线性上采样到 448×448，`align_corners=False`；
3. 不做 Gaussian smoothing、CRF 或形态学后处理；
4. 对每种评分器分别拼接全部 validation-real pixels；
5. 计算：

```python
threshold = np.quantile(scores, 0.99, method="linear")
```

6. 将 threshold 写入带配置哈希的 JSON；
7. 冻结后用于 validation fake、test real/fake、c23/c40。

禁止：

- 使用 fake mask 选 threshold；
- 按视频调 threshold；
- 在 c40 重新校准；
- 用 top-5% 代替最终 binary mask。

validation c23 上的 pooled FPR 应接近 1%，但因 ties/quantile 定义不保证精确等于 1%。需报告实际值。

---

## 13. Gate A-α：同图参考池的有害不稳定性

### 13.1 目的

不是证明“换 pool 分数会变”，而是判断 SP-LID 对无关 benign reference 变化的波动，是否已经达到 manipulation signal 的可比量级。

### 13.2 固定 query 与 nuisance pool

使用 Gate B validation fake pilot 的 64 directed videos/manipulation 及配对 original frames。对每个 query token 固定 DINOv3 embedding，不重新前向。

建立一个 **real nuisance bank**：来自未进入 gallery 的 train-real videos，每视频一个中间帧，排除 gallery 64 videos。对每个 query frame 创建 10 个固定 seed 的 reference-pool variants：

- baseline：同图其余 783 tokens；
- 每个 nuisance variant：从 baseline pool 均匀选 25% reference tokens，以 real nuisance bank 中均匀采样的 token 替换；
- pool 大小保持 783；
- 替换不按与 query 的距离筛选；
- query embedding 始终不变；
- CIR 始终使用固定 50,176-token real gallery。

50% 替换只作为附加 stress test，不进入 Gate A-α 主判据。

### 13.3 视频级主指标

对 fake 视频中 token mask coverage≥25% 的位置：

```text
numerator = median over positive tokens of SD across 10 nuisance variants S(q)
denominator = |mean baseline score on fake positive tokens
               - mean baseline score on paired real tokens at same positions| + 1e-12
R_video = numerator / denominator
```

对 CIR 也按相同 query 计算；由于 gallery 不变，其 nuisance SD 应接近数值误差，但仍要真实计算而不是写死为 0。

辅助指标：

- 全 crop token 排名的 Spearman correlation；
- top-5% suspicious tokens 的 Jaccard overlap；
- source-frozen threshold 下的 decision flip rate。

### 13.4 Gate A-α

主结果为四种 manipulation 分层后的 video-level 汇总：

1. `median(R_SP) >= 0.50`；
2. 2,000 次 paired、manipulation-stratified video bootstrap 下，`median_video(R_SP-R_CIR)` 的 95% CI 下界 `>0`。

两条同时满足才通过。top-5% 和 flip rate 只解释机制，不替代 Gate。

---

## 14. Gate A-β：reference contamination 降低可分性

### 14.1 污染池

对每种 manipulation 单独构建 fake-reference bank：

- 来源：第 7.4 节中未进入 Gate B pilot 的 validation fake videos；
- 只取 token mask coverage≥25% 的 manipulated tokens；
- 与 query video names 严格不重叠；
- 不按距离挑选 near/far token；
- 四种 manipulation 分开运行。

对每种 manipulation 固定一次 gallery index permutation 和 fake-token permutation。对：

```text
epsilon ∈ {0.0, 0.1, 0.3, 0.5, 0.8}
```

以 `floor(epsilon * 50176)` 个 fake tokens 替换相同数量的 real gallery tokens，总池大小始终为 50,176。各 epsilon 使用嵌套前缀，使污染集合随 epsilon 单调包含。

### 14.2 query 与标签

query 为 Gate B validation fake pilot 与配对 validation real：

- fake token：coverage≥25% 为正；
- paired real 同位置 token 为负；
- 每个视频保留 cluster ID；
- 主分析使用 CIR 公式，但 reference gallery 替换为 `G_epsilon`。

### 14.3 统计

每个 manipulation、每个 epsilon 先计算 token AUROC。主曲线是四种 manipulation 的宏平均 AUROC。

对五个 epsilon 点拟合带截距的一元 OLS：

```text
macro_token_AUROC = intercept + slope * epsilon
```

2,000 次 manipulation-stratified video cluster bootstrap：在每个 manipulation 内重采样 query videos，重新计算 AUROC 曲线和 slope；污染 gallery 固定。

辅助报告：

- `C(epsilon)=median S_fake - median S_real`；
- real/fake score trajectories；
- Pixel AP；
- 每种 manipulation 的单独曲线。

### 14.4 Gate A-β

同时满足：

1. slope 的 95% CI 上界 `<0`；
2. `macro_AUROC(0)-macro_AUROC(0.5) >= 0.05`。

near/far targeted contamination 只能作为附加 stress test，不得替代上述自然随机替换结果。

---

## 15. Gate B 第一部分：合成 anomaly × nuisance

### 15.1 数据生成

每个 seed 生成一个嵌入到 `D=64` 的平滑 4 维曲面。令 `z=(z1,z2,z3,z4)`，先定义 18 维显式映射：

```text
psi(z) = [
  z1, z2, z3, z4,
  0.5*sin(pi*z1/2), 0.5*cos(pi*z1/2),
  0.5*sin(pi*z2/2), 0.5*cos(pi*z2/2),
  0.5*sin(pi*z3/2), 0.5*cos(pi*z3/2),
  0.5*sin(pi*z4/2), 0.5*cos(pi*z4/2),
  0.25*z1*z2, 0.25*z1*z3, 0.25*z1*z4,
  0.25*z2*z3, 0.25*z2*z4, 0.25*z3*z4
]
```

把 `psi(z)` 补零至 64 维，再乘以该 seed 生成并固定的 64×64 随机正交矩阵 `R`：`phi(z)=R[psi(z),0,...,0]`。随后：

1. clean query latent 固定采样自 `Uniform([-0.5,0.5]^4)`；
2. 由 `phi` 的 Jacobian 做 QR 分解，得到局部 tangent basis；
3. 从 tangent space 的正交补中采样单位 normal `n`；
4. clean query 为 `x=phi(z)`；
5. anomaly query 为 `q=x+delta*m*n`，其中 `m` 是 canonical 条件下该 clean query 的局部 20-NN 距离中位数。

```text
delta ∈ {0, 0.25, 0.5, 1.0}
seed count = 20
```

每个 seed 至少 2,000 clean base queries，并对同一个 base query 生成配对 anomaly queries。

### 15.2 nuisance 条件

完整因子：

- gallery size：`{2000, 5000, 10000}`，从同一个 10,000-point gallery permutation 取嵌套前缀；
- latent sampling density：`{sparse, uniform, concentrated}`；
- radial scale：`{0.5, 1.0, 2.0}`，对同一条件的 query-gallery embedding 同时缩放；
- canonical condition：`gallery=5000, uniform, scale=1.0`。

密度条件使用同一组基础均匀随机数 `u∈(0,1)^4`，逐坐标变换后映射为 `z=4v-2`：

```text
sparse:       v = BetaPPF(u; 0.5, 0.5)   # 中央查询区域的 gallery 密度较低
uniform:      v = u
concentrated: v = BetaPPF(u; 2.0, 2.0)   # 中央查询区域的 gallery 密度较高
```

每个 seed 共用同一 clean query latent points、基础 `u`、gallery permutation 和 anomaly directions，以减少非配对噪声。不同方法使用完全相同的距离。

### 15.3 指标

报告：

- 每个 delta、nuisance 条件下的 AUROC；
- canonical 条件下 anomaly score separation；
- 固定 delta 的 nuisance sensitivity；
- nuisance/anomaly sensitivity ratio：

```text
Q_method(seed, delta) =
  SD over nuisance conditions of median anomaly score
  / (|median anomaly score under canonical condition
       - median clean score under canonical condition| + 1e-12)
```

主 `Q_method` 为 seed 和 `delta∈{0.25,0.5,1.0}` 上的中位数。

### 15.4 合成 Gate B 判据

CIR 相对 C2 的 nuisance-sensitivity ratio 至少降低 10%：

```text
(Q_C2 - Q_CIR) / Q_C2 >= 0.10
```

同时必须展示其 anomaly AUROC 没有因“稳定”而退化为无判别力。该实验不要求 CIR 比 `r1` 对离流形位移更敏感。

---

## 16. Gate B 第二部分：validation fake c23→c40 pilot

### 16.1 数据

使用第 7.4 节预先冻结的 64 directed validation fake video names/manipulation：

- 同一视频取 25%、50%、75% 三帧；
- 同一 crop 用于 c23 与 c40；
- gallery 始终是 c23 train real；
- threshold 始终来自 validation real c23；
- 比较至少包括 CIR、C2、C1、`r1`、`r1/rk`。

### 16.2 主终点

每个 fake 视频：

1. 将三帧的 448×448 score pixels 拼接；
2. 将三帧 binary mask pixels 拼接；
3. 计算一个 video-level Pixel AP；
4. manipulation 内对视频宏平均；
5. 四种 manipulation 再宏平均。

主比较为 c40 条件下：

```text
Delta PixelAP = macro PixelAP(CIR) - macro PixelAP(C2)
```

2,000 次 manipulation-stratified paired video bootstrap，CIR 与 C2 使用同一批重采样视频。

辅助报告：

- c23 Pixel AP；
- c23→c40 AP drop；
- frozen threshold 下 real pixel FPR；
- FPR/calibration shift；
- Pixel AUROC、IoU、F1。

### 16.3 真实 Gate B 判据

```text
Delta PixelAP(CIR-C2) >= 0.02
且 95% CI lower bound > 0
```

合成 Gate B 与真实 Gate B 必须同时通过。若 C2 与 CIR 持平，不得因 CIR 名称更“理论化”而继续原投稿叙事。

---

## 17. Gate D：官方 test 的完整实验

只有 Gate A-α、A-β 和 Gate B 均通过后，才生成 test features 和 test scores。

### 17.1 主结果

对官方 test split 的四种 manipulation、c23/c40、每视频三帧运行：

- C1：SP-LID；
- CIR-LID；
- C2：normalized arithmetic multi-neighbor；
- `r1`；
- mean-kNN；
- `r1/rk`；
- LOF；
- local PCA PR（若正文空间不足放补充材料）。

主表至少包含：

```text
method × manipulation Pixel AP
four-manipulation macro Pixel AP
95% CI
CIR-C1 paired difference
CIR-C2 paired difference
```

### 17.2 Gate D 判据

full test 上，CIR 相对 C1 与 C2 都必须满足：

```text
Delta macro PixelAP >= 0.02
且 paired 95% CI lower bound > 0
```

两组比较缺一不可。

### 17.3 sensitivity

仅在主配置跑完后运行：

- layer `{3,6,9,12}`；
- gallery frames `{16,32,64,128}`；
- `k={10,20,40}`。

主配置始终是 layer 6、64 gallery frames、`k=20`。sensitivity 不允许用于事后更换主设置。

---

## 18. heatmap 与像素指标

### 18.1 heatmap

```python
score_28 = token_score.reshape(1, 1, 28, 28)
score_448 = torch.nn.functional.interpolate(
    score_28,
    size=(448, 448),
    mode="bilinear",
    align_corners=False,
)
```

不平滑、不 CRF、不做 test-time augmentation。

### 18.2 指标实现

- Pixel AP：`sklearn.metrics.average_precision_score`；
- Pixel AUROC：`roc_auc_score`；
- binary mask：`score > frozen_threshold`；
- IoU/F1：在每视频三帧拼接结果上计算；
- real FPR：报告 pooled pixels，同时给出 video-level FPR 分布作为辅助；
- calibration shift：报告 c40 real FPR − c23 real FPR，以及 real-score 99th percentile shift。

所有方法的有效视频集合必须相同。若一个方法出现 NaN，不得只删除该方法的样本；应先判定实现错误，修复后全量重跑。

---

## 19. bootstrap 的固定实现

### 19.1 主表

一次 bootstrap replicate：

1. 在每种 manipulation 内，以视频为 cluster、有放回抽取与原样本数相同的视频；
2. 对每个方法计算该 manipulation 的 video-level Pixel AP 平均；
3. 对四种 manipulation 做等权宏平均；
4. paired difference 使用同一组抽样 index；
5. 重复 2,000 次；
6. percentile 2.5%/97.5% 为 95% CI。

固定 bootstrap seeds 从 `20260919` 派生并保存，不因结果重新抽样。

### 19.2 β slope

每个 replicate 仍按 manipulation 分层重采样 query videos，然后在每个 epsilon 重新计算 token AUROC、四方法宏平均曲线和 OLS slope。不得把 token 当作独立 bootstrap unit。

---

## 20. 建议的命令接口与运行顺序

下面是建议的代码接口契约；脚本实现后应能按此顺序无人工改配置地运行：

```bash
# 0. 环境与外部文件审计
python -m src.audit_environment --config configs/frozen_v5_1.yaml
python -m src.audit_dataset --config configs/frozen_v5_1.yaml

# 1. 清单与 crop
python -m src.build_manifest --config configs/frozen_v5_1.yaml
python -m src.cache_crops --config configs/frozen_v5_1.yaml

# 2. 模型与数值测试
pytest -q
python -m src.smoke_test --config configs/frozen_v5_1.yaml

# 3. 特征与 gallery
python -m src.extract_features --subset gallery --config configs/frozen_v5_1.yaml
python -m src.build_gallery --config configs/frozen_v5_1.yaml
python -m src.extract_features --subset validation --config configs/frozen_v5_1.yaml

# 4. real-only threshold
python -m src.calibrate_thresholds --config configs/frozen_v5_1.yaml

# 5. Gates
python -m src.run_gate_alpha --config configs/frozen_v5_1.yaml
python -m src.run_gate_beta --config configs/frozen_v5_1.yaml
python -m src.run_synthetic --config configs/frozen_v5_1.yaml
python -m src.run_gate_b_pilot --config configs/frozen_v5_1.yaml
python -m src.decide_gates --through B --config configs/frozen_v5_1.yaml

# 6. 仅通过 Gate A/B 后
python -m src.extract_features --subset test --config configs/frozen_v5_1.yaml
python -m src.run_gate_d --config configs/frozen_v5_1.yaml
python -m src.bootstrap_results --config configs/frozen_v5_1.yaml
python -m src.make_figures --config configs/frozen_v5_1.yaml
```

`decide_gates` 应读取冻结判据并输出机器可读 JSON，不由操作者凭图主观决定。

---

## 21. 预计结果：方向、门槛与失败解释

以下不是承诺的实验数值，而是预注册的方向和 go/no-go 标准。

| 实验 | 合理预期 | 冻结的继续条件 | 未达到时的解释 |
|---|---|---|---|
| 数值测试 | CIR/C2/`r1/rk` 对 query-wise 径向缩放不变 | 最大误差 `<1e-5` | 实现错误，不能进入数据实验 |
| Gate A-α | SP-LID 对 benign pool replacement 明显波动，CIR 因固定 gallery 接近不变 | `median R_SP≥0.50` 且 paired CI 下界 `>0` | 同图 pool 依赖存在但未证明有实际危害 |
| Gate A-β | fake contamination 增大时 token AUROC 下降 | slope CI 上界 `<0` 且 ε=0→0.5 下降 `≥0.05` | self-cancellation 主张不成立或效应太小 |
| 合成 Gate B | ratio 方法比绝对距离更少受密度/尺度 nuisance；CIR 可能略优于 C2 | CIR 的 `Q` 相对 C2 降低 `≥10%` | 对数/几何聚合无特异性价值 |
| c23→c40 pilot | `r1` 的 threshold/FPR 漂移较大；ratio 方法相对稳定 | CIR−C2 `≥0.02` Pixel AP 且 CI 下界 `>0` | 只能主张归一化/多近邻价值，不能主张 CIR 特异性 |
| Gate D | CIR 背景误报低于 SP-LID，伪造区域排序优于 C2 | 对 C1、C2 均 `≥+0.02` Pixel AP 且 CI 下界 `>0` | 不足以支撑应用型 SPL 主贡献 |

### 最大科学风险

最可能的负结果是：

```text
CIR ≈ normalized arithmetic multi-neighbor score (C2)
```

这不是代码失败，而是有价值的科学裁决：主要增益来自固定 real gallery、多近邻和尺度归一化，而不是 LID/对数聚合。此时应放弃 CIR-LID 的强命名，不能删除 C2 或改换较弱基线。

### 不预测绝对 Pixel AP

在尚未跑数据前，不给出“预计 AP=xx%”。绝对值会受到：

- mask 定义；
- crop detector；
- frame sampling；
- manipulation 类型；
- compression；
- DINOv3 层；
- 视频级宏平均方式

共同影响。本文只冻结可检验的相对效应和最小实用差异。

---

## 22. 大致工作量与时间

以下按单张 24 GB NVIDIA GPU、数据权限已获得估算，仅用于排期：

| 阶段 | 典型工作量 |
|---|---|
| 环境、模型、权限与下载 | 0.5–1 天；数据审批等待时间另计 |
| manifest、crop、smoke test | 0.5–1 天 |
| gallery、validation features、threshold | 0.5 天左右 |
| Gate A-α/A-β | 0.5–1.5 天，取决于 FAISS 和缓存 |
| 合成 + Gate B pilot | 0.5–1.5 天 |
| full Gate D | 1–3 天，包括 exact kNN、bootstrap 和核查 |
| sensitivity 与最终图表 | 1–2 天 |

若只使用 CPU FAISS，kNN 阶段可能显著变慢。先用少量 query 验证 correctness，再决定是否配置 GPU FAISS。

---

## 23. 每阶段必须保存的产物

### 环境

- `pip-freeze.txt`；
- `nvidia-smi` 输出；
- OS/Python/CUDA 信息；
- DINOv3 commit；
- checkpoint SHA-256。

### 数据

- split JSON 哈希；
- immutable manifest；
- gallery IDs；
- Gate B/β 子集名单；
- exclusion log；
- crop boxes。

### 特征和分数

- feature metadata；
- gallery array hash；
- FAISS index type/ntotal；
- clipping frequency；
- threshold JSON；
- 每视频原始指标，不只保存均值。

### 统计

- 2,000 次 bootstrap replicate 数值；
- seed；
- paired sample IDs；
- Gate decision JSON；
- 生成图表的脚本和输入表。

---

## 24. 最终运行前检查表

### 权限与环境

- [ ] 已接受 FF++ Terms of Use 并使用官方脚本；
- [ ] DINOv3 ViT-B/16 LVD-1689M 权重来源和哈希已记录；
- [ ] PyTorch、torchvision、CUDA、FAISS 版本已冻结；
- [ ] DINOv3 repo commit 已冻结；
- [ ] 可用磁盘至少 100 GB。

### 数据与裁剪

- [ ] 使用官方 train/val/test split；
- [ ] gallery 仅来自 train real c23；
- [ ] 64 个 gallery IDs 已落盘；
- [ ] fake/mask 从不用于确定 crop；
- [ ] c23/c40 与 real/fake 复用 target original c23 bbox；
- [ ] 排除规则在看分数前冻结。

### 模型与距离

- [ ] 输出为 `[B,784,768]`；
- [ ] `n=[5]`、`norm=True`、`reshape=False`；
- [ ] 无额外 L2 normalization；
- [ ] features/distances 为 float32；
- [ ] FAISS squared L2 已开平方；
- [ ] `k=20`；
- [ ] gallery 恰为 `[50176,768]`。

### 评估

- [ ] C1、CIR、C2 同时实现；
- [ ] C2 为全部 `j<k` 的算术均值，不是 `r1/rk`；
- [ ] threshold 为 validation real c23 的 method-specific 99th percentile；
- [ ] test 前不重新校准；
- [ ] 每视频三帧拼接后算 Pixel AP；
- [ ] bootstrap unit 是 video，不是 pixel/token；
- [ ] test 在 Gate A/B 通过前保持封存。

---

## 25. 官方来源

1. DINOv3 官方仓库与 PyTorch Hub 用法：  
   https://github.com/facebookresearch/dinov3
2. DINOv3 ViT-B/16 hub 定义：  
   https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py
3. DINOv3 ViT intermediate-layer API：  
   https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py
4. DINOv3 ViT-B/16 LVD-1689M 官方 Hugging Face 页面：  
   https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
5. FaceForensics++ 官方仓库：  
   https://github.com/ondyari/FaceForensics
6. FaceForensics++ 数据下载、大小、mask 与 split 说明：  
   https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md
7. FaceForensics++ 官方处理示例：  
   https://github.com/ondyari/FaceForensicsDataProcessing
8. PyTorch 官方安装选择器：  
   https://pytorch.org/get-started/locally/
9. PyTorch 历史版本安装命令：  
   https://pytorch.org/get-started/previous-versions/
10. IEEE SPL 作者说明与篇幅要求：  
    https://signalprocessingsociety.org/publications-resources/ieee-signal-processing-letters/information-authors-spl

---

## 26. 当前可执行结论

现在可以立即做：

- 环境建立；
- 数据和权重权限申请；
- DINOv3 smoke test；
- 数值单元测试；
- 合成 anomaly × nuisance；
- FF++ 到位后的 manifest、gallery、threshold 和 Gate A/B。

现在不应做：

- 在 Gate A/B 前运行完整 test；
- 根据 pilot 结果修改 layer、`k`、gallery size 或 C2 定义；
- 只挑最好看的 manipulation；
- 提前写“CIR-LID 显著优于其他方法”；
- 把预注册门槛写成已经取得的实验结果。
