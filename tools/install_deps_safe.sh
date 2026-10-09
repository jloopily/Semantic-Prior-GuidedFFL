#!/bin/bash
# CIR-LID 服务器依赖安装（安全版）
#
# 设计原则（2026-09-21 内存打满事故后重写）：
#   1. 只装 wheel，--only-binary=:all: 强制禁止源码编译
#   2. 即使出现意外编译，CMAKE/MAKE 并行度也锁死为 1
#   3. dlib 这类需要编译的包**默认跳过**，等 Jiang sir 决策后单独处理
#   4. 前台运行，不后台挂 —— 避免孤儿进程占住 ssh 通道
#   5. 启动前先确认没有别的 pip 在跑
#
# 用法（前台运行，看完整日志）：
#   B64=$(base64 < tools/install_deps_safe.sh | tr -d '\n')
#   tools/cirlid_run.sh "echo $B64 | base64 -d > /tmp/install_deps_safe.sh && bash /tmp/install_deps_safe.sh"
#
# 若确实需要 dlib（需 Jiang sir 明确同意）：
#   tools/cirlid_run.sh "INSTALL_DLIB=1 bash /tmp/install_deps_safe.sh"

set -uo pipefail

PY=/root/miniconda3/bin/python
CONDA=/root/miniconda3/bin/conda

echo "############ 资源预检 $(date '+%F %T') ############"
echo "memory.max = $(cat /sys/fs/cgroup/memory.max 2>/dev/null)   (2147483648 = 2 GB)"
echo "cpu.max    = $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)      (50000 100000 = 0.5 核)"
echo "注意：nproc 报的是宿主机核数，禁止用它决定并行度"
if pgrep -af '[p]ip install' >/dev/null 2>&1; then
    echo "ABORT: 已有 pip install 在运行，先停掉它"
    exit 1
fi

# 防御性限流：任何意外触发的编译都只能单线程
export CMAKE_BUILD_PARALLEL_LEVEL=1
export MAKEFLAGS=-j1
export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo
echo "############ 阶段 1/3：apt 装 ffmpeg（单独跑，不与 pip 并发） ############"
if command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg 已存在，跳过"
else
    apt-get update -qq
    apt-get install -y -qq ffmpeg
    echo "apt_exit=$?"
fi
ffmpeg -version 2>/dev/null | head -1

echo
echo "############ 阶段 2/3：纯 wheel 批（禁止源码编译） ############"
PACKAGES="scipy scikit-learn pandas opencv-python-headless seaborn pytest \
faiss-cpu h5py pyarrow zarr av einops \
ftfy omegaconf regex submitit termcolor torchmetrics iopath"

$PY -m pip install --no-cache-dir --only-binary=:all: $PACKAGES
echo "wheel_batch_exit=$?"

echo
echo "############ 阶段 3/3：需要编译/系统库的包 ############"
if [ "${INSTALL_DLIB:-0}" = "1" ]; then
    echo "--- 走 conda-forge 预编译 dlib（不编译源码） ---"
    $CONDA install -y -c conda-forge dlib
    echo "dlib_exit=$?"
else
    echo "按设计跳过：dlib 需要编译，本机 0.5 核 / 2 GB 不适合，"
    echo "           且它在数据准备阶段才用得上。等 Jiang sir 决策。"
fi

echo
echo "############ 校验 ############"
$PY -m pip check
$PY - <<'PY'
import importlib
mods = ["numpy","scipy","sklearn","pandas","cv2","PIL","faiss","h5py","pyarrow",
        "zarr","av","einops","seaborn","pytest","ftfy","omegaconf","regex",
        "submitit","termcolor","torchmetrics","iopath","torch","torchvision",
        "matplotlib","tqdm","yaml"]
miss = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print("OK    %-14s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        miss.append(m)
        print("MISS  %-14s %s" % (m, type(e).__name__))
try:
    import dlib
    print("OK    %-14s %s" % ("dlib", dlib.__version__))
except Exception:
    print("MISS  dlib           (按设计跳过)")
print()
print("STILL_MISSING = " + (", ".join(miss) if miss else "NONE"))
PY

echo
echo "############ 磁盘 ############"
df -h / /root/autodl-tmp | tail -2
echo "############ 完成 $(date '+%F %T') ############"
