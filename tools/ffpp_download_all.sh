#!/bin/bash
# FaceForensics++ 全量下载（按 Jiang sir 指定的范围，2026-09-22）
#
# 下载清单：
#   -d original  -c c23 -t videos
#   -d original  -c c40 -t videos
#   Deepfakes / Face2Face / FaceSwap / NeuralTextures 各：-c c23 -t videos
#   Deepfakes / Face2Face / FaceSwap / NeuralTextures 各：-t masks
#   Deepfakes / Face2Face / FaceSwap / NeuralTextures 各：-c c40 -t videos
#
# 设计要点：
#   1. 串行执行 —— 实测两个实例并发反而互相拖慢，且会在同一目录里抢写
#   2. 防重复启动 —— pgrep 查重 + pidfile（这个坑踩过两次）
#   3. 脚本自带断点续传 —— download_file 会跳过已存在的文件，中断后重跑即可
#   4. 每步结束清理未完成的 tmp* 文件
#   5. 全部日志写盘，可随时 tail

set -uo pipefail

ROOT=/root/autodl-tmp/cirlid55
FFPP=$ROOT/data/raw/ffpp
SCRIPT=$ROOT/faceforensics_download_v4_patched.py
PY=/root/miniconda3/bin/python
LOG=$FFPP/_download_all.log
PIDFILE=$FFPP/_download_all.pid

# 服务器：EU2 -> kaldir.vc.in.tum.de（实测唯一可达的一台）
# EU(canis) 与 CA(falas) 均连接超时/拒绝。绝不能用脚本默认值 EU。
SERVER=EU2

mkdir -p "$FFPP"

# ---- 防重复启动 ----
# 主守卫：pidfile + kill -0 判活（不要用 pgrep 匹配本脚本名，
# 否则启动命令行里含同名字面量时会误判成"已在运行"，2026-09-22 踩过）
if [ -f "$PIDFILE" ]; then
    OLD=$(tr -d '\r\n' < "$PIDFILE" 2>/dev/null)
    if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
        echo "ABORT: 已有实例在运行 (pid=$OLD)" >> "$LOG"
        exit 1
    fi
fi
# 次守卫：是否已有 python 下载进程
if pgrep -f '[f]aceforensics_download_v4' >/dev/null 2>&1; then
    echo "ABORT: 已有 faceforensics 下载进程在运行" >> "$LOG"
    exit 1
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

exec >>"$LOG" 2>&1

echo
echo "############################################################"
echo "############ FF++ 下载开始 $(date '+%F %T') ############"
echo "############ PID=$$  输出目录=$FFPP"
echo "############################################################"

run() {
    local desc="$1"; shift
    echo
    echo "===== [$(date '+%F %T')] 开始：$desc ====="
    # 脚本第 144 行有 input('')，必须喂一个回车，否则非交互运行会 EOFError
    printf '\n' | "$@"
    echo "===== [$(date '+%F %T')] 结束：$desc  exit=$? ====="
    # 清理本次可能残留的未完成下载
    find "$FFPP" -type f -name 'tmp*' -delete 2>/dev/null
    df -h /root/autodl-tmp | tail -1
}

dl() { "$PY" "$SCRIPT" --server "$SERVER" "$FFPP" "$@"; }

# ---- 预检：EU2 源必须可达，否则 14 个步骤会全部白跑 ----
echo
echo "--- 预检 ${SERVER} 源可达性 ---"
FL_URL="http://kaldir.vc.in.tum.de/faceforensics/v3/misc/filelist.json"
if ! curl -s -f --max-time 25 -o /dev/null "$FL_URL"; then
    echo "ABORT: EU2 源不可达 ($FL_URL)"
    exit 1
fi
echo "EU2 源可达，开始下载"

# ---- 1) original ----
run "original c23 videos" dl -d original -c c23 -t videos
run "original c40 videos" dl -d original -c c40 -t videos

# ---- 2) 四种 manipulation：c23 videos + masks ----
for M in Deepfakes Face2Face FaceSwap NeuralTextures; do
    run "$M c23 videos" dl -d "$M" -c c23 -t videos
    run "$M masks"      dl -d "$M" -t masks
done

# ---- 3) 四种 manipulation：c40 videos ----
for M in Deepfakes Face2Face FaceSwap NeuralTextures; do
    run "$M c40 videos" dl -d "$M" -c c40 -t videos
done

echo
echo "############################################################"
echo "############ 全部结束 $(date '+%F %T') ############"
echo "############################################################"
echo "--- 各目录文件数 ---"
for d in "$FFPP"/original_sequences/youtube/*/videos \
         "$FFPP"/manipulated_sequences/*/*/videos \
         "$FFPP"/manipulated_sequences/*/masks/videos; do
    [ -d "$d" ] && printf "%6s  %s\n" "$(find "$d" -maxdepth 1 -name '*.mp4' | wc -l)" "${d#$FFPP/}"
done
echo "--- 总体积 ---"
du -sh "$FFPP"
df -h /root/autodl-tmp | tail -1
rm -f "$PIDFILE"
