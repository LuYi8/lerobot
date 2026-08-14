#!/usr/bin/env bash
# ============================================================
# 扩量采集演示数据（gym-hil HIL-SAC）——一条命令完成全流程
#
# 用法:
#   bash gym-hil/collect_demo.sh [采集集数] [record配置json]
#     默认集数: 60；默认配置: gym-hil/record_hil_env.json
#
# 流程:
#   1. 备份现有数据集（gym-hil/dataset → *_backup_<时间戳>，旧数据保留）
#   2. 更新 record json 的 num_episodes_to_record
#   3. 启动采集（键盘遥操作，录满 N 集自动结束；Ctrl+C 中断保留已完成的集、
#      正在录的集丢弃，并打印恢复提示）
#   4. 采集完成后重算数据集统计（写回 meta/stats.json）
#   5. 同步 dataset_stats 到 train_hil_env.json / actor_hil_env.json
#   6. 打印核对信息（集数/帧数/offline_buffer_capacity 校验）
#
# 注意:
#   - 需要 GPU 环境（record json 里 device=cuda）；采集环节需要人工键盘操作
#   - 采集中断时旧数据在 *_backup_* 目录，可手动 mv 恢复
# ============================================================
set -euo pipefail

N_EPISODES="${1:-60}"
RECORD_JSON="${2:-gym-hil/record_hil_env.json}"
PYTHON=/home/embody/miniconda3/envs/lero6/bin/python
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT"

# ---------- 0. 前置检查 ----------
if [ ! -f "$RECORD_JSON" ]; then
    echo "[错误] 找不到采集配置: $RECORD_JSON" >&2
    exit 1
fi
if [ ! -x "$PYTHON" ]; then
    echo "[错误] 找不到 python: $PYTHON（需先激活 lero6 环境或修改脚本顶部 PYTHON 变量）" >&2
    exit 1
fi

# 读取数据集 root
DATASET_ROOT="$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['dataset']['root'])" "$RECORD_JSON")"
echo "== 采集配置 =="
echo "  数据集目录: $DATASET_ROOT"
echo "  目标集数:   $N_EPISODES"

# ---------- 1. 备份现有数据集 ----------
if [ -d "$DATASET_ROOT" ]; then
    BACKUP_ROOT="${DATASET_ROOT}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "== 备份旧数据集 =="
    echo "  $DATASET_ROOT -> $BACKUP_ROOT"
    mv "$DATASET_ROOT" "$BACKUP_ROOT"
    echo "  备份完成（旧数据保留；中断恢复: mv $BACKUP_ROOT $DATASET_ROOT）"
fi

# ---------- 2. 更新采集集数 ----------
echo "== 更新 $RECORD_JSON 的 num_episodes_to_record = $N_EPISODES =="
"$PYTHON" - "$RECORD_JSON" "$N_EPISODES" <<'PYEOF'
import json, sys
json_path, n = sys.argv[1], int(sys.argv[2])
cfg = json.load(open(json_path))
cfg["dataset"]["num_episodes_to_record"] = n
json.dump(cfg, open(json_path, "w"), indent=4, ensure_ascii=False)
print(f"  已写入 num_episodes_to_record={n}")
PYEOF

# ---------- 3. 采集（人工键盘遥操作） ----------
# Ctrl+C 中断保护：已完成 episode 已逐集落盘（保留），正在录的丢失（帧只在内存）；
# 中断后不重算 stats / 不同步 json，并提示恢复备份。
interrupt_handler() {
    echo ""
    echo "[中断] 采集被 Ctrl+C 终止"
    local saved=0
    if [ -f "$DATASET_ROOT/meta/info.json" ]; then
        saved="$("$PYTHON" -c "import json; print(json.load(open('$DATASET_ROOT/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)"
    fi
    echo "  已保存完整 episode: ${saved} 集（save_episode 已逐集落盘，保留可用）"
    echo "  中断时正在录的 episode: 已丢弃（帧只进内存 buffer，未落盘）"
    echo "  本次未重算 stats / 未同步 json（不完整数据不会被当作完整处理）"
    echo "  旧数据集备份完好:"
    ls -d "${DATASET_ROOT}"_backup_* 2>/dev/null | sed 's/^/    /'
    echo "  恢复命令: mv ${DATASET_ROOT}_backup_<时间戳> $DATASET_ROOT"
    exit 130
}
trap interrupt_handler INT

echo ""
echo "== 开始采集 ${N_EPISODES} 集（键盘遥操作；录满自动结束；Ctrl+C 中断保留已完成的集）=="
echo "  日志: gym-hil/record_trace.log"
"$PYTHON" -m lerobot.rl.gym_manipulator --config_path "$RECORD_JSON" 2>&1 | tee gym-hil/record_trace.log
trap - INT

# ---------- 4. 采集结果检查 ----------
if [ ! -d "$DATASET_ROOT" ] || [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
    echo "[错误] 采集未产生有效数据集 $DATASET_ROOT —— 若需恢复旧数据: mv ${DATASET_ROOT}_backup_* $DATASET_ROOT" >&2
    exit 1
fi

# ---------- 5. stats 重算 + 同步两个 json ----------
echo ""
echo "== 重算数据集统计并同步到 train/actor json =="
"$PYTHON" - "$DATASET_ROOT" gym-hil/train_hil_env.json gym-hil/actor_hil_env.json <<'PYEOF'
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from tqdm import tqdm
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.io_utils import write_stats
from lerobot.scripts.augment_dataset_quantile_stats import process_single_episode

root, train_json, actor_json = sys.argv[1], sys.argv[2], sys.argv[3]

# 加载数据集（pyav 解码：沙箱/训练机均可用，避免 torchcodec 依赖）
dataset = LeRobotDataset(repo_id=root, root=root, video_backend="pyav")
episode_stats_list = []
for i in tqdm(range(dataset.num_episodes), desc="  computing stats"):
    episode_stats_list.append(process_single_episode(dataset, i))
new_stats = aggregate_stats(episode_stats_list)
write_stats(new_stats, Path(root))
print(f"  已写回 {root}/meta/stats.json（{dataset.num_episodes} 集 / {len(dataset)} 帧）")

# 取训练使用的 4 个特征子集，同步两个 json 的 policy.dataset_stats
# 注意：视频特征的统计是 (C,1,1) 嵌套结构（[[[v]]]），必须压平成 [C] 一维
# ——draccus 解析 dataset_stats 时要求 min/mean/std 等为扁平列表，否则
# "Couldn't parse '[[0.0]]' into a float"（2026-08-14 踩坑修复）。
stats = json.load(open(Path(root) / "meta" / "stats.json"))
subset_keys = ["observation.images.wrist", "observation.images.front", "observation.state", "action"]
missing = [k for k in subset_keys if k not in stats]
if missing:
    print(f"[错误] 数据集缺少特征 {missing}，未同步 json", file=sys.stderr)
    sys.exit(1)
subset = {
    k: {sk: np.asarray(sv).reshape(-1).tolist() for sk, sv in stats[k].items()}
    for k in subset_keys
}

total_frames = stats["observation.state"]["count"][0]
for json_path in (train_json, actor_json):
    cfg = json.load(open(json_path))
    cfg["policy"]["dataset_stats"] = subset
    json.dump(cfg, open(json_path, "w"), indent=4, ensure_ascii=False)
    print(f"  已同步 {json_path}")

# 校验 learner 侧 offline_buffer_capacity >= 数据集长度（buffer.py:554 硬校验）
train_cfg = json.load(open(train_json))
cap = train_cfg["policy"]["offline_buffer_capacity"]
print(f"  dataset_stats 核对: 总帧数={total_frames}, offline_buffer_capacity={cap}")
if cap < total_frames:
    print(f"  [警告] offline_buffer_capacity({cap}) < 数据集帧数({total_frames})——from_lerobot_dataset 会抛 ValueError，请调大该值")
else:
    print(f"  ✓ offline_buffer_capacity({cap}) >= 帧数({total_frames})，可直接训练")
PYEOF

echo ""
echo "== 全部完成 =="
echo "  新数据集: $DATASET_ROOT（$N_EPISODES 集目标，实际以 info.json 为准）"
echo "  旧数据集备份: ${DATASET_ROOT}_backup_*（确认新数据无误后可删除）"
echo "  下一步: 训练命令见 gym-hil/命令.txt；若换数据管线记得用同配置重跑所有算法基线"
