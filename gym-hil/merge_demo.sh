#!/usr/bin/env bash
# ============================================================
# 合并两个演示数据集（把旧备份的 30 集追加进新采集的数据集）
#
# 用法:
#   bash gym-hil/merge_demo.sh [新数据集] [旧数据集]
#     默认新数据集: gym-hil/dataset（最新采集）
#     默认旧数据集: 最新的 gym-hil/dataset_backup_<时间戳>（collect_demo.sh 的备份）
#
# 流程:
#   1. 校验两个数据集可读、目标目录不存在
#   2. 用上游 aggregate_datasets 合并（视频重新打包编码，自动重算 stats，
#      30+60 集预计几分钟~十几分钟）
#   3. 校验输出集数 = 两数据集之和
#   4. 同步 dataset_stats 到 train_hil_env.json / actor_hil_env.json
#   5. 提示替换命令（确认后 dataset_merged 顶替 gym-hil/dataset）
#
# 注意: 合并只做读取不修改源数据集；替换前旧数据始终可恢复。
# ============================================================
set -euo pipefail

NEW_ROOT="${1:-gym-hil/dataset}"
OLD_ROOT="${2:-}"
OUT_ROOT="${3:-gym-hil/dataset_merged}"
TRAIN_JSON="${4:-gym-hil/train_hil_env.json}"
ACTOR_JSON="${5:-gym-hil/actor_hil_env.json}"
PYTHON=/home/embody/miniconda3/envs/lero6/bin/python

cd "$(cd "$(dirname "$0")/.." && pwd)"

# ---------- 0. 前置检查 ----------
if [ ! -x "$PYTHON" ]; then
    echo "[错误] 找不到 python: $PYTHON（需先激活 lero6 环境或修改脚本顶部 PYTHON 变量）" >&2
    exit 1
fi

# 默认旧数据集 = 最新备份
if [ -z "$OLD_ROOT" ]; then
    OLD_ROOT="$(ls -d gym-hil/dataset_backup_* 2>/dev/null | tail -1 || true)"
fi
if [ -z "$OLD_ROOT" ] || [ ! -d "$OLD_ROOT" ]; then
    echo "[错误] 找不到旧数据集备份（默认找 gym-hil/dataset_backup_*，可显式传第二个参数）" >&2
    exit 1
fi
if [ ! -d "$NEW_ROOT" ]; then
    echo "[错误] 新数据集不存在: $NEW_ROOT" >&2
    exit 1
fi
if [ -e "$OUT_ROOT" ]; then
    echo "[错误] 输出目录已存在（不覆盖）: $OUT_ROOT —— 请先删除或改名" >&2
    exit 1
fi

echo "== 合并配置 =="
echo "  新数据集: $NEW_ROOT"
echo "  旧数据集: $OLD_ROOT"
echo "  输出目录: $OUT_ROOT"
echo "  （合并会重新编码视频，耗时几分钟~十几分钟；源数据集不会被修改）"

# ---------- 1. 合并 + stats 同步 ----------
"$PYTHON" - "$NEW_ROOT" "$OLD_ROOT" "$OUT_ROOT" "$TRAIN_JSON" "$ACTOR_JSON" <<'PYEOF'
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

new_root, old_root, out_root, train_json, actor_json = sys.argv[1:]

ds_new = LeRobotDataset(repo_id=new_root, root=new_root, video_backend="pyav")
ds_old = LeRobotDataset(repo_id=old_root, root=old_root, video_backend="pyav")
expect_ep = ds_new.num_episodes + ds_old.num_episodes
expect_fr = len(ds_new) + len(ds_old)
print(f"  源: 新 {ds_new.num_episodes} 集/{len(ds_new)} 帧 + 旧 {ds_old.num_episodes} 集/{len(ds_old)} 帧")

merged = merge_datasets(
    datasets=[ds_new, ds_old],
    output_repo_id=out_root,
    output_dir=out_root,
)
got_ep, got_fr = merged.num_episodes, len(merged)
print(f"  合并完成: {got_ep} 集 / {got_fr} 帧")

# 校验：集数/帧数必须等于两源之和（aggregate 失败会在这里暴露）
if got_ep != expect_ep or got_fr != expect_fr:
    raise SystemExit(f"[错误] 合并结果不符：期望 {expect_ep} 集/{expect_fr} 帧，实际 {got_ep} 集/{got_fr} 帧")

# 同步 dataset_stats（aggregate 自带 stats；缺失时回退重算）
stats_path = Path(out_root) / "meta" / "stats.json"
if not stats_path.exists():
    print("  输出数据集缺 stats，重算中...")
    from tqdm import tqdm
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import write_stats
    from lerobot.scripts.augment_dataset_quantile_stats import process_single_episode
    ep_stats = [process_single_episode(merged, i) for i in tqdm(range(merged.num_episodes), desc="  computing stats")]
    write_stats(aggregate_stats(ep_stats), Path(out_root))
    print("  重算完成")

stats = json.load(open(stats_path))
subset_keys = ["observation.images.wrist", "observation.images.front", "observation.state", "action"]
missing = [k for k in subset_keys if k not in stats]
if missing:
    raise SystemExit(f"[错误] 输出数据集缺少特征 {missing}，未同步 json")
# 压平嵌套统计：视频特征 stats 为 (C,1,1) 嵌套结构（[[[v]]]），draccus 要求
# 扁平 [C] 列表（否则 "Couldn't parse '[[0.0]]' into a float"，2026-08-14 踩坑修复）
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

cap = json.load(open(train_json))["policy"]["offline_buffer_capacity"]
print(f"  dataset_stats 核对: 总帧数={total_frames}, offline_buffer_capacity={cap}")
if cap < total_frames:
    print(f"  [警告] offline_buffer_capacity({cap}) < 帧数({total_frames})——训练时 from_lerobot_dataset 会抛 ValueError，请调大")
else:
    print(f"  ✓ offline_buffer_capacity({cap}) >= 帧数({total_frames})，可直接训练")
PYEOF

# ---------- 2. 替换提示 ----------
echo ""
echo "== 合并完成 =="
echo "  输出: $OUT_ROOT（$NEW_ROOT + $OLD_ROOT）"
echo "  确认无误后替换（顶替当前数据集）:"
echo "    rm -rf $NEW_ROOT && mv $OUT_ROOT $NEW_ROOT"
echo "  源数据集未被修改，替换前随时可重来。"
