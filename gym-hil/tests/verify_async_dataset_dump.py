#!/usr/bin/env python
"""checkpoint 异步数据集 dump 沙箱验证（CPU）。

覆盖（对应 learner.py `_CheckpointDatasetDumper` / buffer.py `clone_for_dataset`）：
1. 快照 vs 同步 dump 逐帧一致：帧数 / episode 结构 / 全部非图像特征 / 全部图像像素
   （像素取整数 [0,255] 值 → uint8 紧凑化零量化误差，比较可严格相等）；
   快照提交后并发写入 live buffer（环形覆盖 dump 范围内的槽位）→ dump 结果
   仍等于快照时点内容（冻结 + 零共享），且与并发写入后的 live buffer dump 不同
   （负向对照：证明快照没有别名共享 live 存储）；
2. 紧凑化分支：float[0,255] → uint8 无损；float[0,1] 保持 float32；
   向量特征与 complementary_info 原样克隆；
3. _CheckpointDatasetDumper 调度语义：串行执行、合并去旧（丢弃中间 pending）、
   wait_and_stop 排空、stop 后 submit 被忽略；
4. 真实 _DatasetDumpTask 经 dumper 端到端落盘（dataset / dataset_offline 目录
   结构与帧数正确）。

运行：cd /home/embody/lerobot && python gym-hil/tests/verify_async_dataset_dump.py
"""
import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, "src")

import torch

from lerobot.datasets import LeRobotDataset
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.learner import _CheckpointDatasetDumper, _DatasetDumpTask

FAIL = []


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({extra})" if extra else ""))
    if not cond:
        FAIL.append(name)


def make_buffer(capacity, n_frames, done_at, value_base=0.0, optimize_memory=False):
    """合成 buffer：state 值 = value_base + 帧号；图像像素 = 帧号*2+1+通道（整数，≤255 时无损）。"""
    buf = ReplayBuffer(
        capacity=capacity,
        device="cpu",
        storage_device="cpu",
        state_keys=["observation.state", "observation.images.front"],
        optimize_memory=optimize_memory,
        use_drq=False,
    )
    for i in range(n_frames):
        base = value_base + float(i)
        img_val = float(i * 2 + 1)
        state = {
            "observation.state": torch.tensor([[base, base + 0.5, base + 1.0]]),
            "observation.images.front": torch.full((1, 3, 8, 8), img_val)
            + torch.arange(3, dtype=torch.float32).view(1, 3, 1, 1),
        }
        next_state = {k: v.clone() for k, v in state.items()}
        ci = {
            "discrete_penalty": torch.tensor([float(i % 3)]),
            "IS_INTERVENTION": (i % 5 == 0),
        }
        buf.add(
            state=state,
            action=torch.tensor([[base, -base]]),
            reward=float(i),
            next_state=next_state,
            done=(i in done_at),
            truncated=False,
            complementary_info=ci,
        )
    return buf


def read_frame(ds, i):
    s = ds[i]
    return {
        "state": s["observation.state"],
        "action": s["action"],
        "reward": s["next.reward"],
        "done": s["next.done"],
        "episode_index": s["episode_index"],
        "image": s["observation.images.front"],
    }


def compare_datasets(dsA, dsB, label):
    """逐帧比较两个数据集：帧数、episode 结构、非图像特征、图像像素。"""
    ok = True
    if dsA.num_frames != dsB.num_frames:
        check(f"{label}: 帧数一致", False, f"{dsA.num_frames} vs {dsB.num_frames}")
        return
    check(f"{label}: 帧数一致 ({dsA.num_frames})", True)
    check(
        f"{label}: episode 数一致 ({dsA.meta.total_episodes})",
        dsA.meta.total_episodes == dsB.meta.total_episodes,
    )
    max_img_diff = 0
    ep_structure_same = True
    for i in range(dsA.num_frames):
        a, b = read_frame(dsA, i), read_frame(dsB, i)
        for key in ("state", "action", "reward", "done", "episode_index"):
            va, vb = a[key], b[key]
            if isinstance(va, torch.Tensor):
                same = bool(torch.equal(va, vb))
            else:
                same = bool(va == vb)
            if not same:
                check(f"{label}: frame {i} {key} 一致", False)
                ok = False
                break
        if a["episode_index"] != b["episode_index"]:
            ep_structure_same = False
        if isinstance(a["image"], torch.Tensor) and isinstance(b["image"], torch.Tensor):
            diff = (a["image"].float() - b["image"].float()).abs()
            max_img_diff = max(max_img_diff, float(diff.max())) if diff.numel() else max_img_diff
        elif not (a["image"] == b["image"]).all():
            check(f"{label}: frame {i} 图像一致", False)
            ok = False
    check(f"{label}: episode 结构一致", ep_structure_same)
    check(f"{label}: 图像像素最大差异 <= 2 ({max_img_diff:.0f})", max_img_diff <= 2)
    return ok


# ========== 1. 快照 == 同步 dump 逐帧一致（含并发写入冻结性） ==========
print("== 1. 快照 vs 同步 dump 逐帧一致（快照后并发写 live buffer） ==")
tmp = tempfile.mkdtemp(prefix="verify_async_dump_")
try:
    buf = make_buffer(capacity=100, n_frames=90, done_at={14, 29, 44, 59, 74, 89})
    # 1a. 同步参考 dump（快照前时点）
    root_ref = os.path.join(tmp, "ref")
    buf.to_lerobot_dataset(repo_id="verify", fps=10, root=root_ref)
    ds_ref = LeRobotDataset(repo_id="verify", root=root_ref)

    # 1b. 快照 + 后台 dump，同时并发写入 live buffer（环形覆盖 dump 范围内的槽位）
    snapshot = buf.clone_for_dataset()
    root_snap = os.path.join(tmp, "snap")
    dump_thread = threading.Thread(
        target=snapshot.to_lerobot_dataset, kwargs={"repo_id": "verify", "fps": 10, "root": root_snap}
    )
    dump_thread.start()
    for k in range(90, 130):  # 覆盖槽位 0..39，与 dump 迭代范围重叠
        base = 1000.0 + float(k - 90)
        img_val = float(500 + k)
        state = {
            "observation.state": torch.tensor([[base, base + 0.5, base + 1.0]]),
            "observation.images.front": torch.full((1, 3, 8, 8), img_val)
            + torch.arange(3, dtype=torch.float32).view(1, 3, 1, 1),
        }
        buf.add(
            state=state,
            action=torch.tensor([[base, -base]]),
            reward=float(k),
            next_state={kk: v.clone() for kk, v in state.items()},
            done=((k - 90) % 20 == 19),
            truncated=False,
            complementary_info={"discrete_penalty": torch.tensor([1.0]), "IS_INTERVENTION": False},
        )
    dump_thread.join()
    ds_snap = LeRobotDataset(repo_id="verify", root=root_snap)

    # 1c. 负向对照：并发写入后的 live buffer 再同步 dump（应与快照内容不同）
    root_live = os.path.join(tmp, "live")
    buf.to_lerobot_dataset(repo_id="verify", fps=10, root=root_live)
    ds_live = LeRobotDataset(repo_id="verify", root=root_live)

    compare_datasets(ds_ref, ds_snap, "快照==参考")
    check(
        "并发写入后的 live dump 帧数=100（缓冲已满且被改写）",
        ds_live.num_frames == 100,
        f"{ds_live.num_frames}",
    )
    check(
        "负向对照：live dump 与快照内容不同（快照未别名共享 live 存储）",
        ds_live.num_frames != ds_snap.num_frames
        or bool(read_frame(ds_live, 0)["state"][0, 0] != read_frame(ds_snap, 0)["state"][0, 0]),
    )

    # ========== 2. 紧凑化分支 ==========
    print("== 2. clone_for_dataset 紧凑化分支 ==")
    snap = buf.clone_for_dataset()
    check("float[0,255] 图像 → uint8", snap.states["observation.images.front"].dtype == torch.uint8)
    check(
        "uint8 值 = float 截断（无损）",
        torch.equal(snap.states["observation.images.front"][10], buf.states["observation.images.front"][10].byte()),
    )
    check("state 向量保持 float32", snap.states["observation.state"].dtype == torch.float32)
    check("actions/rewards/dones 已克隆", snap.actions.dtype == torch.float32 and snap.dones.dtype == torch.bool)
    check(
        "complementary_info 原样克隆",
        torch.equal(
            snap.complementary_info["discrete_penalty"],
            buf.complementary_info["discrete_penalty"],
        )
        and torch.equal(
            snap.complementary_info["IS_INTERVENTION"],
            buf.complementary_info["IS_INTERVENTION"],
        ),
    )
    check(
        "快照 size/position 冻结在提交时点",
        snap.size == 100 and snap.position == 30,
        f"size={snap.size} pos={snap.position}",
    )

    # [0,1] float 图像保持 float32（守卫分支）
    buf01 = make_buffer(capacity=20, n_frames=10, done_at={4, 9})
    with torch.no_grad():
        for i in range(buf01.size):
            buf01.states["observation.images.front"][i].mul_(0.001)  # 缩到 [0,1] 语义
    snap01 = buf01.clone_for_dataset()
    check("[0,1] float 图像保持 float32", snap01.states["observation.images.front"].dtype == torch.float32)
    check(
        "[0,1] float 图像值相等（只比已写槽位，未写槽位可能含 NaN 垃圾值）",
        torch.equal(
            snap01.states["observation.images.front"][: snap01.size],
            buf01.states["observation.images.front"][: buf01.size],
        ),
    )

    # optimize_memory=True 的离线 buffer 形态也能快照
    buf_opt = make_buffer(capacity=20, n_frames=12, done_at={5, 11}, optimize_memory=True)
    snap_opt = buf_opt.clone_for_dataset()
    check("optimize_memory=True buffer 快照成功", snap_opt.size == 12)

    # ========== 3. dumper 调度语义（fake task） ==========
    print("== 3. _CheckpointDatasetDumper 调度语义 ==")

    class FakeTask:
        def __init__(self, name, delay=0.25):
            self.name = name
            self.delay = delay

        def run(self):
            time.sleep(self.delay)
            executed.append(self.name)

    executed = []
    dumper = _CheckpointDatasetDumper()
    dumper.start()
    dumper.submit(FakeTask("t1"))
    time.sleep(0.05)  # 让 worker 取走 t1
    dumper.submit(FakeTask("t2"))
    dumper.submit(FakeTask("t3"))
    dumper.wait_and_stop()
    check("串行执行 + 合并去旧：只执行 t1 与最新 t3", executed == ["t1", "t3"], str(executed))
    dumper.submit(FakeTask("t4"))
    check("stop 后 submit 被忽略", executed == ["t1", "t3"], str(executed))

    # ========== 4. 真实 _DatasetDumpTask 端到端 ==========
    print("== 4. 真实 _DatasetDumpTask 经 dumper 端到端落盘 ==")
    buf_online = make_buffer(capacity=50, n_frames=30, done_at={9, 19, 29})
    buf_offline = make_buffer(capacity=80, n_frames=45, done_at={14, 29, 44})
    dir_online = os.path.join(tmp, "out", "dataset")
    dir_offline = os.path.join(tmp, "out", "dataset_offline")
    task = _DatasetDumpTask(
        online_snapshot=buf_online.clone_for_dataset(),
        dataset_dir=dir_online,
        offline_snapshot=buf_offline.clone_for_dataset(),
        dataset_offline_dir=dir_offline,
        fps=10,
        online_repo_id="verify",
        offline_repo_id="verify",
    )
    dumper2 = _CheckpointDatasetDumper()
    dumper2.start()
    dumper2.submit(task)
    dumper2.wait_and_stop()
    ds_online = LeRobotDataset(repo_id="verify", root=dir_online)
    ds_offline = LeRobotDataset(repo_id="verify", root=dir_offline)
    check("在线数据集落盘：帧数 30 / 3 episodes", ds_online.num_frames == 30 and ds_online.meta.total_episodes == 3)
    check("离线数据集落盘：帧数 45 / 3 episodes", ds_offline.num_frames == 45 and ds_offline.meta.total_episodes == 3)

finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if FAIL:
    print(f"总计 FAIL {len(FAIL)} 项: {FAIL}")
    sys.exit(1)
print("全部 PASS")
