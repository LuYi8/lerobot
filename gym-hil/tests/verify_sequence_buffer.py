#!/usr/bin/env python
"""§6 验证 2：Buffer 序列采样正确性测试（沙箱 CPU）。

构造已知数据：2 段 episode 各 10 帧（done 在位置 9/19），state 值 = 位置索引，
可据采样值反推窗口起点，逐位断言对齐/越界/环形/保护/DRQ 还原/计算量。
运行：cd /home/embody/lerobot && python gym-hil/tests/verify_sequence_buffer.py
"""
import sys

sys.path.insert(0, "src")

import torch

from lerobot.rl.buffer import ReplayBuffer

FAIL = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        FAIL.append(name)


def make_buffer(capacity, n_frames, done_at, optimize_memory, use_drq=False):
    buf = ReplayBuffer(
        capacity=capacity,
        device="cpu",
        storage_device="cpu",
        state_keys=["observation.state", "observation.images.front"],
        optimize_memory=optimize_memory,
        use_drq=use_drq,
    )
    for i in range(n_frames):
        val = i % capacity  # 值 = 位置（回绕后仍成立）
        nval = (i + 1) % capacity
        state = {
            "observation.state": torch.tensor([[float(val), float(val) + 0.5]]),
            "observation.images.front": torch.full((1, 3, 8, 8), float(val % 10) / 10.0),
        }
        next_state = {
            "observation.state": torch.tensor([[float(nval), float(nval) + 0.5]]),
            "observation.images.front": torch.full((1, 3, 8, 8), float(nval % 10) / 10.0),
        }
        buf.add(
            state=state,
            action=torch.tensor([[float(i)]]),
            reward=float(i),
            next_state=next_state,
            done=(i in done_at),
            truncated=False,
        )
    return buf


# ---------- 1. 未满 + optimize_memory=True：越界 / 对齐 ----------
print("== 未满(20帧) + optimize_memory=True, T=4 ==")
buf = make_buffer(100, 20, {9, 19}, optimize_memory=True)
torch.manual_seed(0)
ok_all = True
ok_align = True
for _ in range(200):
    b = buf.sample(8, sequence_length=4)
    B, T = b["state"]["observation.state"].shape[:2]
    assert T == 4 and 0 < B <= 5, f"shape {b['state']['observation.state'].shape}"
    idx = b["state"]["observation.state"][:, 0, 0].long()
    for bi in range(B):
        p = idx[bi].item()
        if not (0 <= p <= 15):  # high = size-T = 16（半开）→ idx ≤ 15
            ok_all = False
        for t in range(T):
            # 窗口内 state 逐位 = 位置
            if b["state"]["observation.state"][bi, t, 0].item() != p + t:
                ok_all = False
            # next_state_t ≡ state_{t+1}（错位 1，optimize_memory 语义）
            if b["next_state"]["observation.state"][bi, t, 0].item() != p + t + 1:
                ok_align = False
            # 与底层数组逐位一致（含图像键）
            if not torch.equal(
                b["next_state"]["observation.images.front"][bi, t],
                buf.states["observation.images.front"][(p + t + 1) % 100],
            ):
                ok_align = False
check("窗口全在已写区（idx ≤ size-T-1，末帧 next 不越界）", ok_all)
check("next_state_t == state_{t+1} 逐位一致（state+图像）", ok_align)

# ---------- 2. 未满 + optimize_memory=False：对齐 ----------
print("== 未满(20帧) + optimize_memory=False, T=4 ==")
buf = make_buffer(100, 20, {9, 19}, optimize_memory=False)
torch.manual_seed(0)
ok_align = True
ok_range = True
for _ in range(200):
    b = buf.sample(8, sequence_length=4)
    B, T = b["state"]["observation.state"].shape[:2]
    assert T == 4 and 0 < B <= 5
    idx = b["state"]["observation.state"][:, 0, 0].long()
    for bi in range(B):
        p = idx[bi].item()
        if not (0 <= p <= 16):  # high = size-T+1 = 17（半开）→ idx ≤ 16
            ok_range = False
        for t in range(T):
            if b["state"]["observation.state"][bi, t, 0].item() != p + t:
                ok_range = False
            if b["next_state"]["observation.state"][bi, t, 0].item() != p + t + 1:
                ok_align = False
            if not torch.equal(
                b["next_state"]["observation.images.front"][bi, t],
                buf.next_states["observation.images.front"][p + t],
            ):
                ok_align = False
check("窗口全在已写区（idx ≤ size-T）", ok_range)
check("next_state_t == state_{t+1} 逐位一致（独立 next_states）", ok_align)

# ---------- 3. 跨 episode：done 标志正确 ----------
print("== 跨 episode 窗口（done 在 9/19） ==")
buf = make_buffer(100, 20, {9, 19}, optimize_memory=True)
torch.manual_seed(0)
ok_done = True
found_cross = False
for _ in range(200):
    b = buf.sample(8, sequence_length=4)
    B, T = b["state"]["observation.state"].shape[:2]
    idx = b["state"]["observation.state"][:, 0, 0].long()
    for bi in range(B):
        p = idx[bi].item()
        expected = torch.tensor(
            [float(p + t in {9, 19}) for t in range(T)], dtype=b["done"].dtype
        )
        if not torch.equal(b["done"][bi], expected):
            ok_done = False
        if p < 9 < p + 3 or p < 19 < p + 3:
            found_cross = True
check("done 标志与位置逐位一致", ok_done)
check("采样到跨 episode 窗口（T=4 覆盖边界）", found_cross)

# ---------- 4. 满（环形）：idx 范围 / wrap 窗口 / next 环形索引 ----------
print("== 满(容量20, 40帧) + optimize_memory=True, T=4 ==")
buf = make_buffer(20, 40, {9, 19, 29, 39}, optimize_memory=True)
assert buf.size == 20
torch.manual_seed(0)
ok_range = True
ok_wrap = True
found_wrap = False
for _ in range(400):
    b = buf.sample(8, sequence_length=4)
    B, T = b["state"]["observation.state"].shape[:2]
    assert T == 4 and 0 < B <= 5
    idx = b["state"]["observation.state"][:, 0, 0].long()
    for bi in range(B):
        p = idx[bi].item()
        if not (0 <= p <= 16):  # high = capacity-T+1 = 17（半开）→ idx ≤ 16
            ok_range = False
        for t in range(T):
            if b["state"]["observation.state"][bi, t, 0].item() != (p + t) % 20:
                ok_range = False
            if b["next_state"]["observation.state"][bi, t, 0].item() != (p + t + 1) % 20:
                ok_wrap = False
            if not torch.equal(
                b["next_state"]["observation.images.front"][bi, t],
                buf.states["observation.images.front"][(p + t + 1) % 20],
            ):
                ok_wrap = False
        if p + T >= 20:
            found_wrap = True
check("环形 idx 范围（≤ capacity-T）+ wrap 窗口位置正确", ok_range)
check("next 用 (idx+T) % capacity 环形索引（含跨界窗口）", ok_wrap)
check("采样到 wrap 窗口（idx+T ≥ capacity）", found_wrap)

# ---------- 5. 保护：size < T 抛错 ----------
print("== 保护 ==")
buf = make_buffer(100, 3, {2}, optimize_memory=True)
try:
    buf.sample(1, sequence_length=4)
    check("size < T 抛 ValueError", False)
except ValueError:
    check("size < T 抛 ValueError", True)

# ---------- 6. DRQ 增强后 shape 还原 (B, T, C, H, W) ----------
print("== DRQ 图像增强 ==")
buf = make_buffer(100, 20, {9, 19}, optimize_memory=True, use_drq=True)
torch.manual_seed(0)
b = buf.sample(8, sequence_length=4)
img = b["state"]["observation.images.front"]
check("DRQ 后 shape (B, T, C, H, W)", tuple(img.shape) == (5, 4, 3, 8, 8))
check("DRQ 后 next 同 shape", tuple(b["next_state"]["observation.images.front"].shape) == (5, 4, 3, 8, 8))
check("DRQ 后为 float 且在 [0,1]", img.dtype.is_floating_point and img.min() >= 0 and img.max() <= 1)

# ---------- 7. 计算量：B×T == batch_size ----------
print("== 计算量恒等 ==")
buf = make_buffer(200, 200, {99, 199}, optimize_memory=True)
b = buf.sample(8, sequence_length=8)
check("B×T == batch_size(64)", b["state"]["observation.state"].shape[0] * 8 == 64)

# ---------- 8. OnlineOfflineMixer 序列透传 ----------
print("== OnlineOfflineMixer（online_ratio=0.5） ==")
from lerobot.rl.data_sources.data_mixer import OnlineOfflineMixer

on = make_buffer(100, 20, {9, 19}, optimize_memory=True)
off = make_buffer(100, 20, {9, 19}, optimize_memory=True)
mixer = OnlineOfflineMixer(online_buffer=on, offline_buffer=off, online_ratio=0.5)
torch.manual_seed(0)
b = mixer.sample(8, sequence_length=4)
check("mixer 序列 batch (8, 4, ...) 拼接", b["state"]["observation.state"].shape[:2] == (8, 4))
check("mixer done (8, 4)", tuple(b["done"].shape) == (8, 4))
it = mixer.get_iterator(batch_size=8, async_prefetch=False, sequence_length=4)
b2 = next(it)
check("mixer get_iterator 透传 sequence_length", b2["state"]["observation.state"].shape[:2] == (8, 4))

print()
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("=== ALL BUFFER TESTS PASSED ===")
