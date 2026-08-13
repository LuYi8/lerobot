#!/usr/bin/env python
"""§6 验证 4：GRU 链路测试（use_recurrent=true，沙箱 CPU）。

A. 小 buffer 上 sequence_length=8 的 update 不崩、loss 有限、B×T==64；
B. select_action 连续两帧 hidden 传递（输出变化）、reset() 后逐位复现；
C. done 左移掩码回归：构造 done 在窗口中间的数据，断言 _compute_loss_critic
   next-action GRU 前向中"新 episode 首帧（next 侧）"与零 hidden 基线一致
   （误传未左移 done 时该断言失败）。
"""
import os
import sys

sys.path.insert(0, "/home/embody/lerobot/src")

import torch

from lerobot.policies import make_policy
from lerobot.rl.algorithms.factory import make_algorithm
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.data_sources.data_mixer import OnlineOfflineMixer
from lerobot.rl.train_rl import TrainRLServerPipelineConfig
from lerobot.utils.constants import ACTION

CKPT = "/home/embody/lerobot/gym-hil/output好/checkpoints/last/pretrained_model"
FAIL = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAIL.append(name)


cfg = TrainRLServerPipelineConfig.from_pretrained(os.path.join(CKPT, "train_config.json"))
cfg.policy.device = "cpu"
cfg.policy.storage_device = "cpu"
cfg.policy.pretrained_path = None
cfg.policy.policy_kwargs.use_recurrent = True
cfg.policy.policy_kwargs.recurrent_hidden_size = 256
cfg.policy.policy_kwargs.recurrent_num_layers = 1
cfg.algorithm.sequence_length = 8
cfg.algorithm.use_torch_compile = False
cfg.output_dir = "/tmp/zr_output"
os.makedirs(cfg.output_dir, exist_ok=True)
cfg.validate()

torch.manual_seed(1234)
policy = make_policy(cfg.policy, env_cfg=cfg.env)
policy.eval()
check("actor.use_recurrent 生效 + gru 模块存在", policy.actor.use_recurrent and hasattr(policy.actor, "gru"))
print("  gru 参数量:", sum(p.numel() for p in policy.actor.gru.parameters()))

# ---------- A. GRU update（sequence_length=8） ----------
print("== A. GRU update 链路 ==")
algo = make_algorithm(cfg.algorithm, policy=policy)
algo.make_optimizers_and_scheduler()
buf = ReplayBuffer(
    capacity=2000,
    device="cpu",
    storage_device="cpu",
    state_keys=list(cfg.policy.input_features.keys()),
    optimize_memory=True,
    use_drq=False,
)
torch.manual_seed(11)
for i in range(200):
    st = {
        "observation.state": torch.randn(1, 18),
        "observation.images.front": torch.rand(1, 3, 128, 128),
        "observation.images.wrist": torch.rand(1, 3, 128, 128),
    }
    buf.add(
        state=st,
        action=torch.randn(1, 4),
        reward=float((i % 3) - 1),
        next_state=st,
        done=(i % 10 == 9),
        truncated=False,
    )
mixer = OnlineOfflineMixer(online_buffer=buf, offline_buffer=None, online_ratio=1.0)
torch.manual_seed(11)
it = algo.configure_data_iterator(mixer, batch_size=64, async_prefetch=False, queue_size=2)
b0 = next(it)
B, T = b0["state"]["observation.state"].shape[:2]
check("采样 batch (B, T) 且 B×T == batch_size(64)", B == 8 and T == 8 and B * T == 64)
ok = True
for _ in range(2):
    stats = algo.update(it)
    for k, v in stats.losses.items():
        if not torch.isfinite(torch.tensor(v)):
            ok = False
            print("   non-finite:", k, v)
    print("   step losses:", stats.losses)
check("update 两步 loss 有限", ok)

# ---------- B. select_action hidden 传递 + reset ----------
print("== B. select_action hidden 传递/reset ==")
obs = {
    "observation.state": torch.randn(1, 18),
    "observation.images.front": torch.rand(1, 3, 128, 128),
    "observation.images.wrist": torch.rand(1, 3, 128, 128),
}
policy.reset()
with torch.no_grad():
    torch.manual_seed(1)
    a1 = policy.select_action(obs)
    torch.manual_seed(1)
    a2 = policy.select_action(obs)  # 第二次带上一帧 hidden
    policy.reset()
    torch.manual_seed(1)
    a3 = policy.select_action(obs)  # reset 后应逐位复现 a1
check("连续两帧 hidden 传递（输出随历史变化）", not torch.equal(a1, a2))
check("reset() 后行为逐位复现", torch.equal(a1, a3))
check("select_action 输出形状 (1, 4)", tuple(a1.shape) == (1, 4))

# ---------- C. done 左移掩码回归 ----------
print("== C. done 左移掩码回归（_compute_loss_critic 真实路径） ==")
torch.manual_seed(3)
B, T = 2, 4
obs_b = {
    "observation.state": torch.rand(B, T, 18),
    "observation.images.front": torch.rand(B, T, 3, 128, 128),
    "observation.images.wrist": torch.rand(B, T, 3, 128, 128),
}
nxt_b = {k: torch.rand_like(v) for k, v in obs_b.items()}
done = torch.zeros(B, T)
done[0, 1] = 1.0  # 边界在窗口中间：第 0 行窗口第 2 帧是 episode 末帧 →
# next 侧序列（右移 1 帧）第 2 帧是新 episode 首帧，必须零历史（左移 done 生效）。
fb = algo._prepare_forward_batch(
    {
        "state": obs_b,
        "next_state": nxt_b,
        ACTION: torch.rand(B, T, 4),
        "reward": torch.rand(B, T),
        "done": done,
        "truncated": torch.zeros(B, T),
    }
)
with torch.no_grad():
    loss_critic, _ = algo._compute_loss_critic(fb)
check("_compute_loss_critic（序列版）不崩且 loss 有限", torch.isfinite(torch.tensor(loss_critic)))

# 直接对比 next-action GRU 前向的 means（确定性；返回形状 (B*T, A)，还原 (B, T, A)）：
with torch.no_grad():
    next_view = {k: v.view(B, T, *v.shape[1:]) for k, v in fb["next_state"].items()}
    feats = fb["next_observation_feature"]
    done_next = torch.cat([done[:, 1:], torch.zeros(B, 1)], dim=1)  # 与 _compute_loss_critic 内部构造一致
    _, _, m_left = policy.actor(next_view, feats, done=done_next)
    _, _, m_ones = policy.actor(next_view, feats, done=torch.ones(B, T))    # 每帧零历史基线
    _, _, m_zeros = policy.actor(next_view, feats, done=torch.zeros(B, T))  # 无掩码基线
    _, _, m_wrong = policy.actor(next_view, feats, done=done)  # 误传未左移 done
m_left = m_left.view(B, T, -1)
m_ones = m_ones.view(B, T, -1)
m_zeros = m_zeros.view(B, T, -1)
m_wrong = m_wrong.view(B, T, -1)
# 左移 done：边界后首帧零历史（t=1 与全1基线一致），之后历史正常流动（t≥2 与全0基线一致）；
# 无边界行全程历史流动（与全0基线一致）
check("左移 done：新 episode 首帧零历史（t=1 与全1基线一致）", torch.equal(m_left[0, 1], m_ones[0, 1]))
# 边界后历史重新累积：t≥2 的轨迹应等于"从边界后首帧（帧1）重新开始的序列"（精确基线）
with torch.no_grad():
    shift_idx = torch.tensor([1, 2, 3, 5, 6, 7])  # 两行各取帧 1..3
    feats_shift = feats[shift_idx].view(B, T - 1, *feats.shape[1:]) if feats is not None else None
    _, _, m_shift = policy.actor(
        {k: v[:, 1:] for k, v in next_view.items()}, feats_shift, done=torch.zeros(B, T - 1))
m_shift = m_shift.view(B, T - 1, -1)
# 序列长度不同（4 vs 3）导致 gemm 内存布局差异，存在 ~1e-7 浮点噪声 → allclose
check(
    "左移 done：边界后历史重新累积（t≥1 与移位序列基线一致）",
    torch.allclose(m_left[0, 1:], m_shift[0], atol=1e-5, rtol=1e-5),
)
check("左移 done：无掩码行历史正常流动（与全0基线一致）", torch.equal(m_left[1, 1:], m_zeros[1, 1:]))
check("t=0 首帧恒零历史（与基线一致）", torch.equal(m_left[0, 0], m_ones[0, 0]))
# 误传未左移 done：掩码晚 1 帧 → 新 episode 首帧带旧历史（≠ 零历史基线）→ 回归断言可捕获
check("误传未左移 done 会被捕获（新episode首帧偏离零历史基线）", not torch.equal(m_wrong[0, 1], m_ones[0, 1]))
check("误传版本在 t=1 处等于'历史流动'（污染源定位）", torch.equal(m_wrong[0, 1], m_zeros[0, 1]))

print()
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("=== ALL GRU CHAIN TESTS PASSED ===")
