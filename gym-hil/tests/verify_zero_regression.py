#!/usr/bin/env python
"""§6 验证 3：零回归对比脚本（同一脚本在改动前/后代码上各跑一次，输出到文件后逐位对比）。

用法：python zr_verify.py <repo_src_dir> <output_prefix>
  - <repo_src_dir>: 改动前 /tmp/lerobot-orig/src 或 改动后 /home/embody/lerobot/src
  - 输出 <prefix>.fwd.pt（Policy.forward 三元组）与 <prefix>.loss.json（update 两步 losses）
"""
import json
import os
import sys

sys.path.insert(0, sys.argv[1])

import torch

CKPT = "/home/embody/lerobot/gym-hil/output好/checkpoints/last/pretrained_model"
PREFIX = sys.argv[2]

torch.manual_seed(1234)
torch.set_num_threads(4)

from lerobot.policies import make_policy
from lerobot.rl.algorithms.factory import make_algorithm
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.data_sources.data_mixer import OnlineOfflineMixer
from lerobot.rl.train_rl import TrainRLServerPipelineConfig
from lerobot.rl.trainer import RLTrainer

cfg = TrainRLServerPipelineConfig.from_pretrained(os.path.join(CKPT, "train_config.json"))
cfg.policy.device = "cpu"
cfg.policy.storage_device = "cpu"
cfg.policy.pretrained_path = CKPT
# 沙箱 CPU 下 dynamo 无法对 frozen ResNet10 的符号形状做 fake-tensor 推断
# （真机 GPU 训练时 use_torch_compile=true 可正常收敛）；两侧同样关闭以对比。
cfg.algorithm.use_torch_compile = False
# 与 learner.py train() 一致：validate 填充 algorithm.policy_config 等派生字段；
# output_dir 指到临时目录避免与真实训练输出冲突。
cfg.output_dir = "/tmp/zr_output"
os.makedirs(cfg.output_dir, exist_ok=True)
cfg.validate()

policy = make_policy(cfg.policy, env_cfg=cfg.env)
policy.eval()

# ---------- 1) Policy.forward 输出（use_recurrent=false 默认路径） ----------
torch.manual_seed(7)
state = torch.randn(8, 18)
front = torch.rand(8, 3, 128, 128)
wrist = torch.rand(8, 3, 128, 128)
obs = {"observation.state": state, "observation.images.front": front, "observation.images.wrist": wrist}
with torch.no_grad():
    actions, log_probs, means = policy.actor(obs, None)
torch.save(
    {"actions": actions, "log_probs": log_probs, "means": means, "state": state, "obs_shapes": {k: tuple(v.shape) for k, v in obs.items()}},
    PREFIX + ".fwd.pt",
)

# ---------- 2) SACAlgorithm.update 两步 loss 有限 ----------
algo = make_algorithm(cfg.algorithm, policy=policy)
buf = ReplayBuffer(
    capacity=2000,
    device="cpu",
    storage_device="cpu",
    state_keys=list(cfg.policy.input_features.keys()),
    optimize_memory=True,
    use_drq=False,
)
torch.manual_seed(11)
for i in range(400):
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
algo.make_optimizers_and_scheduler()  # RLTrainer.__init__ 里做，这里显式调用
# 用同步迭代器（async_prefetch=False）保证采样批次可复现：
# 异步预取线程的采样时序不确定，导致跨进程 loss 不可比（与代码改动无关）。
torch.manual_seed(11)
it = algo.configure_data_iterator(mixer, batch_size=64, async_prefetch=False, queue_size=2)
losses = []
for _ in range(2):
    stats = algo.update(it)
    losses.append({k: v for k, v in stats.losses.items()})
    print("step losses:", losses[-1])

for d in losses:
    for k, v in d.items():
        assert torch.isfinite(torch.tensor(v)), f"non-finite loss {k}={v}"
with open(PREFIX + ".loss.json", "w") as f:
    json.dump(losses, f, indent=1)
print("ZR OK ->", PREFIX)
