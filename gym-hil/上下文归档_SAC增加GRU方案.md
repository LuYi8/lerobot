# 上下文归档：SAC 增加 GRU 方案（R-SAC）（2026-08-13）

> 本文件用于跨对话窗口续接。实施前先读本文件 + `AGENTS.md`，从"下一步待办"（第 8 节）继续。
> 相关会话任务：以最小改动量、最大兼容性在当前 SAC 算法中增加 GRU，提高抓方块成功率。
> 方案已与用户逐条确认（第 1 节三项决策），**尚未实施**；本文档为完整实施蓝图。

## 0. 环境信息（沙箱验证用）

- 仓库：`/home/embody/lerobot`（LeRobot fork，gym-hil HIL-SAC，分支 `原始SAC`）
- conda 环境：**`lero6`**（python 位于 `/home/embody/miniconda3/envs/lero6/bin/python`，torch 2.11.0+cu126）
- 沙箱限制：torchcodec 视频解码不可用（缺 ffmpeg 库），读数据集视频必须传 `video_backend="pyav"`；沙箱无 GPU，验证权重时把 `cfg.policy.device` 改 `"cpu"`
- 跑验证脚本：`cd /home/embody/lerobot && /home/embody/miniconda3/envs/lero6/bin/python - <<'EOF' ... EOF`（记得 `sys.path.insert(0, "src")`）
- 上次成功训练完整输出：`gym-hil/output好/`（checkpoints 001000~005000，`last` 符号链接可用），用于零回归对比
- 相关归档：`gym-hil/上下文归档_SAC审查与修复.md`（已修复 Bug 1/3/4/5；checkpoint 保存阻塞未修）

## 1. 需求与已确认决策

**需求**：最小改动量、最大兼容性，在当前 SAC（gym-hil HIL-SAC，Panda 抓方块）中增加 GRU，提高算法成功率。

**决策（2026-08-13 已与用户确认）**：

| # | 决策 | 理由 |
|---|---|---|
| 1 | **仅 actor 循环**：GRU 插在 encoder 与高斯头之间（逐时间步循环，替代 MLP 前馈段，见 §3.3）；critic 保持前馈 | 改动面最小（TD 目标、Q 计算逻辑基本不动）；策略从图像/状态历史推断目标运动 → 动作更平滑、成功率提升 |
| 2 | **默认关闭 + CLI 覆盖**：`use_recurrent=false` + `sequence_length=1` 为默认 | 行为逐位不变，旧 checkpoint/resume/gRPC 完全兼容；实验时 CLI 覆盖开启，json 不动 |
| 3 | **计算量不增加**：mixer 采样 batch 缩放 `B = batch_size // T` | 每批帧数 `B×T ≤ batch_size` 恒定（utd 不变 → 每优化步总帧数 = batch_size×utd，与现状完全相同）；GRU(256) 相对 frozen ResNet10 编码额外开销 <5% |

## 2. 设计要点

- **数据流约定（开启后）**：batch 形状 `(B, T, ...)`，T=`sequence_length`；`Policy.forward` 统一收 `(B, T, ...)`（推理单帧 T=1），返回 `(actions, log_probs, means)` 形状 `(B*T, A)` —— 与 critic 展平批对齐，**3 元组返回签名不变**，现有调用点（sac_algorithm 三处、select_action）签名不动。
- **hidden 生命周期**：
  - 训练：`hidden=None, done=done` → 零初始化、done 帧处清零、不保存；**掩码时序固定为"处理完第 t 帧后"执行 `hidden *= (1 - done[:, t])`**，使 done[t]=True 时第 t+1 帧（新 episode 首帧）零历史；**done 必须与被处理序列同帧对齐**（next 序列传左移 1 位的 done，见 §3.6）；
  - 推理：`update_internal_hidden=True` → 用/更新 `Policy._hidden`（首帧零），`select_action` 签名不变；
  - episode 边界：`policy.reset()` 清 `_hidden`（`GaussianActorPolicy.reset()` 现为 pass，改转调 `actor.reset()`）。
- **计算量恒等**：`SACAlgorithm.configure_data_iterator` 中 `batch_size_eff = max(1, batch_size // sequence_length)`，再透传 `sequence_length` 给 mixer；`B×T ≤ 64` 恒成立。
- **权重大传递**：`gru.*` 随 `policy.actor.state_dict()` 自动走 gRPC（`get_weights`/`load_weights`）与 checkpoint（policy save_pretrained），算法侧 `state_dict`/`load_state_dict` **零改动**。
- **episode 边界在训练中不新增 episode_ends 写入**（避免死代码）：序列采样允许窗口跨 episode，靠已正确存储的 `dones`/`truncateds` + 双兜底（TD 掩码逐字沿用现有公式 `td_target = r + (1-done)·γ·Q'`；GRU hidden 清零），与现有 `sample()` 的跨 episode 处理同构，且无拒绝采样偏差（拒绝采样会系统性偏好长 episode）。

## 3. 改动清单（按文件；learner/trainer/gym_manipulator **零改动**）

### 3.1 `src/lerobot/policies/gaussian_actor/configuration_gaussian_actor.py`
`PolicyConfig`（:71-76）新增 3 字段（经 `_init_actor` 的 `**asdict(self.config.policy_kwargs)` 自动流入 `Policy.__init__`）：
```python
use_recurrent: bool = False
recurrent_hidden_size: int = 256
recurrent_num_layers: int = 1
```

### 3.2 `src/lerobot/rl/algorithms/sac/configuration_sac.py`
`SACAlgorithmConfig` 新增 `sequence_length: int = 1`（1 = 原采样行为）。

### 3.3 `src/lerobot/policies/gaussian_actor/modeling_gaussian_actor.py`
- `Policy.__init__`（:402-450）：新增参数 `use_recurrent/recurrent_hidden_size/recurrent_num_layers`；use_recurrent 时创建
  `self.gru = nn.GRU(encoder.output_dim, recurrent_hidden_size, recurrent_num_layers, batch_first=True)`；`self._hidden = None`。
- `Policy.forward(observations, observation_features=None, hidden=None, done=None, update_internal_hidden=False)`（:452-495）：
  - `use_recurrent=False`：走原逻辑（**零回归路径**）；
  - `use_recurrent=True`：输入各键 `(B, T, ...)` → 展平 `(B*T, ...)` → encoder（frozen 缓存特征**保持展平态 `(B*T, C', H', W')` 原样传入**，无需还原 T 维）→ `(B*T, D)` → reshape `(B, T, D)` → 逐时间步 GRU 循环；掩码时序固定为"处理完第 t 帧后"执行 `hidden = hidden * (1 - done[:, t].view(1, B, 1))`（使 done[t]=True 时第 t+1 帧零历史，新 episode 首帧正确；掩码在进入帧前执行则整条链错位 1 帧）→ 输出 reshape `(B*T, D)` → 原 mean/std 头（含 Bug 3 修复的 log 空间 clamp）→ 返回 `(actions, log_probs, means)` 形状 `(B*T, A)`；
  - hidden 语义：`hidden=None` 且 `update_internal_hidden=False`（训练）→ 零初始化、不保存；`hidden=None` 且 `update_internal_hidden=True`（推理）→ 用 `self._hidden`、结束后 `self._hidden = hidden.detach()`；
  - 注意：`GaussianActorPolicy.forward`（:112）调 actor 时不带 hidden 参数（默认 `hidden=None, update_internal_hidden=False`）→ 连续推理不会保持 hidden；训练侧无影响，仅提示未来调用方不要误用。
- 新增 `Policy.reset()`：`self._hidden = None`。
- `GaussianActorPolicy.reset()`（:66-68，现为 `pass`）：改调 `self.actor.reset()`。
- `GaussianActorPolicy.select_action`（:77-97）：单帧输入各键 `(B, ...)` → **先在原 4D batch 上算 `get_cached_image_features`**（:83 现有行，shared_encoder 且 has_images 时真实执行，本仓库 resnet10 非 None 必走）→ 再 unsqueeze observations 成 `(B, 1, ...)` → `self.actor(batch_unsq, feats_4d, update_internal_hidden=True)`（forward 内部展平回 `(B, ...)` 与 4D 特征自然对齐；**顺序颠倒会把 5D 图像喂给 ResNet10 崩溃**）→ 结果 squeeze 回 `(B, A)`；`num_discrete_actions` 拼接逻辑不动。**上述 unsqueeze/update_internal_hidden 路径仅在 `use_recurrent=true` 时走；`use_recurrent=false` 保持现有三行原逻辑不变**（否则 4D 特征配 `(B,1,...)` 观测会让 state 编码器输出 3D 张量，与图像特征 cat 维度不匹配直接崩溃，违反决策 2 零回归）。

### 3.4 `src/lerobot/rl/buffer.py`
- `sample(batch_size, sequence_length=1)`（:234-300）：`sequence_length>1` 时走新 `_sample_sequences(batch_size, T)`（正确性论证见第 4 节）；否则原逻辑。
- `get_iterator`/`_get_async_iterator`/`_get_naive_iterator`（:302-408）：加 `sequence_length=1` 参数透传。
- **不新增 `episode_ends` 写入**（:122 已分配数组保持原样，序列采样不依赖它）。

### 3.5 `src/lerobot/rl/data_sources/data_mixer.py`
`DataMixer.sample/get_iterator` 与 `OnlineOfflineMixer.sample/get_iterator` 加 `sequence_length=1` 参数透传给 buffer；`concatenate_batch_transitions`（buffer.py:790）沿 dim0 拼接 `(B, T, ...)` 天然有效，**零改动**。

### 3.6 `src/lerobot/rl/algorithms/sac/sac_algorithm.py`
- override `configure_data_iterator`（基类 base.py:62-79）：`data_mixer.get_iterator(batch_size=max(1, batch_size // self.config.sequence_length), sequence_length=self.config.sequence_length, ...)`。
- `_prepare_forward_batch`（:460-479）：sequence_length>1 时**先把 batch 展平再做后续**——state/next_state 各键 `(B,T,...) → (B*T,...)`、action `(B,T,A) → (B*T,A)`、reward/done `(B,T) → (B*T,)`，并记录 `self._seq_shape = (B, T)`。**顺序硬性要求：展平必须先于 `get_observation_features`（:465，函数第一句）**——否则 `get_cached_image_features` 对 `(B,T,C,H,W)` 图像键 cat 出 5D 张量喂给 ResNet10 直接崩溃；展平后 cat 得 4D 正常，缓存特征图即 `(B*T, C', H', W')` 展平态（critic 路径零改动）。
- `_compute_loss_critic`（:277-350）：`next_action_preds` 调用处（:289）还原 next_state 的 `(B,T,...)` 视图 + next features（**保持 `(B*T, C', H', W')` 展平态即可，无需还原视图**，与 forward 内部展平后的 observations 天然对齐），调 `self.policy.actor(..., hidden=None, done=done_next)` → `(B*T, A)` 直接可用。**关键：next 序列比 observations 右移 1 帧（pos idx+1..idx+T），done 必须左移 1 位再传**——`done_next = cat([done[:, 1:], zeros(B,1)], dim=1)`（末位无后续帧、值随意）；否则掩码早 1 帧（静默偏差）：跨 episode 窗口时新 episode 首帧（next 侧）被旧 episode 历史污染（其 td 恰被 `(1-done)` 掩码，损失主项不受影响）、边界后第二帧丢失首帧上下文。td_target 本身仍用原 observations 对齐的 `done`（:313 公式零改动）。
- `_compute_loss_actor`（:411-426）/`_compute_loss_temperature`（:428-438）：同样还原序列视图调 actor，**并同样传 `done=done`（observations 对齐，掩码时序见 §3.3）**——否则跨 episode 窗口会把上一 episode 的历史带进 log_probs/Q，而这两处没有 TD 掩码兜底，污染直接进梯度；`log_probs (B*T,)` 直接用。
- `get_weights`/`load_weights`/`state_dict`/`load_state_dict`：**零改动**。

### 3.7 `src/lerobot/rl/actor.py` / `src/lerobot/rl/eval_simple.py`
- `actor.py`：episode 结束 `reset_and_build_transition`（:431）后加一行 `policy.reset()`。顺序正确：先 `update_policy_parameters`（:395，拉新权重）后 reset（新权重 + 干净 hidden）。
- `eval_simple.py`：每 episode 循环开始（:70 `reset_and_build_transition` 后）加一行 `policy.reset()`。

### 3.8 文档
`AGENTS.md` 与 `gym-hil/命令.txt` 补 GRU 实验小节：开关参数、actor/learner 两侧配置一致性要求、实验命令（第 5 节）。

## 4. Buffer 序列采样正确性论证（对照现有 `sample()` 逐条推演）

> 现有 `sample()` 关键行（buffer.py:241-243）：`high = max(0, size-1) if optimize_memory and size < capacity else size`；`idx = randint(0, high)`（**半开区间**）→ 未满时 idx ≤ size-2 保证 `next_idx = (idx+1)%capacity ≤ size-1` 已写。序列版必须继承同一约束。

1. **索引范围（未满时）**：`high = size - T` → `idx ≤ size-T-1` → 窗口 `[idx, idx+T)` 全在已写区（0..size-1），且窗口末帧（位置 idx+T-1）的 next（位置 idx+T）≤ size-1 已写。**关键：排除"窗口末帧即 buffer 末帧且 done=False"的情形**（否则 `states[idx+T]` 是 `torch.empty` 未初始化内存，TD 无掩码 → 污染）。
2. **满时（环形）**：`high = capacity - T + 1` → `idx+T-1 ≤ capacity-1` 全位置已写；末帧 next 用 `(idx+T) % capacity` 环形索引（与现有 `next_idx=(idx+1)%capacity` 同语义）。
3. **episode 边界**：允许窗口跨 episode，双兜底——(a) TD：done 帧 bootstrap 被掩码（公式逐字沿用现有 :313）；(b) GRU：done 帧处 hidden 清零（新 episode 零历史）。唯一"next_state 无意义"的情形是窗口末帧 done=True → 已被 (a) 掩码。**正确且无拒绝采样偏差**。
4. **切片对齐**：`states[key][i:i+T]` 与 next 帧 `states[key][i+1:i+T+1]`（或环形 `(i+1..i+T) % capacity`）同一数组错位 1 → 保证 `next_state_t ≡ state_{t+1}` 逐位一致（optimize_memory 语义，与现有 next_idx 一致）。
5. **形状与杂项**：各键 → `(B, T, ...)`；reward/done/truncated/complementary_info → `(B, T)`；DRQ 图像增强：`(B,T,C,H,W)` 展平 `(B*T,C,H,W)` 增强后还原（与现有"拼接 state+next_state 一起增强"同构）；`size < T` 时 raise 清晰错误（在线预热 `online_step_before_learning=100` 远大于 T=8，不触发）。

## 5. 实验参数（CLI 覆盖，json 不动；计算量恒等于现状）

```bash
# learner（train_hil_env.json 基础上追加；actor_hil_env.json 同样追加）
python -m lerobot.rl.learner --config_path gym-hil/train_hil_env.json --resume false \
  --policy.policy_kwargs.use_recurrent true \
  --policy.policy_kwargs.recurrent_hidden_size 256 \
  --policy.policy_kwargs.recurrent_num_layers 1 \
  --algorithm.sequence_length 8 2>&1 | tee full_trace.log

# actor（参数必须与 learner 一致）
python -m lerobot.rl.actor --config_path gym-hil/actor_hil_env.json \
  --policy.policy_kwargs.use_recurrent true \
  --policy.policy_kwargs.recurrent_hidden_size 256 \
  --policy.policy_kwargs.recurrent_num_layers 1 \
  --algorithm.sequence_length 8 2>&1 | tee actor_trace.log
```

| sequence_length | batch_size_eff | 每批帧数 | GRU 历史窗口（fps=10） |
|---|---|---|---|
| 8（推荐） | 8 | 64 | 0.8s |
| 16 | 4 | 64 | 1.6s（batch=4 梯度噪声偏大） |
| 4 | 16 | 64 | 0.4s（历史偏短） |

- `utd_ratio=2` 不变；`batch_size=64` 不变 → 每优化步 128 帧，与现状相同。
- 从头训练前删 `gym-hil/output` 与 `gym-hil/output_actor`；先 learner 后 actor。
- 预期收益：连续动作更平滑（图像流推断目标速度），干预率下降、成功率提升；参考基线：现算法约 6000 优化步 60% 成功率。

## 6. 验证清单（实施后执行）

1. `py_compile` 全链（3.1-3.7 改到的 7 个文件）。
2. **Buffer 正确性测试（沙箱 CPU，构造已知数据：2 段 episode 各 10 帧，done 在 9/19）**：
   - 越界：T=4 大量采样，断言窗口全在已写区、`idx+T ≤ size`（未满时）；
   - 对齐：窗口内每帧 `next_state_t == state_{t+1}` 逐位成立；
   - 跨 episode：T 覆盖边界 → done 帧位置 TD 掩码生效、GRU hidden 边界后归零；
   - **done 对齐（§3.6 掩码错位回归）**：构造 done 在窗口中间的已知数据，跑 `_compute_loss_critic` 的 next-action GRU 前向，断言新 episode 首帧（next 侧）的 log_prob 与"零 hidden 基线"一致（即左移 done 生效；若误传未左移的 done 该断言失败）；
   - 环形（size==capacity）：idx 范围、wrap 窗口、`(idx+T)%cap` next 索引正确；
   - 保护：`size < T` 抛错；DRQ 增强后 shape `(B,T,C,H,W)` 还原；
   - 计算量断言：`B×T == batch_size`（64）。
3. **零回归**：`use_recurrent=false` 加载 `gym-hil/output好` checkpoint（`pretrained_model` 目录），`Policy.forward` 输出与改动前逐位一致；`SACAlgorithm.update` 1-2 步 loss 有限。
4. **GRU 链路**：`use_recurrent=true` 小 buffer 上 `update` 不崩、loss 有限；`select_action` 连续两帧 hidden 传递（输出变化连续）、`reset()` 后行为复位。
5. 真机实验：双侧同参 CLI 开启，从头训练，对比成功率/干预率，结果记录回本文档。

## 7. 风险与注意

- GRU 逐时间步 python 循环（T=8）训练开销小；后续可选优化：torch.compile actor 或整序列单次 GRU（done 处只掩码损失）。
- resume 兼容：开 GRU 的 checkpoint 加载时两侧 `use_recurrent` 必须同值（strict 加载缺键会自然暴露）；默认关的旧流程完全兼容（无 `gru.*` 键）。
- `to_lerobot_dataset`/checkpoint 保存无需改（buffer 仍按帧存储）。
- 序列采样要求 `buffer.size ≥ T`（启动保护 raise；预热 100 步不触发）。
- actor/learner 两侧 `policy.policy_kwargs`、`algorithm.sequence_length` 必须一致（AGENTS.md 既有约定）。
- 若以后启用离散夹爪（Bug 2），离散 critic 前馈不受 GRU 影响，但 `select_action` 拼接逻辑需回归测试。

## 8. 下一步待办（按优先级）

1. 按第 3 节实施代码改动（全部带 `#修改 ... #结束` / `#===` 标记）。
2. 沙箱执行第 6 节验证 1-4 项，通过后提交（中文提交信息，参照历史风格）。
3. 真机实验：第 5 节命令开启 GRU 从头训练，对比成功率/干预率。
4. 实验结论（成功/失败/调参）记录回本文档第 5 节，更新 AGENTS.md 与 `gym-hil/命令.txt`。

## 9. 仓库约定提醒（详见 AGENTS.md）

- 本地改动用 `#修改 ... #结束` / `#===` 注释标记（搜 `#修改` 可定位全部改动点）。
- 分支名/提交信息用中文；提交信息用分号列出改动要点。
- 先 learner 后 actor；从头训练前删 `gym-hil/output` 和 `gym-hil/output_actor`。
- resume 必须用 checkpoint 内 `train_config.json` + `--resume true`。
- 修改 `rl/` 或 `policies/gaussian_actor/` 时需同时考虑 actor/learner 两侧一致性。
- wandb project=`hil_test`；训练/评估命令以 `gym-hil/命令.txt` 与 AGENTS.md 为准。
