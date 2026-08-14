# 上下文归档：SAC 增加 critic GRU 方案（R-SAC + critic 循环）（2026-08-14）

> 本文件用于跨对话窗口续接。实施前先读本文件 + `AGENTS.md` + 第一篇《上下文归档_SAC增加GRU方案.md》（R-SAC 实施蓝图），从"下一步待办"（第 8 节）继续。
> 相关会话任务：以最小改动量、最大兼容性，在当前 R-SAC（仅 actor 循环，2026-08-13 已实施并真机验证：14000 步成功率 99% vs 纯 SAC 45%）的 critic 中增加 GRU，进一步提高抓方块成功率。
> 方案已与用户逐条确认（第 1 节决策 1"独立开关"、决策 2"GRU 纯观测输入 + 动作在头"、决策 7"离线 8500 / 在线 4000"均为 2026-08-14 用户确认；决策 7 当日修订：原"离线 6000"装不下扩采后的演示数据集 8252 帧，`from_lerobot_dataset` 容量不足直接 ValueError，离线容量提到 8500），**尚未实施**；本文档为完整实施蓝图。
> 数据管线事实（见 §0 与 §4.5）：历史 99% 运行（GRUoutput，2026-08-13）为 在线 8000 / 离线 2000 + 582 帧演示；当前 json（560bd5d 起）为 在线 1500 / 离线 8500 + 8252 帧演示（150 集）；离线 buffer 内容 = 演示数据集 + 仅干预 transition（100% 人类动作）。

## 0. 环境信息（沙箱验证用）

- 仓库：`/home/embody/lerobot`（LeRobot fork，gym-hil HIL-SAC，分支 `原始SAC`）
- conda 环境：**`lero6`**（python 位于 `/home/embody/miniconda3/envs/lero6/bin/python`，torch 2.11.0+cu126）
- 沙箱限制：torchcodec 视频解码不可用（缺 ffmpeg 库），读数据集视频必须传 `video_backend="pyav"`；沙箱无 GPU，验证权重时把 `cfg.policy.device` 改 `"cpu"`，且 `use_torch_compile=false`（CPU 沙箱下 dynamo 无法对 frozen ResNet10 做 fake-tensor 推断）
- 跑验证脚本：`cd /home/embody/lerobot && /home/embody/miniconda3/envs/lero6/bin/python - <<'EOF' ... EOF`（记得 `sys.path.insert(0, "src")`）
- 上次成功训练完整输出：`gym-hil/output好/`（纯 SAC，checkpoints 001000~005000）、`gym-hil/GRUoutput/`（R-SAC 真机实验，014000 步 99%），均用于零回归对比与基线
- **数据管线演变**：
  - 历史 99% 运行（GRUoutput，2026-08-13）：当时 json 为 在线 8000 / 离线 2000、演示 ~582 帧；**实测填充时序（用户 2026-08-14 提供）**：离线 buffer 在总交互步 ~2800 满（容量 2000 + 582 演示 → 前 2800 步干预率 ≈ 50%；**演示数据在 ~4000 总步前被环形淘汰**）；在线 buffer 在 ~8100 总步满（8000 容量 + 预热 100，1:1 吻合）。
  - 当前 json（560bd5d 起，2026-08-14）：`online_buffer_capacity=1500`、`offline_buffer_capacity=8500`，演示数据集扩至 **150 集 / 8252 帧**（离线容量 8500 恰好容纳演示全集 + 248 帧干预余位）。
  - 通用（两侧一致）：`online_ratio=0.5` → 每批 50% 来自在线 buffer、50% 来自离线 buffer（`OnlineOfflineMixer.sample`，n_online = int(batch×0.5)）；在线 buffer = **全部**在线数据——自动 + 干预，最近窗口（learner.py:1013 `replay_buffer.add(**transition)` 无条件）；离线 buffer = 演示数据集 seed（make_dataset，人类遥操作，learner.py:894-900）+ **仅干预** transition（learner.py:1016-1018 `if IS_INTERVENTION: offline_replay_buffer.add(...)`）；内容 **100% 人类动作**）。量化分析见 §4.5
  - 环境观测：`observation.state` 18 维 = qpos(7)+qvel(7)+gripper(1)+tcp_pos(3)（gym-hil/gym_hil/mujoco_gym_env.py:260-268），**不含方块位置**；方块位姿只出现在 front/wrist 图像（128×128 @10fps）→ 环境真部分可观测，历史信息有价值
- 相关归档：第一篇《上下文归档_SAC增加GRU方案.md》（R-SAC 蓝图，已实施；`_NonFlatteningGRU` 修复记录见其 §10）、`gym-hil/上下文归档_SAC审查与修复.md`（Bug 1/3/4/5 已修；2/6 待修）
- 真机实验基线（2026-08-14，commit 6c9b7e9）：R-SAC（actor GRU）14000 步 99% vs 纯 SAC 45%，同步数稳定领先 25~54 个百分点

## 1. 需求与已确认决策

**需求**：最小改动量、最大兼容性，在当前 R-SAC（gym-hil HIL-SAC，Panda 抓方块）的 critic 中增加 GRU，提高算法成功率。

**决策（2026-08-14 已与用户确认）**：

| # | 决策 | 理由 |
|---|---|---|
| 1 | **独立开关**：新增 `SACAlgorithmConfig.critic_use_recurrent: bool = False`，与 `policy_kwargs.use_recurrent` 相互独立，可自由组合（全关 / 仅 actor=现有 R-SAC / 仅 critic=消融 / 双关=目标实验） | 用户确认；默认关闭行为逐位不变；可单独做 critic 循环消融（隔离其贡献）。放 algorithm 侧的理由：critic 归算法所有（与 `sequence_length` 同侧、对称）；`policy_kwargs` 经 `_init_actor` 的 `**asdict` 流入 `Policy.__init__`，加字段需改其签名（TypeError） |
| 2 | **GRU 输入 = 纯观测（obs_enc），动作在 GRU 之后、Q 头之前拼入**（head 输入 = `cat([gru_out, action])` = 256+4）；GRU 插在 CriticEnsemble 共享 encoder 与 Q1/Q2 头之间，输出维 = `recurrent_hidden_size` | **消除 pred/target 历史分布问题**（详见 §4.4）：历史侧纯观测（与 actor GRU 同构），动作只作单步查询——pred 用行为动作、target 用策略动作，与现有前馈 critic 的 off-policy 语义完全一致，不新增任何近似；实现更简（critic GRU 构造 = actor GRU 逐字拷贝，无 gru_input_dim、无 target 末位填充问题）。"动作驱动动力学"代价的量化评估见 §4.4 |
| 3 | **复用 `recurrent_hidden_size`(256)/`recurrent_num_layers`(1)，不新增 size 字段；复用 `_NonFlatteningGRU`** | 最小改动；防 CUDA 上 cuDNN 把 RNN 权重原地展平成共享 storage 的 view、checkpoint 存 safetensors 直接 RuntimeError（e4ac49b 修复，直接复用） |
| 4 | **done 掩码时序镜像 actor**："处理完第 t 帧后 `h *= (1 - done[:, t])`"；pred Q 传 observations 对齐的 done、target Q 传左移 1 位的 `done_next`、actor loss 的 Q 传 observations 对齐的 done | 与 actor 完全一致（含第一篇 §3.6 的"done 左移"教训）；td_target 公式零改动 |
| 5 | **配置校验（learner 侧 ValueError）**：`critic_use_recurrent=true` 与 `use_recurrent=true` 均要求 `sequence_length>1` | critic GRU 依赖序列采样；`use_recurrent=true + seq=1` 目前是静默垃圾形状（潜在雷，顺带修复） |
| 6 | **actor 前馈 else 分支加 ~6 行序列展平** | 决策 1 的前置：`use_recurrent=false + seq>1` 时 loss 函数传 `(B, T, ...)` 视图，需展平 `(B*T, ...)` 再进 encoder（否则 5D 图像喂 ResNet10 崩溃）；seq=1 时检测不命中，**逐位不变** |
| 7 | **消融实验数据管线 = 离线 8500 / 在线 4000**（2026-08-14 修订：原 6000 装不下扩采后的 8252 帧演示，`from_lerobot_dataset` 容量不足直接 ValueError → 离线提到 8500；总帧数 12500，较当前 10000 内存 +25%）；**同一新配置下重跑纯 SAC / R-SAC 基线**做公平消融；历史 99% 基线（2000/8000）仅作参考 | 2026-08-14 用户确认；离线池 = 演示全集 103 集 + 仅 248 帧干预余位（干预超限后从最老演示帧开始环形淘汰）；在线池 4000 步 = 50 集 = 6.7 分钟（比历史基线 8000 更新鲜、比当前 1500 更陈旧，给离线池腾容量）；量化分析见 §4.5；代价：checkpoint 保存 dataset_offline 写盘量 ×4.25（vs 历史 2000；见 §8 待办 1） |

## 2. 设计要点

- **数据流约定（critic_use_recurrent=true 时）**：buffer/mixer 序列采样 `(B, T, ...)`（现成，第一篇 §3.4/3.5）→ `_prepare_forward_batch` 展平 `(B*T, ...)`（现成，第一篇 §3.6）→ **critic 调用点**（`_compute_loss_critic` 的 q_preds/q_targets、`_compute_loss_actor` 的 Q 共 3 处）在开关开启时还原 `(B, T, ...)` 视图 + 传 done；`CriticEnsemble.forward` 内部再展平进 encoder（与冻结缓存特征 `(B*T, C', H', W')` 展平态天然对齐）→ `view(B, T, -1)` → 逐时间步 GRU 循环（**输入纯 obs_enc，不含动作**）→ `reshape(B*T, -1)` → 每帧拼上该帧动作 → 各头 → 返回 `(num_critics, B*T)`（**返回契约不变**）。
- **前向伪代码**（GRU 分支镜像 modeling_gaussian_actor.py `Policy.forward`；头输入与 actor 的区别仅在拼接动作）：
```python
if self.use_recurrent:
    first_key = next(iter(observations))
    B, T = observations[first_key].shape[:2]
    obs_flat = {k: v.reshape(B * T, *v.shape[2:]) for k, v in observations.items()}
    obs_enc = self.encoder(obs_flat, cache=observation_features)   # 不 detach（与现 critic 路径一致）
    obs_enc = obs_enc.view(B, T, -1)
    actions = actions.reshape(B, T, -1)   # 一律 reshape 不用 view（防御非连续输入，见 §3.3 注意）
    h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size,
                    device=obs_enc.device, dtype=obs_enc.dtype)    # 训练零初始化、不保存
    done_seq = done.view(B, T) if done is not None else None
    outputs = []
    for t in range(T):
        out_t, h = self.gru(obs_enc[:, t : t + 1, :], h)          # 纯观测循环（与 actor 相同）
        outputs.append(out_t)
        if done_seq is not None:
            # 掩码时序固定为"处理完第 t 帧后"执行（与 actor 完全一致）：
            # done[t]=True 时第 t+1 帧（新 episode 首帧）零历史
            h = h * (1 - done_seq[:, t].view(1, B, 1))
    obs_enc = torch.cat(outputs, dim=1).reshape(B * T, -1)
    inputs = torch.cat([obs_enc, actions.reshape(B * T, -1)], dim=-1)  # 动作在头：256+4
else:
    # 零回归路径：与改动前逐位一致
    obs_enc = self.encoder(observations, cache=observation_features)
    inputs = torch.cat([obs_enc, actions], dim=-1)
q_values = [critic(inputs) for critic in self.critics]
q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)   # (num_critics, B*T)
```
- **动作角色（决策 2 的核心语义）**：动作只作**单步查询**——pred 用行为动作 a^β_t（现有 batch[ACTION]）、target 用策略动作 a^π_{t+1}（现有 next_action_preds）、actor loss 用 a^π_t（现有 actions_pi）。历史（GRU 隐藏态）完全由观测承载，pred/target 同流形，**无历史分布问题**（§4.4）。
- **target 对齐**：`done_next = cat([done_seq[:, 1:], zeros(B, 1)], dim=1)`（现成局部变量，第一篇 §3.6 已实现）；target critic 的 GRU 掩码用 `done_next`，td_target 仍用 observations 对齐的 done（公式零改动）。target 的 next-obs 序列全部已知（帧 idx+1..idx+T 的观测都在 batch 内）→ **无末位未知动作填充问题**（A3 才有此问题，见 §4.4）。
- **hidden 生命周期**：critic 仅在训练中使用（learner 侧），每序列零初始化、不保存；**critic 无推理路径**（select_action/eval 只用 actor 网络）→ 不需要 `_hidden` 属性、不需要 `reset()`、actor.py/eval_simple.py 零改动。
- **target 创建/更新**：target heads+GRU 为独立实例，`load_state_dict(self.critic_ensemble.state_dict())` 自动复制 GRU 权重；`_update_target_networks` 的 EMA `zip(..., strict=True)` 遍历全部参数，GRU 参数自动纳入，**零改动**。
- **checkpoint**：`gru.*` 键（`critic_ensemble.gru.weight_ih_l0` 等 8 个 ×2 网络）随算法 `state_dict`/`load_state_dict` 自动走（`_strip_encoder_keys` 只剥 `encoder.` 前缀，gru 键自然保留；learner.py:613 `algorithm.save_pretrained` 落盘）。
- **计算量**：数据吞吐不变（`B×T ≤ 64` 恒定，机制现成）；GRU 参数 ≈ `4·h·(d_in+h)` = 4×256×(192+256) ≈ **46 万/网络**（d_in = encoder.output_dim 192，纯观测），online+target 两份；critic MLP 头输入 260（256+4）相对现 196 略增；相对整体策略占比小。

## 3. 改动清单（按文件；仅 3 个文件，learner/trainer/buffer/data_mixer/actor/eval_simple **零改动**）

### 3.1 `src/lerobot/rl/algorithms/sac/configuration_sac.py`
`SACAlgorithmConfig`（`sequence_length` 之后，:90-96 同块）新增 1 字段：
```python
#修改 ============ critic 循环（critic_use_recurrent，独立开关） ============
# critic_use_recurrent=true 时 critic 在共享 encoder 与 Q1/Q2 头之间插入 GRU。
# GRU 输入 = 纯观测（obs_enc，与 actor GRU 同构），动作在 Q 头前拼入
# （head 输入 = cat([gru_out, action])）——历史侧无动作，pred/target 同流形，
# 不存在行为/策略动作历史分布问题；动作只作单步查询（行为/策略动作语义与
# 现有前馈 critic 完全一致）。输出维 = policy_kwargs.recurrent_hidden_size。
# 与 policy_kwargs.use_recurrent 相互独立，可自由组合：
#   全关 = 上游逐位一致；仅 actor = 现有 R-SAC；仅 critic = 消融；双关 = 目标实验。
# 默认关闭，critic 前馈行为逐位不变（旧 checkpoint/resume 完全兼容）。
critic_use_recurrent: bool = False
#结束 ============================================
```

### 3.2 `src/lerobot/policies/gaussian_actor/modeling_gaussian_actor.py`
`Policy.forward` 的 else（前馈）分支（:553-555）加序列展平（决策 1 的前置）：
```python
else:
    # 零回归路径：与改动前逐位一致（输入为展平 (B*T, ...) 时）
    # 序列支持（critic_use_recurrent 独立开关的前置）：use_recurrent=false 但
    # sequence_length>1 时，loss 函数传 (B, T, ...) 视图——先展平 (B*T, ...)
    # 再进 encoder（输出仍 (B*T, ...)，与调用点期望一致）；seq=1 时检测不命中。
    first_key = next(iter(observations))
    if observations[first_key].ndim in (5, 3):  # 图像 (B,T,C,H,W) / 状态 (B,T,D)
        B, T = observations[first_key].shape[:2]
        observations = {k: v.reshape(B * T, *v.shape[2:]) for k, v in observations.items()}
    obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
```

### 3.3 `src/lerobot/rl/algorithms/sac/sac_algorithm.py`（核心改动）
- 导入 `_NonFlatteningGRU`（modeling_gaussian_actor.py:426-436 已有）。
- `__init__`（:57-73）加配置校验：
```python
#修改 ============ GRU 配置校验 ============
if self.config.critic_use_recurrent and self.config.sequence_length <= 1:
    raise ValueError("critic_use_recurrent=true 要求 algorithm.sequence_length>1（critic GRU 依赖序列采样）")
if self.policy_config.policy_kwargs.use_recurrent and self.config.sequence_length <= 1:
    raise ValueError("use_recurrent=true 要求 algorithm.sequence_length>1（actor GRU 训练依赖序列采样；seq=1 时 batch 为展平态，GRU 分支形状错乱）")
#结束 ============================================
```
- `_init_critics`（:75-95）：头输入维三目 + CriticEnsemble 传 GRU 参数（online/target 同参）：
```python
critic_input_dim = encoder.output_dim + action_dim
use_rec = self.config.critic_use_recurrent
rec_kwargs = (
    dict(use_recurrent=True,
         recurrent_hidden_size=self.policy_config.policy_kwargs.recurrent_hidden_size,
         recurrent_num_layers=self.policy_config.policy_kwargs.recurrent_num_layers)
    if use_rec else {}
)
head_input_dim = (
    self.policy_config.policy_kwargs.recurrent_hidden_size + action_dim  # 256+4，动作在头
    if use_rec else critic_input_dim
)
heads = [CriticHead(input_dim=head_input_dim, **asdict(self.config.critic_network_kwargs)) for _ in range(self.config.num_critics)]
self.critic_ensemble = CriticEnsemble(encoder=encoder, ensemble=heads, **rec_kwargs)
target_heads = [CriticHead(input_dim=head_input_dim, **asdict(self.config.critic_network_kwargs)) for _ in range(self.config.num_critics)]
self.critic_target = CriticEnsemble(encoder=encoder, ensemble=target_heads, **rec_kwargs)
self.critic_target.load_state_dict(self.critic_ensemble.state_dict())  # GRU 权重自动复制
```
- `CriticEnsemble`（:732-777）：构造可选参数 `use_recurrent=False / recurrent_hidden_size / recurrent_num_layers`（默认关闭 → 零回归）；use_recurrent 时建 `self.gru = _NonFlatteningGRU(encoder.output_dim, recurrent_hidden_size, recurrent_num_layers, batch_first=True)`（**与 actor GRU 逐字同构，无 gru_input_dim**）；`forward(..., done=None)` 加 GRU 分支（§2 伪代码），else 分支逐字节不动。
- `_critic_forward`（:148-168）：加 `done: Tensor | None = None` 参数透传。
- `_compute_loss_critic`（:307-400）：seq 分支（:327-337）补建 `obs_view`（现只有 `next_obs_view`）；q_preds（:371-376）与 q_targets（:344-349）在 `critic_use_recurrent` 时改传视图 + done，否则维持展平态：
```python
q_targets = self._critic_forward(
    observations=next_obs_view if self.config.critic_use_recurrent else next_observations,
    actions=next_action_preds.reshape(B, T, -1) if self.config.critic_use_recurrent else next_action_preds,
    use_target=True,
    observation_features=next_observation_features,
    done=done_next if self.config.critic_use_recurrent else None,
)
q_preds = self._critic_forward(
    observations=obs_view if self.config.critic_use_recurrent else observations,
    actions=actions.reshape(B, T, -1) if self.config.critic_use_recurrent else actions,
    use_target=False,
    observation_features=observation_features,
    done=done_seq if self.config.critic_use_recurrent else None,
)
```
  注意：`done_next`/`done_seq` 为现成局部变量（:331-334）；**序列视图一律用 `reshape` 不用 `view`**——`actions` 的离散维截断（:370）后为非连续切片（stride (4,1)），view 直接 RuntimeError（当前 `num_discrete_actions=null` 不触发截断，属防御性写法）；td_target 公式（:363）、2 元组返回契约、q_stats（:389-398）**零改动**。
- `_compute_loss_actor`（:461-488）：Q 调用（:479-484）在开关开启时传 `obs_view + actions_pi.reshape(B, T, -1) + done_view`（现成局部变量，:473-474）。
- `load_state_dict`（:641-663）：开关开启时校验 `gru.*` 键存在，缺失直接报错（防静默随机权重，与 6c9b7e9 的 strict 化方向一致）：
```python
if self.config.critic_use_recurrent:
    missing = [k for k in self.critic_ensemble.state_dict()
               if k.startswith("gru.") and k not in critic_ensemble_state]
    if missing:
        raise RuntimeError(f"critic_use_recurrent=true 但 checkpoint 缺 GRU 权重 {missing}；"
                           "请确认恢复源与配置一致（应使用 checkpoint 内 train_config.json）")
```

### 3.4 零改动清单（明确不改，防范围蔓延）
- `_update_target_networks`（:511-529，EMA zip 自动覆盖 GRU）、`state_dict`/`_strip_encoder_keys`（:624-639，gru 键自然保留）、`make_optimizers_and_scheduler`（:567-600，`critic_ensemble.parameters()` 含 GRU）、`_move_to_device`（:135-141）、`configure_data_iterator`（:187-211）、`_compute_loss_temperature`（:490-509，不调 critic）、`_prepare_forward_batch`（:531-565）。
- **离散 critic**（`num_discrete_actions` 非空时）：保持前馈，不在本次范围（本实验 `num_discrete_actions: null` 未启用）。
- `buffer.py` / `data_mixer.py` / `actor.py` / `eval_simple.py` / `configuration_gaussian_actor.py` / `modeling_gaussian_actor.py` 其余部分：**零改动**（序列采样、done 约定、推理 hidden 均现成；eval_simple 只加载 `actor.*`/`encoder_actor.*` 键，critic 权重不参与评估）。

### 3.5 文档
`AGENTS.md` 与 `命令.txt` 补 critic GRU 小节（第 5 节命令）；本文档 §10 实施后回填。

## 4. 正确性论证

### 4.1 序列化 TD 目标
pred `Q_t = Q(s_t, a_t | h_t)`，h_t 为观测历史（窗口起点前零初始化）；target `r_t + (1-done_t)·γ·min_j Q'_j(s_{t+1}, a_{t+1}~π | h'_{t+1})`。critic 对序列 T 步逐帧出 Q、全部帧参与 MSE（与 actor 逐帧 log_prob 同构）；窗口起点随机 → 模型学会"给定可用历史"估值，与截断 BPTT 同语义。

### 4.2 done 掩码对齐推导（核心，镜像 actor）
- pred 序列 = observations（帧 t 对齐 done[t]）：位置 t 的 hidden 在位置 t-1 处理完后清零 iff `done[t-1]` → done[t-1]=1（t 为新 episode 首帧）时 h_t=0 ✓；
- target 序列 = next_observations（位置 i 对应帧 i+1）：掩码必须用 `done_next[i]=done[i+1]`（左移 1 位）→ 位置 t 的 hidden 清零 iff `done_next[t-1]=done[t]` → done[t]=1 时 h'_{t+1}=0，且该帧 td 同时被 `(1-done[t])` 掩码（**双保险**）✓；
- 不左移的后果（静默偏差）：掩码早 1 帧，新 episode 首帧（next 侧）沿用旧 episode 末帧历史——与第一篇 §3.6 同款错误，验证清单 C 断言覆盖。

### 4.3 跨 episode 窗口
允许窗口跨 episode（buffer 现成行为），双兜底——(a) TD：done 帧 bootstrap 被掩码（公式零改动）；(b) GRU：done 帧处 hidden 清零。无拒绝采样偏差（与 buffer 注释一致）。

### 4.4 pred/target 历史分布问题（决策 2 的核心论证，2026-08-14 重写）
**问题来源（若动作进 GRU 循环，即原 A1 设计）**：动作槽位同时承担"历史"与"当前查询"两个角色，逐帧 TD 下无法两全——target 被迫用**当前策略动作**（next_action_preds）全序列建史，而 pred 用**行为动作**建史。本数据管线让该差异最大化：

- **离线 buffer 内容 100% 是人类动作**（演示数据集 + 干预 transition，learner.py:1014-1016 只写干预），每批 50% 来自它 → 一半批次的 pred 历史是"人类怎么把状态带到这"；
- 干预是**系统性修正**（策略做错时才发生），与策略动作流形高度不重合；
- 在线半区（8000 步 ≈ 13.3 分钟）在训练早期也含分钟级陈旧的策略数据。
- 这与"标准 SAC 的 off-policy 差异"不同：前馈 critic 的差异只有**单步 query 动作**（a_t 行为 vs a_{t+1} 策略，off-policy 学习理论标准处理）；A1 把差异放大到**整条 GRU 历史**（隐藏态内容直接由动作序列决定）。R2D2/DRQ 等循环 off-policy 文献的惯例是行为动作驱动循环（记录实际轨迹）、查询动作单独注入——A1 的"策略动作建史"并非标准做法。

**A2 的消除机制（本方案）**：GRU 输入 = 纯观测 → 历史侧 pred/target **同流形**，不存在任何动作分布问题；动作只作单步查询（行为/策略动作），与现有前馈 critic 完全同语义。**采纳 A2 后，buffer 构成（0.5/0.5、演示+干预）从"正确性风险"降级为"纯数据多样性调参"问题（§4.5）**。

**"动作驱动动力学"代价的量化评估（为什么 obs 流足够承载轨迹）**：
- 环境真部分可观测（方块位姿不在 18 维 state 中，只在图像里）→ 历史信息的价值在于从观测流跟踪方块/推断运动；
- 动作 a_t 的效果**滞后 1 帧（100ms @10fps）进入观测**：臂运动进 qpos/qvel/tcp_pos（state 全含），方块运动进 front/wrist 图像（帧间差可见，10fps 下帧间位移小）——A2 从观测流推断与 A1 从动作信号的差距只有这 1 帧 + 遮挡窗口；
- 双相机（front 全局 + wrist 近景）下遮挡通常短暂；被推/被抓的方块轨迹逐帧可跟踪；
- **经验证据**：actor GRU 输入即纯 obs_enc（modeling_gaussian_actor.py:526），14000 步 99%——策略需要同样的"方块在哪/往哪动"上下文并成功学到；critic 所需状态信念与 actor 同源；
- 残留风险（诚实声明）：方块被遮挡且恰在被推的窗口内，A1 有增益；若 A2 实测不足，升级路径 **A3**：行为动作进循环 + 查询动作在头（`Q'(s_{t+1}, a^π | h'^β)`，最贴近 R2D2 语义），但 target 末位帧（帧 idx+T）的行为动作不在 batch 内，需 buffer 补 `next_actions` 或填充约定——改动最大，作为备选消融而非本方案默认。

### 4.5 缓冲区大小分析（2026-08-14 实测核实；2026-08-14 修订对齐当前 json 与 8252 帧演示集）
- 事实：`online_ratio=0.5`；当前 json（560bd5d 起）= **在线 1500 / 离线 8500**，演示 150 集 8252 帧；历史 99% 运行（GRUoutput）= 在线 8000 / 离线 2000、演示 582 帧。离线 buffer = 演示全集 + 仅干预。
- **硬约束（2026-08-14 修订触发点）**：`from_lerobot_dataset`（buffer.py）在 `capacity < len(dataset)` 时直接 `raise ValueError`——离线容量必须 ≥ 8252；原决策 7 的 6000 按蓝图命令启动即崩。当前 json 的 8500 已满足。
- **本实验立场（决策 7，2026-08-14 修订）**：消融实验采用 **离线 8500 / 在线 4000**（总帧数 12500，较当前 10000 内存 +25%；存储于 CPU，实际占用可忽略）；**同一新配置下重跑纯 SAC / R-SAC 基线**做公平消融（99% 历史基线仅参考，不可直接比）。
- **量化**：离线 8500 = 演示 8252 帧（103 集）+ **248 帧干预余位**——加载即 97% 满，干预每写入 1 帧就环形淘汰 1 帧最老数据（先淘汰干预自身，第 249 帧干预起开始淘汰最老演示帧）；对比历史 2000+582（演示占 74.5%、~4000 总步才被淘汰）：新配置演示**数量**多 14 倍、被淘汰得更早但相对占比更小。在线 4000 = 50 集 = 6.7 分钟窗口（比历史基线 8000 的 13.3 分钟更新鲜、比当前 1500 的 2.5 分钟陈旧——在线池缩小是为给离线池腾容量，与旧表述"在线更新鲜"口径不同，以本节为准）。
- **若单独调的方向（用户直觉"离线大、在线小"有理论支撑）**：离线 buffer 容量 = 人类修正池多样性（当前 8500 ≈ 106 集含 103 集演示；容量再大 → 干预余位更多、演示淘汰更慢）；在线 buffer 容量 = 当前策略窗口新鲜度（当前 1500 ≈ 2.5 分钟最鲜；历史 8000 ≈ 13.3 分钟最陈旧）。**当前 json 的 1500/8500 已实现"离线大、在线小"方向**，本实验仅把在线提到 4000。
- 注意：(i) 离线余位仅 248 帧——只要干预总量 > 248 步（按历史干预率几乎必然），演示从第 249 帧干预起即开始被淘汰（可从 wandb 干预率确认实际干预量）；(ii) 0.5/0.5 混合比本身是另一个旋钮。

### 4.6 计算量与梯度
- 数据吞吐 `B×T=64` 恒定（机制现成）；GRU 参数 ≈ 46 万/网络（§2），相对整体策略占比小。
- q_preds（critic loss）不 detach encoder（与现行为一致，critic 优化器含 encoder 参数）；actor loss 的 Q 项 `∂Q/∂a` 回传策略（动作在头、无历史梯度链，与现前馈路径同构，语义最简）。

## 5. 实验参数（CLI 覆盖，json 不动）

**四种组合**：

| 模式 | use_recurrent | critic_use_recurrent | 用途 |
|---|---|---|---|
| 纯 SAC | false（默认） | false（默认） | 基线/零回归 |
| R-SAC | true | false（默认） | 现有对照基线（已验证 99%） |
| 仅 critic GRU | false | true | 消融（隔离 critic 循环贡献） |
| **双关（目标实验）** | true | true | 完整方案 |

```bash
# learner：双关目标实验（从头训练，先删 gym-hil/output；数据管线 = 决策 7 修订：离线 8500 / 在线 4000）
python -m lerobot.rl.learner --config_path gym-hil/train_hil_env.json --resume false \
  --policy.policy_kwargs.use_recurrent true \
  --policy.policy_kwargs.recurrent_hidden_size 256 \
  --policy.policy_kwargs.recurrent_num_layers 1 \
  --algorithm.sequence_length 8 --algorithm.critic_use_recurrent true \
  --algorithm.use_torch_compile false \
  --policy.offline_buffer_capacity 8500 --policy.online_buffer_capacity 4000 2>&1 | tee full_trace.log

# learner：消融（仅 critic GRU；use_recurrent 不传 → 默认 false；同一 8500/4000 配置）
python -m lerobot.rl.learner --config_path gym-hil/train_hil_env.json --resume false \
  --algorithm.sequence_length 8 --algorithm.critic_use_recurrent true \
  --algorithm.use_torch_compile false \
  --policy.offline_buffer_capacity 8500 --policy.online_buffer_capacity 4000 2>&1 | tee full_trace.log

# learner：新配置基线（R-SAC 仅 actor GRU——必须与消融同配置重跑；纯 SAC 同理去掉 GRU 开关即可）
python -m lerobot.rl.learner --config_path gym-hil/train_hil_env.json --resume false \
  --policy.policy_kwargs.use_recurrent true \
  --policy.policy_kwargs.recurrent_hidden_size 256 \
  --policy.policy_kwargs.recurrent_num_layers 1 \
  --algorithm.sequence_length 8 \
  --algorithm.use_torch_compile false \
  --policy.offline_buffer_capacity 8500 --policy.online_buffer_capacity 4000 2>&1 | tee full_trace.log

# 续训（用 checkpoint 内 train_config.json，critic_use_recurrent 已固化其中，无需再传 CLI）
python -m lerobot.rl.learner --config_path gym-hil/output/checkpoints/last/pretrained_model/train_config.json --resume true 2>&1 | tee full_trace.log

# actor：critic 仅 learner 侧生效（actor 不构造 critic），命令与 R-SAC 相同；
# 若追求两侧配置一致可同传 --algorithm.critic_use_recurrent true（无运行时影响）
python -m lerobot.rl.actor --config_path gym-hil/actor_hil_env.json \
  --policy.policy_kwargs.use_recurrent true \
  --policy.policy_kwargs.recurrent_hidden_size 256 \
  --policy.policy_kwargs.recurrent_num_layers 1 \
  --algorithm.sequence_length 8 2>&1 | tee actor_trace.log

# 评估（与 R-SAC 完全相同；critic 开关不参与——eval_simple 只加载 actor.* 权重）
python -m lerobot.rl.eval_simple --config_path gym-hil/actor_hil_env.json \
  --policy.pretrained_path gym-hil/output/checkpoints/006000/pretrained_model \
  --policy.policy_kwargs.use_recurrent true \
  --policy.policy_kwargs.recurrent_hidden_size 256 \
  --policy.policy_kwargs.recurrent_num_layers 1 \
  --eval.n_episodes 10 --policy.device cuda
```

- `sequence_length` 档位沿用第一篇 §5 表（8 推荐 / 16 / 4）；`utd_ratio=2`、`batch_size=64` 不变 → 每优化步 128 帧，与现状相同。
- **统一带 `--algorithm.use_torch_compile false`**：torch.compile 在 `_init_critics`（:103-105）包 critic，critic GRU 首次进 dynamo（CUDA 首次前向 specialize）有 TracingShapeError 风险（§7）；4 条实验命令（双关/消融/两条基线）统一关闭 → 消融对比不含 compile 混杂因素。续训命令不带（`use_torch_compile` 已固化在 checkpoint 内 `train_config.json`）。actor 侧不编译 critic、无需带。
- 从头训练前删 `gym-hil/output` 与 `gym-hil/output_actor`；先 learner 后 actor。
- **数据管线 = 决策 7（修订）：离线 8500 / 在线 4000**（总帧数 12500、内存较当前 +25%；离线 8500 是装下 8252 帧演示的硬约束，见 §4.5）；纯 SAC / R-SAC 基线用同一配置重跑（§4.5）。
- 预期收益：critic 的 Q 估值用上观测历史（部分可观测：方块位姿不在 state 中、靠图像推断；历史提供运动上下文）→ 估值更准 → 策略梯度信号更稳；对照基线 R-SAC 14000 步 99% vs 纯 SAC 45%。

## 6. 验证清单（实施后执行）

1. `py_compile` 3 个改动文件（configuration_sac.py / modeling_gaussian_actor.py / sac_algorithm.py）。
2. **零回归**（`git worktree add /tmp/lerobot-orig HEAD` 造改动前代码，对比时必须 `async_prefetch=False`）：默认配置与 R-SAC 配置两条路径，`Policy.forward` 与 `SACAlgorithm.update` 与改动前**逐位一致**（critic else 分支未动、调用点不传视图）。
3. **新脚本 `gym-hil/tests/verify_gru_critic_chain.py`**（镜像 verify_gru_chain.py 风格；强制 `use_torch_compile=false`、`device=cpu`、基线 `output好` 的 train_config.json + CLI 覆盖）：
   - A. **双关链路**：`critic_use_recurrent=true + use_recurrent=true + seq=8` → `update()` 不崩、loss 有限、q_preds 形状 (2, 64)、B×T==64；
   - B. **消融路径**：`critic_use_recurrent=true + use_recurrent=false + seq=8` → actor 前馈 else 分支展平后 `update()` 正常（§3.2 支撑路径）；**补 `_compute_loss_actor` 的 Q 调用形状断言：q_preds 应为 (2, 64)**（B 项最容易悄悄错位处，覆盖 `actions_pi.reshape(B, T, -1)` 与 done 透传两条链路）；
   - C. **done 掩码回归**：构造 done 在窗口中间（B=2, T=4，`done[0,1]=1`），pred/target Q 与"零历史基线"逐位一致、与"未掩码基线"有差异（误传未左移 done 会被断言捕获——镜像 verify_gru_chain C 节范式）；
   - D. **checkpoint 往返**：state_dict 含 `critic_ensemble.gru.*`/`critic_target.gru.*` 键；load 后前向逐位一致；缺失 gru 键时 `load_state_dict` 报错（§3.3 校验）；**CUDA 构造 + save_pretrained 产出 model.safetensors 不抛 RuntimeError**（`_NonFlatteningGRU` 回归，同 e4ac49b 验证法）；
   - E. **配置校验**：`critic_use_recurrent=true + seq=1` → ValueError；`use_recurrent=true + seq=1` → ValueError。
4. 真机实验：双关 vs R-SAC 从头训练对比（成功率/干预率），结论记录回本文档。

## 7. 风险与注意

- **torch.compile**：json 默认 `use_torch_compile=true`，critic GRU 首次进入 compile 路径（现有 compile 只包 critic；actor GRU 不经 compile）。dynamo 对 T 静态的 GRU 循环应可处理（`_NonFlatteningGRU` 的 flatten 置空在 compile 下仍生效）；**§5 实验命令已统一带 `--algorithm.use_torch_compile false` 规避**（消融对比也不含 compile 混杂因素）；若想验证 compile 路径，去掉该 flag 试跑（沙箱验证本就 false）。
- **"动作驱动动力学"的损失**（§4.4 量化）：A2 历史纯观测，动作效果滞后 1 帧可见；遮挡+被推窗口内有增益损失，预期小；若实测 Q 不稳/不收敛，升级路径 A3（行为动作进循环 + 查询动作在头，需 buffer 补 next_actions）。
- **resume 兼容**：`critic_use_recurrent` 固化在 checkpoint 内 `train_config.json`，resume 必须用该文件；开 critic GRU 加载无 gru 键的旧 checkpoint 会直接报错（§3.3 校验，防静默随机权重）。
- **`_NonFlatteningGRU`**：进程内一次性 cuDNN UserWarning（权重非连续内存），无碍（同 e4ac49b 遗留）。
- 离散 critic（`num_discrete_actions` 非空时）保持前馈，不在本次范围；若以后启用需单独评估。
- actor/learner 两侧 `policy.policy_kwargs` 与 `algorithm.sequence_length` 必须一致（AGENTS.md 既有约定）；`critic_use_recurrent` 仅 learner 生效。
- **buffer 参数（决策 7 修订）**：消融实验固定 离线 8500 / 在线 4000；0.5/0.5 混合比与其余数据管线参数保持基线值；纯 SAC / R-SAC 基线须同配置重跑（99% 历史基线仅参考）。

## 8. 下一步待办（按优先级）

1. **~~[优先] 修复 checkpoint 全量写盘阻塞~~ ✅ 已实施（2026-08-14）**：to_lerobot_dataset 同步写盘（learner.py:636/643）改为**异步后台 dump**——`save_training_checkpoint` 提交 `_DatasetDumpTask`（快照 `ReplayBuffer.clone_for_dataset()` 在主线程完成：紧凑拷贝、图像转 uint8 约 100KB/帧、~1s 内），单一 `_CheckpointDatasetDumper` 线程串行写盘；dump 耗时超保存间隔时只保留最新待写任务（合并丢弃中间态，内存上界 2 份快照）；训练循环结束后 `wait_and_stop()` 排空，保证最后一份 dataset 完整落盘（resume 恢复源）。实现选**异步保存**方向（非只存增量/去视频化）；同步兜底分支保留（无 dumper 时行为与原来逐位一致）。验证：`gym-hil/tests/verify_async_dataset_dump.py` 16 项全 PASS（快照==同步 dump 逐帧一致含并发写环形覆盖、负向对照证明零共享、uint8 紧凑化分支、dumper 合并去旧语义、真实任务端到端落盘）；buffer 序列采样零回归 PASS。改动点：`buffer.py:clone_for_dataset`、`learner.py:_DatasetDumpTask/_CheckpointDatasetDumper`（均带 `#修改` 标记）。遗留说明：快照提交在主线程约阻塞 0.5~1s/次（10000 帧），相对原全量写盘数十秒级阻塞可忽略。
2. **[优先] 数据管线改为 离线 8500 / 在线 4000**（决策 7 修订，消融实验配置）：`--policy.offline_buffer_capacity 8500 --policy.online_buffer_capacity 4000`（离线 8500 为装下 8252 帧演示的硬约束，`from_lerobot_dataset` 容量不足直接 ValueError；总帧数 12500、内存较当前 +25%；当前 json 已为 8500/1500，仅需覆盖 online=4000）；**必须用同一新配置重跑纯 SAC / R-SAC 基线**做公平消融（99% 历史基线仅参考，不可直接比）。
3. 按第 3 节实施 critic GRU 代码改动（全部带 `#修改 ... #结束` / `#===` 标记）。**实施注意事项（2026-08-14 蓝图全文审核结论，与代码逐条核对无实质 bug，以下为防御/确认项）**：
   - ① `actions.view(B, T, -1)` 用 **`reshape`** 代替：`actions[:, :DISCRETE_DIMENSION_INDEX]` 截断（:370）后是非连续切片（stride (4,1)），view 对非连续张量直接 RuntimeError；当前 `num_discrete_actions=null` 不触发截断，但属防御性改法（离散 critic 未来启用即踩雷）。**已固化：§3.3 片段/§2 伪代码现一律用 reshape。**
   - ② GRU 实验命令统一加 **`--algorithm.use_torch_compile false`**：torch.compile 在 `_init_critics`（:103-105，CPU 上包装）时就把 critic 包进编译图，critic GRU 首次进 dynamo（CUDA 首次前向 specialize），若遇 TracingShapeError 用此退路（沙箱验证本就 false；json 默认 true，见 §7）。**已固化：§5 四条实验命令已带该 flag。**
   - ③ 再确认**消融路径隐式前提**：`use_recurrent=false + seq=8` 时 `_compute_loss_critic` 的 seq 分支（:327）按 `sequence_length` 门控（**非 use_recurrent**），会调 `policy.actor(next_obs_view, ..., done=done_next)` → actor else 分支（§3.2 展平）兜住——该路径当前是崩的（5D 图像喂 ResNet10），§3.2 的 6 行正是修它；seq=1 时检测不命中、逐位不变。**✅ 已确认（2026-08-14 蓝图全文审核时逐行核对过 sac_algorithm.py:327/335-337 与 modeling_gaussian_actor.py:553-556）。**
   - ④ 确认 `B, T` / `obs_view` / `done_view` 在 seq 分支（:327-337 / :471-474）内定义、q_preds/q_targets 片段作用域可用。**✅ 已确认（:328 定义 `B, T`/`next_obs_view`，`obs_view` 按本蓝图补建；:473-474 定义 `obs_view`/`done_view`；q_preds 片段在 no_grad 块外、同函数作用域内，均可访问）。**
   - ⑤ **`CriticEnsemble.forward` 现有设备搬运行必须保留**：`device = get_device_from_parameters(self)`（:761）+ `observations = {k: v.to(device) for k, v in observations.items()}`（:763）在 GRU 分支中仍需先执行——§2 伪代码省略了该行（属伪代码省略而非删除）；GRU 分支应在搬移后的 observations 上取 `first_key` / `B, T`。**✅ 已确认（2026-08-14 复审，对照 sac_algorithm.py:761-763）。**
   - ⑥ **蓝图行号复核对账（2026-08-14 复审，逐行 grep 验证，实施以语义描述为准）**：核心引用**精确命中**——`td_target` 公式 :363、离散截断 :370、q_preds :371-376、`_compute_loss_actor` 的 obs_view/done_view :473-474 与 Q 调用 :479-484、`Policy.forward` else 分支 :553-555；仅 4 处小漂移：q_stats :389-398→实际 :390-398、`sequence_length` :90-96→:91-95（均差 1 行，数空行所致）、`_NonFlatteningGRU` 类定义行 :426-436→:434、`_strip_encoder_keys` 为模块级函数在 :691（蓝图并入 state_dict :624-639 一段，state_dict 本身 :624-640 无误）、learner.py 干预写离线 :1014-1016→实际 :1016-1018。
4. 沙箱执行第 6 节验证 1-3 项，通过后提交（中文提交信息，参照历史风格）。**验证清单补一条（写入 §6）**：消融路径（B 项）下 `_compute_loss_actor` 的 Q 调用形状断言——q_preds 应为 **(2, 64)**（B 项最容易悄悄错位处，含 `actions_pi.view(B, T, -1)` 与 done 透传两条链路）。
5. 真机实验：双关 vs R-SAC（同 8500/4000 配置，第 5 节命令）从头训练，对比成功率/干预率。
6. 实验结论（成功/失败/调参）记录回本文档第 5 节，更新 AGENTS.md 与 `命令.txt`。
7. **[工作流，2026-08-14 起] 评估结论落盘**：每次 eval_simple.py 评估，最终结论写入**归属的 checkpoint 文件夹**、**按超参数命名**的实验记录文件（如 `gym-hil/<run>/checkpoints/006000/pretrained_model/实验记录_双关_h256_seq8_off8500_on4000.md`），本文件 §5 不替代该落盘。
8. **[工作流，2026-08-14 起] 整体实验记录文档**：`gym-hil/实验记录_整体.md` 为算法修改决策、消融实验、评估结果的**单一总账**；每次修改算法 / 跑实验 / 评估，先登记再执行，完成后回填（与各归档文档联动）。

## 9. 仓库约定提醒（详见 AGENTS.md）

- 本地改动用 `#修改 ... #结束` / `#===` 注释标记（搜 `#修改` 可定位全部改动点）。
- 分支名/提交信息用中文；提交信息用分号列出改动要点。
- 先 learner 后 actor；从头训练前删 `gym-hil/output` 和 `gym-hil/output_actor`。
- resume 必须用 checkpoint 内 `train_config.json` + `--resume true`。
- 修改 `rl/` 或 `policies/gaussian_actor/` 时需同时考虑 actor/learner 两侧一致性。
- wandb project=`hil_test`；训练/评估命令以 `命令.txt` 与 AGENTS.md 为准。

## 10. 实施记录（待实施后回填）

**状态**：蓝图定稿（2026-08-14），**尚未实施**。本文件用于跨对话窗口续接，实施前先读本文件 + `AGENTS.md` + 第一篇，从第 8 节继续。

（实施后按第一篇 §10 格式回填：落地清单、与蓝图偏差表、§6 验证结果、真机实验结果与遗留。）
