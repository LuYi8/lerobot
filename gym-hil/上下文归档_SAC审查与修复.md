# 上下文归档：SAC 实现审查 + Bug 1 修复（2026-08-13）

> 本文件用于跨对话窗口续接。新窗口先读本文件 + `AGENTS.md`，再从"待办/下一步"继续。
> 相关会话任务：审阅五个 SAC 相关代码文件是否有 bug → 已修复 Bug 1（resume 权重恢复）。

## 0. 环境信息（沙箱验证用）

- 仓库：`/home/embody/lerobot`（LeRobot fork，gym-hil HIL-SAC，分支 `原始SAC`）
- conda 环境：**`lero6`**（python 位于 `/home/embody/miniconda3/envs/lero6/bin/python`，torch 2.11.0+cu126）
- 沙箱限制：torchcodec 视频解码不可用（缺 ffmpeg 库），需要读数据集视频时**必须**传 `video_backend="pyav"`（pyav 15.1.0 可用）；沙箱无 GPU，验证权重时把 `cfg.policy.device` 改 `"cpu"`
- 跑验证脚本：`cd /home/embody/lerobot && /home/embody/miniconda3/envs/lero6/bin/python - <<'EOF' ... EOF`（记得 `sys.path.insert(0, "src")`）
- 上次成功训练的完整输出：`gym-hil/output好/`（checkpoints 001000~005000，`last` 符号链接可用）；当前 `gym-hil/output` 不存在（按约定删除后从头训练）

## 1. 已完成的修改

> Bug 1 修复已于 2026-08-13 提交（commit b3c05a5，与 batch_size 128→64、AGENTS.md、本归档文档同批）。

**Bug 1 修复：resume 时 learner 不恢复策略权重** —— `src/lerobot/rl/learner.py:708-717`

`handle_resume_logic` 中在 `checkpoint_cfg.resume = True` 之后新增（带 `#修改 ... #结束` 标记注释）：

```python
checkpoint_cfg.policy.pretrained_path = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR)
```

- 问题本质：原代码加载 checkpoint 的 train_config.json 后 `policy.pretrained_path=None`，`make_policy`（learner.py:327）随机初始化 actor/encoder/discrete_critic；learner 启动即 `push_actor_policy_to_queue` 推随机权重，actor 每 episode 结束 `update_policy_parameters` 取最新权重 → 覆盖 actor 用 `--policy.pretrained_path` 加载的真权重，续训实际从零开始（且 critic head 与随机 encoder 不匹配、Adam 动量错位）。
- 验证：用 `output好/checkpoints/last/pretrained_model` 实测，修复前 62/62 权重键不匹配（diff 5.9e-1），修复后 62/62 完全一致（diff 0）；真实 `handle_resume_logic` 函数跑通（resume=True、pretrained_path 指向 checkpoint、model.safetensors 存在）；`py_compile` 通过。
- resume 命令不变（见 AGENTS.md）：learner 用 checkpoint 的 `train_config.json --resume true`；actor 加 `--policy.pretrained_path gym-hil/output/checkpoints/last/pretrained_model`。
- 注意：`from_pretrained` 里 `policy.to(config.device)`（device=cuda），须在有 GPU 的机器上跑（正常训练环境 OK）。

**Bug 4 修复：use_tanh_squash 从未被读取** —— `modeling_gaussian_actor.py:466-475`（已实现并验证，待提交）

`Policy.forward` 新增分支：`use_tanh_squash=True` 用 `TanhMultivariateNormalDiag`（行为与原来完全一致，配置均显式设 true）；`False` 时退化为普通对角高斯 `MultivariateNormal(loc, diag_embed(std))`（与 Tanh 分支 base_dist 同构）。
- 验证：沙箱实测 tanh 分支 max|a|=0.98（≤1）、raw 分支 max|a|=3.25（无界）、log_prob 有限、形状不变；`py_compile` 通过。
- 注意：若以后把配置改成 `false`，actor 输出将无界，需同步确认动作归一化链路（当前 env 动作空间 [-1,1]，依赖 tanh squash）。

**Bug 5 修复：torch.compile 注释与配置矛盾** —— `sac_algorithm.py:93-97`（已实现并验证，待提交）

按用户决定**保留编译、改注释**：`use_torch_compile: true` 与上次成功训练（6000 步 60% 成功率）一致；删除上游 "policy does not converge when enabled" 旧注释，改为说明本仓库实测可正常收敛，消除误导。

**Bug 3 修复：std clamp 语义与注释不符** —— `modeling_gaussian_actor.py`（已实现并验证，待提交）

三处改动（均带 `#修改` 标记）：
- `forward`（~:465-478）：clamp 从 std 空间改为 **log 空间** `std = exp(clamp(log_std, log(std_min), log(std_max)))`，与注释声称的 JAX 语义（clamp log_std）一致；`clamp(exp(x),a,b)=exp(clamp(x,log a,log b))`，当前配置 1e-5/5 行为**完全不变**（实测 allclose=True）。
- `Policy.__init__` 默认值 `-5/2` → `1e-5/10.0`（与 PolicyConfig 一致）：旧默认按 std 解释时下限失效（std 可趋 0 → 策略过早确定化）、上限压到 2。
- 新增 `import math`。
- resume 兼容性已确认：`output好` 全部 checkpoint 的 config 里 std_min/std_max 均为正数（1e-5/5），不会触发 log(负数) 崩溃。

## 2. 审查结论：未修复的 bug（按严重度）

### Bug 2（潜在，当前配置 null 未触发）：启用 num_discrete_actions 时链路维度不一致
- `src/lerobot/policies/gaussian_actor/modeling_gaussian_actor.py:50-52`：`continuous_action_dim = output_features[ACTION].shape[0]` = 4（含夹爪维）→ actor 输出 4 维；`select_action`（:86-95）再拼离散 → 5 维。
- `src/lerobot/rl/algorithms/sac/sac_algorithm.py:64`：critic head 输入 = encoder+4；`_compute_loss_critic`（:312-316）切 `actions[:, :-1]` = 3 维 → 形状不匹配崩溃。
- 当前 `gym-hil/*.json` 均为 `num_discrete_actions: null`（夹爪当连续维 [0,2]），所以能跑。若启用：需 `output_features.action.shape=[3]` + 3 维 action stats + 相应切片，否则必崩。
- 连带问题：`sac_algorithm.py:203` UTD 循环 `include_complementary_info=True`，:222 最后一轮 `False` → 启用离散 critic 时最后一次更新丢夹爪惩罚项（训练目标不一致）。

### Bug 3（已修复，见第 1 节）：std clamp 语义与注释不符
- ~~`modeling_gaussian_actor.py:461-463`：`std = torch.clamp(std, std_min, std_max)`，注释说 "Match JAX default clip"（JAX 是 clamp log_std ∈ [-5,2])。~~ 已修复：clamp 改 log 空间（行为不变），类默认值 -5/2 改为 1e-5/10.0（与 PolicyConfig 一致）。

### Bug 4（已修复，见第 1 节）：use_tanh_squash 从未被读取
- ~~`modeling_gaussian_actor.py:420-423` 存了 `self.use_tanh_squash`，`forward`（:468）无条件用 `TanhMultivariateNormalDiag`，设 False 无效果。~~ 已修复：forward 按 flag 分支，配置均 true，行为不变。

### Bug 5（已修复，见第 1 节）：torch.compile
- ~~`sac_algorithm.py:93-97` 注释 "torch.compile is disabled, policy does not converge when enabled"，但两个配置 `use_torch_compile: true` 且代码真的会 compile critic。~~ 已修复：保留编译（上次成功训练即 true），注释改为实测结论。

### Bug 6（近似，影响小）：满 buffer 时 next_state 跨 episode 污染
- `buffer.py:256-257`：`optimize_memory=True` 时 `next_idx=(idx+1)%capacity`，buffer 写满（8000）后位置末位↔位置 0 之间、episode 边界与环形位置不对齐处，bootstrap 目标用到别的 episode 状态。done=True 被掩码不受影响；truncated 样本受影响。DrQ 系常见近似。

### 轻微 / 提示
- `learner.py:636` 每次 save 把整个 replay buffer 转视频数据集（8000 帧×2 相机编码），一次几十秒~几分钟，阻塞训练循环。
- `learner.py:463` `0 % 1000 == 0` → step 0 也保存 checkpoint（仅 ~100 样本、interaction_step=0）。
- `actor.py:740` 只查 state 的 NaN（learner 侧 `check_nan_in_transition` 全查，有兜底）。
- 混合 batch 中 `is_intervention` 只有 online 一半有值（离线数据无此键，`concatenate_batch_transitions` 对单边键直接赋值、长度只有半批）——当前无 loss 使用，以后用会错位。
- 两配置算法超参不一致：`critic_target_update_weight` actor=0.005 vs learner=0.05；`policy_update_freq` 1 vs 2；buffer 容量 5000 vs 8000——不影响 gRPC 协议（actor 不训练），但与 AGENTS.md"必须一致"说法冲突。
- `sac_algorithm.py:335-344` Q 统计假设 `num_critics ≥ 2`（设 1 会 IndexError）。

## 3. 已排查确认没问题（避免下个窗口重复排查）

- **图像尺度链路一致**：env uint8[0,255] → `VanillaObservationProcessorStep`（observation_processor.py:89-90）转 float/255 → buffer 存 [0,1]；`to_lerobot_dataset` 存 [0,1]（writer 转 uint8 视频）；`LeRobotDataset` 读回默认 float[0,1]（video_utils.py decode 默认 return_uint8=False）；MEAN_STD 统计（mean≈0.27）同为 [0,1] 尺度。✓
- **动作不归一化**：`trainer.py` 的 `preprocess_rl_batch` 只处理 observation（state/next_state），buffer 里是原始 env 空间动作（连续 [-1,1]、夹爪 0~2）→ 离散 gather 的 `round().long()` 得到正确索引。✓
- **在线/离线 complementary_info 混合不崩**：实测数据集 reader 把 (1,) 特征压成**标量**（dataset_reader get_item），在线（torch.tensor([x])→squeeze→标量行）与离线存储形状一致 `(capacity,)`，`concatenate_batch_transitions` 不崩。（曾怀疑离线是 (capacity,1) 会崩，实测不崩。）✓
- **checkpoint 数据集图像读回 [0,1]**，与在线缓冲一致，resume 后两 buffer 尺度统一。✓
- **actor/learner policy 结构一致**（input/output features、vision_encoder=/home/embody/lerobot/resnet10、freeze、shared_encoder、policy_kwargs），协议只需 actor+discrete_critic 权重。✓

## 4. 关键代码定位（本次审查覆盖）

- `src/lerobot/rl/actor.py`（:322-330 连续/离散反归一化本地修改；:395 episode 结束取权重）
- `src/lerobot/rl/buffer.py`（:241-257 sample/next_idx；:505-641 to_lerobot_dataset 图像归一化本地修改；:643-764 _lerobotdataset_to_transitions）
- `src/lerobot/rl/learner.py`（:180 handle_resume_logic 已修；:327 make_policy；:712 load_training_state；:948 process_transitions）
- `src/lerobot/rl/algorithms/sac/sac_algorithm.py`（:179-271 update；:273-346 critic loss+Q统计本地修改；:349-405 离散 critic；:601-608 encoder key 剥离）
- `src/lerobot/policies/gaussian_actor/modeling_gaussian_actor.py`（:50-96 actor 构造/select_action；:446-476 forward 含 std clamp；:633-666 TanhMultivariateNormalDiag）
- 上下游已核对：`processor/hil_processor.py`（GymHILAdapter/Intervention/TimeLimit）、`processor/normalize_processor.py`（无 /255）、`processor/observation_processor.py`（Vanilla 做 /255）、`datasets/dataset_reader.py`（视频读回 float[0,1]、标量挤压）、`gym-hil/gym_hil/`（环境、wrappers）、`rl/data_sources/data_mixer.py`、`rl/trainer.py`

## 5. 下一步建议（待办，按优先级）

1. ~~提交 Bug 1 修复~~（已完成，b3c05a5）。
2. 续训验证：按 AGENTS.md 命令 resume 一次，对比 `0005000` 与 `last` checkpoint 的 eval 成功率，确认续训起点正确。
3. （可选）修 Bug 2：若计划启用离散夹爪（num_discrete_actions），需配套改 output action shape→[3]、action stats→3 维、`update()` 的 include_complementary_info 全改 True。~~std clamp 改为 log 空间~~（已完成，见第 1 节 Bug 3）。
4. ~~（可选）把 `use_torch_compile` 配置改为 false~~（已完成：保留 true、改注释）。
5. （可选）优化 checkpoint 保存耗时（to_lerobot_dataset 全量写盘）。

## 6. 仓库约定提醒（详见 AGENTS.md）

- 本地改动用 `#修改 ... #结束` / `#===` 注释标记；本次改动已带标记。
- 分支名/提交信息用中文；实验命令更新到 `gym-hil/命令.txt`。
- 先 learner 后 actor；从头训练前删 `gym-hil/output` 和 `gym-hil/output_actor`。
- resume 必须用 checkpoint 内 `train_config.json` + `--resume true`。
