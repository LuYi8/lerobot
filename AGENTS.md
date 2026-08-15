# 本仓库专属工程约定（gym-hil HIL-SAC）

> **本仓库是 LeRobot 的 fork，实际工作是 `gym-hil` 的 HIL（Human-in-the-Loop）SAC 训练**：
> Panda 机械臂抓方块（`PandaPickCubeKeyboard-v0`），分布式 learner / actor 架构。
> 以下约定**优先于**本文档下半部分的上游 LeRobot 通用约定。涉及本地改动的代码时，先读这一节。

## 快速命令（完整清单见 `gym-hil/命令.txt`）

```bash
# 1. 采集离线演示数据（键盘遥操作，非 RL 训练）
python -m lerobot.rl.gym_manipulator --config_path gym-hil/record_hil_env.json

# 2. Learner —— 先启动（gRPC server，127.0.0.1:50051）
#    从头训练：先删除 gym-hil/output
python -m lerobot.rl.learner --config_path gym-hil/train_hil_env.json --resume false 2>&1 | tee full_trace.log
#    断点续训：必须用 checkpoint 里的 train_config.json（不是原 json）
python -m lerobot.rl.learner --config_path gym-hil/output/checkpoints/last/pretrained_model/train_config.json --resume true 2>&1 | tee full_trace.log

# 3. Actor —— 后启动（连 learner；从头训练前先删除 gym-hil/output_actor）
python -m lerobot.rl.actor --config_path gym-hil/actor_hil_env.json 2>&1 | tee actor_trace.log
#    继续训练（加载已有权重）
python -m lerobot.rl.actor --config_path gym-hil/actor_hil_env.json \
  --policy.pretrained_path gym-hil/output/checkpoints/last/pretrained_model 2>&1 | tee actor_trace.log

# 4. 评估（eval_simple：轻量、只加载 actor 权重）
python -m lerobot.rl.eval_simple --config_path gym-hil/actor_hil_env.json \
  --policy.pretrained_path gym-hil/output/checkpoints/006000/pretrained_model \
  --eval.n_episodes 10 --policy.device cuda
```

## 架构与数据流（`src/lerobot/rl/`）

- **`learner.py`** — gRPC server（`LearnerService`，端口由 `policy.actor_learner_config` 指定）。主循环：收 transitions → 写 online replay buffer → `RLTrainer.training_step()`（SAC，`utd_ratio=2`）→ 每 `policy_parameters_push_frequency` 秒推权重给 actor → 定期保存 checkpoint。
- **`actor.py`** — gRPC 客户端。策略在环境中 rollout，**episode 结束时**才把整段 transitions 发回 learner、并从 `parameters_queue` 取最新权重（`update_policy_parameters`）。训练中人类可随时干预（`IS_INTERVENTION`），干预步的 transition 同时进 learner 的 offline buffer。
- **`gym_manipulator.py`** — 环境 / processor 工厂 + 数据采集（record 模式），负责 `reset_and_build_transition` / `step_env_and_process_transition`。
- **`buffer.py`** — `ReplayBuffer`（在线 + 离线），`from_lerobot_dataset` 从数据集构建，`to_lerobot_dataset` 把 buffer 存成数据集（用于 checkpoint 和 resume）。
- **`eval_simple.py`** — 评估入口：从 `model.safetensors` 提取 `actor.*` / `encoder_actor.*` 权重加载。
- **`algorithms/sac/`** — `SACAlgorithm`：critic ensemble + target（EMA，`critic_target_update_weight`）、log_alpha 温度、离散夹爪 critic（`num_discrete_actions` 非空时启用）。
- 三个队列：`parameters_queue`（learner→actor 权重）、`transitions_queue`（actor→learner 数据）、`interactions_queue`（episode 统计/wandb）。
- 并发模型由 `policy.concurrency.actor/learner` 控制：`"threads"`（默认）或 `"processes"`。

## 配置文件约定（`gym-hil/*.json`）

- 三个配置：`record_hil_env.json`（采集）、`train_hil_env.json`（learner）、`actor_hil_env.json`（actor）。draccus 解析，CLI 覆盖语法 `--key.subkey value`（如 `--policy.pretrained_path`、`--resume true`）。
- 环境：`env.type=gym_manipulator`，`env.name=gym_hil`（外部包 `gym-hil`），`env.task=PandaPickCubeKeyboard-v0`，`fps=10`，`control_mode=keyboard`。
- 关键超参位置：`algorithm.*`（SAC：`actor_lr`/`critic_lr`/`temperature_lr`/`discount`/`utd_ratio`/`policy_update_freq`/`critic_target_update_weight`/`temperature_init`）、`policy.*`（网络结构、`online_steps`、`online_buffer_capacity`、`offline_buffer_capacity`、`online_step_before_learning`、`actor_learner_config`）、顶层（`online_ratio`、`batch_size`、`save_freq`）。
- **actor 与 learner 的 `policy`/`algorithm`/`env` 定义必须保持一致**（actor 无 dataset 字段，但策略定义需同步）。
- `dataset_stats` 已固化在各 json 的 `policy.dataset_stats` 中；重新采集数据后需要更新统计量并同步到两个文件。

## 训练 / 续训 / 评估约定

- **启动顺序：先 learner 后 actor**。从头训练前删除 `gym-hil/output`（learner `resume=false` 时检测到旧 checkpoint 会直接抛错防覆盖）；actor 从头训练前删除 `gym-hil/output_actor`。
- Checkpoint 位于 `gym-hil/output/checkpoints/<6位step>/pretrained_model`，`last` 为符号链接指向最新；`save_freq=1000`（优化步）保存一次，同时把 replay buffer 存为 `output/dataset`。
- **resume 必须用 checkpoint 内的 `train_config.json` + `--resume true`**，不要用原 json，否则会丢失之后改过的超参。
- 两个步数概念不要混淆：`interaction_step`（环境交互步，actor 侧，episode 统计用）vs `optimization_step`（learner 训练步，日志/保存以它为准）；`policy.online_steps=99000` 是总交互步数上限。
- 训练效果参考：抓方块任务约 6000 优化步成功率 60%（历史提交记录）。
- 干预率是训练质量关键指标：前期频繁干预引导，后期逐步减少。

## GRU 循环增强实验（R-SAC，use_recurrent）

> 完整蓝图与正确性论证见 `gym-hil/上下文归档_SAC增加GRU方案.md`。默认关闭，不开启时行为与上游逐位一致。

- **开关（默认关闭）**：`--policy.policy_kwargs.use_recurrent true` + `recurrent_hidden_size 256` + `recurrent_num_layers 1`；配套 `--algorithm.sequence_length 8`（序列长度，1 = 原逐帧采样）。
- **计算量恒等**：`batch_size_eff = batch_size // sequence_length`（`SACAlgorithm.configure_data_iterator`），每批帧数 `B×T ≤ 64` 恒定；utd 不变 → 每优化步总帧数与现状相同。
- **actor/learner 两侧必须同参**：`policy.policy_kwargs` 与 `algorithm.sequence_length` 都要一致（json 不动，CLI 两侧同样追加）。
- **数据流**：开启后 batch 形状 `(B, T, ...)`；GRU 插在 encoder 与高斯头之间（逐时间步循环，critic 保持前馈）；actor 调用点还原序列视图并传 `done`（掩码时序"处理完第 t 帧后"清零，`_compute_loss_critic` 中 next 侧 done 左移 1 位）。
- **hidden 生命周期**：训练零初始化不保存；推理 `select_action` 用/更新 `actor._hidden`，episode 边界 `policy.reset()` 清零（actor.py / eval_simple.py 已加）。
- **实验命令**：见 `命令.txt` 末尾"GRU 循环增强实验"小节；从头训练前删 `gym-hil/output` 与 `gym-hil/output_actor`。
- 改动点均带 `#修改` 标记（configuration_gaussian_actor.py / configuration_sac.py / modeling_gaussian_actor.py / buffer.py / data_mixer.py / sac_algorithm.py / actor.py / eval_simple.py）。

## critic 加 GRU 实验（方案2，critic_use_recurrent，蓝图定稿**未实施**）

> **实施会话必读**：完整蓝图与实施待办见 `gym-hil/上下文归档_SAC增加GRU方案2_critic.md`（§8 为待办总表：写盘阻塞 → 8500/4000 数据管线 → 按 §3 改 3 个文件 → §6 验证 → 真机对比）；实验命令见 `命令.txt`"critic 加 GRU 实验"小节；算法决策/消融/评估总账见 `gym-hil/实验记录_整体.md`。三份先读再动代码。

- **开关**：`--algorithm.critic_use_recurrent true`（默认 false，与 `use_recurrent` 相互独立：全关 / 仅 actor=R-SAC / 仅 critic=消融 / 双关=目标实验）；critic 仅 learner 侧生效（actor 不构造 critic）。
- **设计**：GRU 输入=纯观测 obs_enc（与 actor GRU 逐字同构、复用 `_NonFlatteningGRU`），**动作在 Q 头前拼入**（head 输入 256+4）；done 掩码镜像 actor（pred 用 done、target 用左移 1 位的 done_next、actor loss 用 done）；td_target 公式零改动；critic 无推理路径（不涉及 actor.py / eval_simple.py）。
- **实验命令统一带 `--algorithm.use_torch_compile false`**（compile 包 critic，critic GRU 首次进 dynamo 的风险退路；消融对比不含 compile 混杂因素）；数据管线用决策 7（修订）：`--policy.offline_buffer_capacity 8500 --policy.online_buffer_capacity 4000`（离线 8500 为装下 8252 帧演示的硬约束，容量不足 `from_lerobot_dataset` 直接报错），纯 SAC / R-SAC 基线同配置重跑。

## 数据与归一化约定

- **图像**：环境输出为 float [0,255]，本地改动将其归一化到 [0,1]（`buffer.py` 中 `to_lerobot_dataset` 处）；数据集以 uint8 存储，读取时转 float/255。改动此逻辑时注意两端一致性。
- 归一化映射：`VISUAL → MEAN_STD`，`STATE/ENV/ACTION → MIN_MAX`。
- `action` 4 维 = 前 3 维连续（笛卡尔位移，tanh squash，[-1,1]）+ 最后 1 维离散夹爪（0~2）；夹爪惩罚 `gripper_penalty=-0.2` 抑制无意义开合。离散动作只在 `num_discrete_actions` 非空时启用。
- `observation.state` 18 维；图像 `front` + `wrist`，各 3×128×128。
- 数据混合：`mixer=online_offline`，`online_ratio=0.5`；干预 transition 同时进 offline buffer。
- 控制节奏：`fps=10`，`reset_time_s=1.0`，`control_time_s=8.0`（超时截断，提高样本效率）。

## 本地代码修改约定（升级上游前必读）

- 所有本地改动都用 `#修改 ... #结束` 或 `#===` 注释在代码中标记（搜 `#修改` 即可定位全部改动点）。
- 当前已知改动点：
  - `src/lerobot/rl/actor.py:322` — `num_discrete_actions` 时只对连续部分做 unnormalize，离散部分拼接回去。
  - `src/lerobot/rl/buffer.py:593,607` — `to_lerobot_dataset` 时图像归一化到 [0,1]。
  - `src/lerobot/rl/algorithms/sac/sac_algorithm.py:204` — critic 更新循环（utd 内多次更新）+ Q 值统计日志（`Q1_mean`/`targetQ_mean` 等，计入 wandb）。
  - `src/lerobot/rl/learner.py` — checkpoint 数据集异步 dump（`_CheckpointDatasetDumper` 后台线程 + `_DatasetDumpTask`；`save_training_checkpoint` 提交 `clone_for_dataset()` 紧凑快照，写盘不再阻塞训练循环；无 dumper 时同步兜底）。
  - `src/lerobot/rl/buffer.py` — `clone_for_dataset()` 紧凑冻结快照（异步 dump 用：图像转 uint8 零共享，范围判定只扫 size 内槽位）。
  - `src/lerobot/rl/eval_simple.py` — 独立评估脚本（不属于上游）；循环内按 `info["succeed"]` 计成功、`TeleopEvents.IS_INTERVENTION` 计干预（成功判据不能用 done——出界/超时也 done），结束自动落盘。
  - `src/lerobot/rl/eval_autolog.py` — 评估自动落盘工具（不属于上游，零 torch 依赖）：超参三级回退（checkpoint 内 train_config.json > CLI cfg > unk）+ 按超参生成 `实验记录_*.md` 文件名（变体 纯SAC/RSAC/仅criticGRU/双关 + seq>1 恒带 + tau/off/on）+ md/jsonl 追加写入 checkpoint 文件夹，写失败仅告警。
- 修改 `rl/` 或 `policies/gaussian_actor/` 时，**需同时考虑 actor/learner 两侧一致性**（两侧各自 `make_policy` 实例化策略，靠 gRPC 传参数）。
- 新实验脚本 / 配置放在 `gym-hil/`；输出目录 `gym-hil/output*`（已 gitignore）。

## 编码 / 提交约定

- 分支名、提交信息用**中文**（当前分支 `原始SAC`）；提交信息用分号列出多项改动要点（参照历史提交风格）。
- 实验命令总结更新到根目录 `命令.txt`。
- wandb：`wandb.enable=true`，project=`hil_test`；调试时可临时禁用。
- 训练/评估的完整命令、参数说明以 `gym-hil/命令.txt` 和本文件为准，其他来源（如上游文档）的 HILSerl 命令可能不适用于本仓库。

---

# 上游 LeRobot 通用约定（Upstream conventions）

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`). New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
