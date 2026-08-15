#!/usr/bin/env python
import logging
import os
import time
import torch
from safetensors.torch import load_file
from lerobot.configs import parser
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.rl.gym_manipulator import make_robot_env, make_processors, reset_and_build_transition, step_env_and_process_transition
from lerobot.rl.train_rl import TrainRLServerPipelineConfig
#修改 ============ 评估结果自动落盘（工具模块，§4 约定） ============
# 评估结束自动计算成功率并把结果追加写入 checkpoint 文件夹（md + jsonl）。
# 逻辑全部在 eval_autolog.py（零 torch 依赖，可独立单测）；本文件只做
# 循环内计数（成功/干预）与收尾一行调用。
# 成功判据 = 环境每步 info["succeed"]（panda_pick env 每步计算）。
# 不能用 done 判成功：基础环境 terminated = success or exceeded_bounds（出界
# 也终止），HIL wrapper 还把 truncated 并入 terminated（hil_wrappers.py:246）。
#结束 ============================================
from lerobot.rl.eval_autolog import collect_hyperparams, dump_eval_record

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    if cfg.policy.pretrained_path is None:
        raise ValueError("Need --policy.pretrained_path")



    env, teleop = make_robot_env(cfg.env)


    env_processor, action_processor = make_processors(env, teleop, cfg.env, cfg.policy.device)


    policy = make_policy(cfg.policy, env_cfg=cfg.env)


    # 加载权重...
    # 注：model.safetensors 缺失时无需本地预检——make_policy 在权重加载阶段
    # 就会抛错（目录在缺文件 → safetensors FileNotFoundError；路径不存在 →
    # HFValidationError），不会静默拿随机权重跑评估。
    weight_path = os.path.join(cfg.policy.pretrained_path, "model.safetensors")

    state_dict = load_file(weight_path)

    actor_state = {}
    for k, v in state_dict.items():
        if k.startswith('actor.'):
            actor_state[k[6:]] = v
        elif k.startswith('encoder_actor.'):
            #修改 ============ encoder_actor 前缀改写 ============
            # Policy.encoder 与 encoder_actor 是同一对象，policy.actor.state_dict()
            # 的键是 'encoder.*'；若 checkpoint 以 'encoder_actor.*' 保存（外部来源
            # 格式），原实现保留原样导致加载时键不匹配被静默忽略（encoder 随机初始化）。
            #结束 ============================================
            actor_state["encoder." + k[len("encoder_actor."):]] = v
    if actor_state:
        #修改 ============ strict=True 防静默随机 ============
        # 原为 strict=False：漏传 GRU 开关（policy 无 gru 模块）或 checkpoint 与配置
        # 不匹配时，权重被静默丢弃、GRU/encoder 保持随机初始化，评估结果无效且无提示。
        # 改为 strict=True 后与 actor.py 侧加载行为一致（load_state_dict 默认严格），
        # 键集合不匹配会直接报出 missing/unexpected keys 定位问题。
        #结束 ============================================
        policy.actor.load_state_dict(actor_state, strict=True)
    else:
        # 兼容性兜底：非本仓库格式（无 'actor.' 前缀）的 checkpoint，宽松加载
        policy.load_state_dict(state_dict, strict=False)


    device = get_safe_torch_device(cfg.policy.device, log=True)

    policy = policy.to(device).eval()


    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        dataset_stats=cfg.policy.dataset_stats,
    )


    n_episodes = getattr(cfg.eval, 'n_episodes', 10) if hasattr(cfg, 'eval') else 10


    all_rewards = []
    all_steps = []
    #修改 ============ 成功/干预逐步计数（自动落盘数据源） ============
    # 成功判据见文件头注释；干预计数沿用 actor.py 同款读法
    # info.get(TeleopEvents.IS_INTERVENTION)，按步累计、口径与 actor 一致。
    #结束 ============================================
    all_success = []
    all_intv_steps = []

    for ep in range(n_episodes):

        transition = reset_and_build_transition(env, env_processor, action_processor)
        #修改 ============ GRU：episode 边界清空 hidden ============
        # use_recurrent=true 时 select_action 持续更新 policy.actor._hidden，
        # 每个 episode 开始前必须清零，否则沿用上一 episode 的历史。
        policy.reset()
        #结束 ============================================

        obs = transition['observation']
        ep_reward = 0.0
        step = 0
        done = False
        #修改 ============ episode 级成功/干预累计 ============
        ep_success = False
        ep_intv_steps = 0
        #结束 ============================================

        while not done:

            norm_obs = preprocessor.process_observation(obs)

            norm_obs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in norm_obs.items()}


            with torch.no_grad():

                action = policy.select_action(norm_obs)


            action_denorm = postprocessor.process_action(action)


            new_transition = step_env_and_process_transition(
                env, transition, action_denorm, env_processor, action_processor
            )

            reward = new_transition.get('reward', 0.0)
            done = new_transition.get('done', False) or new_transition.get('truncated', False)
            ep_reward += float(reward)
            step += 1
            transition = new_transition
            obs = new_transition['observation']

            #修改 ============ 逐步读取成功/干预状态 ============
            info = new_transition.get('info') or {}
            if info.get("succeed", False):
                ep_success = True
            if info.get(TeleopEvents.IS_INTERVENTION, False):
                ep_intv_steps += 1
            #结束 ============================================

        all_rewards.append(ep_reward)
        all_steps.append(step)
        all_success.append(ep_success)
        all_intv_steps.append(ep_intv_steps)
        print(f"Episode {ep+1}: reward={ep_reward:.2f}, steps={step}")


    n_success = sum(1 for s in all_success if s)
    print(f"\n===== Evaluation Summary =====")
    print(f"Episodes: {n_episodes}")
    print(f"Average reward: {sum(all_rewards)/len(all_rewards):.3f}")
    print(f"Average steps: {sum(all_steps)/len(all_steps):.1f}")
    #修改 ============ Summary 增打成功率 + 自动落盘 ============
    print(f"Success rate: {n_success}/{n_episodes} ({n_success/n_episodes:.1%})")
    hp = collect_hyperparams(cfg.policy.pretrained_path, cfg)
    dump_eval_record(
        cfg.policy.pretrained_path,
        hp,
        list(zip(all_rewards, all_steps, all_success, all_intv_steps)),
    )
    #结束 ============================================
    env.close()
    print("Evaluation finished successfully.")
    time.sleep(1)   # 让终端显示
    os._exit(0)

if __name__ == "__main__":
    main()
