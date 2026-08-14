#!/usr/bin/env python
import logging
import os
import time
import torch
from safetensors.torch import load_file
from lerobot.configs import parser
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.rl.gym_manipulator import make_robot_env, make_processors, reset_and_build_transition, step_env_and_process_transition
from lerobot.rl.train_rl import TrainRLServerPipelineConfig

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
    weight_path = os.path.join(cfg.policy.pretrained_path, "model.safetensors")
    #修改 ============ 权重文件缺失直接报错 ============
    # 原实现找不到 model.safetensors 时静默继续，会拿随机权重跑完整评估
    # （路径写错 / last 符号链接断掉时结果完全无效且无提示）。
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"model.safetensors not found in {cfg.policy.pretrained_path}. "
            "Check --policy.pretrained_path."
        )
    #结束 ============================================

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

        all_rewards.append(ep_reward)
        all_steps.append(step)
        print(f"Episode {ep+1}: reward={ep_reward:.2f}, steps={step}")


    print(f"\n===== Evaluation Summary =====")
    print(f"Episodes: {n_episodes}")
    print(f"Average reward: {sum(all_rewards)/len(all_rewards):.3f}")
    print(f"Average steps: {sum(all_steps)/len(all_steps):.1f}")
    env.close()
    print("Evaluation finished successfully.")
    time.sleep(1)   # 让终端显示
    os._exit(0)

if __name__ == "__main__":
    main()