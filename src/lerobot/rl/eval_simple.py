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
    if os.path.exists(weight_path):

        state_dict = load_file(weight_path)

        actor_state = {}
        for k, v in state_dict.items():
            if k.startswith('actor.'):
                actor_state[k[6:]] = v
            elif k.startswith('encoder_actor.'):
                actor_state[k] = v
        if actor_state:
            policy.actor.load_state_dict(actor_state, strict=False)

        else:
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