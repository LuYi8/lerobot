#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
import traceback

import torch

from lerobot.cameras import opencv  # noqa: F401 (required for registry)
from lerobot.configs import parser
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa
from lerobot.teleoperators import gamepad, so_leader  # noqa
from lerobot.utils.device_utils import get_safe_torch_device

from .gym_manipulator import make_robot_env, make_processors
from .train_rl import TrainRLServerPipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def eval_policy(env, env_processor, policy, preprocessor, postprocessor, n_episodes, device):
    """
    Evaluate a policy in the given environment for a number of episodes.
    """
    policy.eval()
    rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs_raw, _ = env.reset()
        # Process raw observation to feature dict (keys like observation.images.front, observation.state)
        obs_features = env_processor.process_observation(obs_raw)
        episode_reward = 0.0
        step = 0
        done = False

        while not done:
            # 1. Preprocess observation (normalize)
            norm_obs = preprocessor.process_observation(obs_features)
            # Move to device
            norm_obs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in norm_obs.items()
            }

            # 2. Policy inference
            with torch.no_grad():
                action = policy.select_action(norm_obs)   # shape: (1, action_dim) or (action_dim,)

            # 3. Postprocess action (denormalize)
            action_denorm = postprocessor.process_action(action)
            # Convert to numpy and squeeze batch dim if present
            action_np = action_denorm.cpu().numpy()
            if action_np.ndim == 2 and action_np.shape[0] == 1:
                action_np = action_np[0]

            # 4. Step environment
            obs_raw, reward, terminated, truncated, _ = env.step(action_np)
            episode_reward += float(reward)
            step += 1
            done = terminated or truncated
            # Process the new raw observation for next iteration
            obs_features = env_processor.process_observation(obs_raw)

        rewards.append(episode_reward)
        episode_lengths.append(step)
        logger.info(f"Episode {ep+1:2d} | reward = {episode_reward:6.2f} | steps = {step:3d}")

    avg_reward = sum(rewards) / n_episodes
    avg_length = sum(episode_lengths) / n_episodes
    logger.info(f"\n===== Evaluation Summary =====")
    logger.info(f"Episodes: {n_episodes}")
    logger.info(f"Average reward: {avg_reward:.3f}")
    logger.info(f"Average episode length: {avg_length:.1f}")
    return rewards, episode_lengths


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    if cfg.policy.pretrained_path is None:
        logger.error("You must specify --policy.pretrained_path to a trained policy checkpoint.")
        sys.exit(1)

    pretrained_path = cfg.policy.pretrained_path
    logger.info(f"Loading policy from: {pretrained_path}")

    try:
        print("DEBUG: start main", flush=True)
        print("DEBUG: before make_robot_env", flush=True)
        env, teleop_device = make_robot_env(cfg.env)
        print("DEBUG: after make_robot_env", flush=True)

        logger.info("Creating environment processor...")
        env_processor, _ = make_processors(env, teleop_device, cfg.env, cfg.policy.device)

        logger.info("Loading policy...")
        policy_cls = get_policy_class(cfg.policy.type)
        policy = policy_cls.from_pretrained(pretrained_path)
        print("DEBUG: after from_pretrained", flush=True)

        device = get_safe_torch_device(cfg.policy.device, log=True)
        print(f"DEBUG: device = {device}", flush=True)
        print("DEBUG: before policy.to(device)", flush=True)
        policy = policy.to(device)
        print("DEBUG: after policy.to(device)", flush=True)

        # 直接构造预处理器（不再尝试 from_pretrained）
        logger.info("Constructing preprocessor and postprocessor from policy config...")
        if hasattr(policy, "config"):
            policy_cfg = policy.config
            dataset_stats = policy_cfg.dataset_stats
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                dataset_stats=dataset_stats,
            )
            logger.info("Preprocessor and postprocessor constructed from policy config.")
        else:
            raise RuntimeError("Cannot construct preprocessor: policy has no 'config' attribute.")

        n_episodes = getattr(cfg.eval, "n_episodes", 10) if hasattr(cfg, "eval") else 10
        logger.info(f"Number of episodes: {n_episodes}")

        logger.info("Starting evaluation...")
        eval_policy(env, env_processor, policy, preprocessor, postprocessor, n_episodes, device)

        env.close()
        logger.info("Evaluation finished successfully.")

    except Exception as e:
        logger.error(f"Evaluation failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()