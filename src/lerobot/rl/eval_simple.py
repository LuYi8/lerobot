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
    
    # ========== 关键修改1：暂存路径并临时清空，禁用自动预加载 ==========
    pretrained_path = cfg.policy.pretrained_path
    cfg.policy.pretrained_path = None

    env, teleop = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, teleop, cfg.env, cfg.policy.device)
    policy = make_policy(cfg.policy, env_cfg=cfg.env)
    # 加载前：打印初始化的参数范数
    spatial_kernel = policy.actor.encoder.spatial_embeddings['observation_images_front'].kernel
    state_weight = policy.actor.encoder.state_encoder[0].weight
    print(f"[加载前] 空间嵌入范数: {spatial_kernel.norm().item():.4f}")
    print(f"[加载前] 状态编码器范数: {state_weight.norm().item():.4f}")
    # ========== 关键修改2：纯手动加载权重，只保留actor有效部分 ==========
    weight_path = os.path.join(pretrained_path, "model.safetensors")
    if os.path.exists(weight_path):
        raw_state_dict = load_file(weight_path)
        # 直接加载到 policy 顶层，自动兼容旧版 encoder_actor / encoder_critic 命名
        missing, unexpected = policy.load_state_dict(raw_state_dict, strict=False)
        # 加载后：再次打印
        print(f"[加载后] 空间嵌入范数: {spatial_kernel.norm().item():.4f}")
        print(f"[加载后] 状态编码器范数: {state_weight.norm().item():.4f}")
        print("=" * 50)
        print(f"顶层权重加载完成")
        print(f"缺失键: {len(missing)} 个")
        print(f"多余键: {len(unexpected)} 个")
        if missing:
            print(f"缺失键示例: {missing[:5]}")
        print("=" * 50)
    else:
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    # 权重加载完成后，验证核心参数非零
    if hasattr(policy.actor.encoder, 'image_encoder'):
        embedder_weight = policy.actor.encoder.image_encoder.image_enc_layers.embedder[0].weight
        print(f"[权重校验] ResNet第一层卷积权重范数: {embedder_weight.norm().item():.4f}")
        
        spatial_kernel = policy.actor.encoder.spatial_embeddings['observation_images_front'].kernel
        print(f"[权重校验] 空间嵌入权重范数: {spatial_kernel.norm().item():.4f}")
    
    # 恢复路径配置（不影响后续逻辑）
    cfg.policy.pretrained_path = pretrained_path


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
        # 每个episode开始必须重置策略隐藏态，保证时序状态从零开始
        policy.reset(batch_size=1)
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
            print(f"[调试] 归一化动作: {action.cpu().numpy().round(4)}")


            action_denorm = postprocessor.process_action(action)
            print(f"真实机械臂动作: {action_denorm}")


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