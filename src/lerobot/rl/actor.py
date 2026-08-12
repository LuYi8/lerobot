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
"""
Actor server runner for distributed HILSerl robot policy training.

This script implements the actor component of the distributed HILSerl architecture.
It executes the policy in the robot environment, collects experience,
and sends transitions to the learner server for policy updates.

Examples of usage:

- Start an actor server for real robot training with human-in-the-loop intervention:
```bash
python -m lerobot.rl.actor --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: The actor server requires a running learner server to connect to. Ensure the learner
server is started before launching the actor.

**NOTE**: Human intervention is key to HILSerl training. Press the upper right trigger button on the
gamepad to take control of the robot during training. Initially intervene frequently, then gradually
reduce interventions as the policy improves.

**WORKFLOW**:
1. Determine robot workspace bounds using `lerobot-find-joint-limits`
2. Record demonstrations with `gym_manipulator.py` in record mode
3. Process the dataset and determine camera crops with `crop_dataset_roi.py`
4. Start the learner server with the training configuration
5. Start this actor server with the same configuration
6. Use human interventions to guide policy learning

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import time
from collections.abc import Generator
from functools import lru_cache
from queue import Empty
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _grpc_available, require_package

if TYPE_CHECKING or _grpc_available:
    import grpc

    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import (
        bytes_to_state_dict,
        grpc_channel_options,
        python_object_to_bytes,
        receive_bytes_in_chunks,
        send_bytes_in_chunks,
        transitions_to_bytes,
    )
else:
    grpc = None
    services_pb2 = None
    services_pb2_grpc = None
    bytes_to_state_dict = None
    grpc_channel_options = None
    python_object_to_bytes = None
    receive_bytes_in_chunks = None
    send_bytes_in_chunks = None
    transitions_to_bytes = None

import torch
from torch import nn
from torch.multiprocessing import Queue

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor import TransitionKey
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import (
    Transition,
    move_transition_to_device,
)
from lerobot.utils.utils import (
    TimerManager,
    init_logging,
)

from .algorithms.base import RLAlgorithm
from .algorithms.factory import make_algorithm
from .gym_manipulator import (
    make_processors,
    make_robot_env,
    reset_and_build_transition,
    step_env_and_process_transition,
)
from .queue import get_last_item_from_queue
from .train_rl import TrainRLServerPipelineConfig

# Main entry point


@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    # Fail fast with a friendly error if the optional ``hilserl`` extra is missing.
    require_package("grpcio", extra="hilserl", import_name="grpc")
    cfg.validate()
    display_pid = False
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_{cfg.job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Actor logging initialized, writing to {log_file}")

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    learner_client, grpc_channel = learner_service_client(
        host=cfg.policy.actor_learner_config.learner_host,
        port=cfg.policy.actor_learner_config.learner_port,
    )

    logging.info("[ACTOR] Establishing connection with Learner")
    if not establish_learner_connection(learner_client, shutdown_event):
        logging.error("[ACTOR] Failed to establish connection with Learner")
        return

    if not use_threads(cfg):
        # If we use multithreading, we can reuse the channel
        grpc_channel.close()
        grpc_channel = None

    logging.info("[ACTOR] Connection with Learner established")

    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()

    concurrency_entity = None
    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from multiprocessing import Process

        concurrency_entity = Process

    receive_policy_process = concurrency_entity(
        target=receive_policy,
        args=(cfg, parameters_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process = concurrency_entity(
        target=send_transitions,
        args=(cfg, transitions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    interactions_process = concurrency_entity(
        target=send_interactions,
        args=(cfg, interactions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process.start()
    interactions_process.start()
    receive_policy_process.start()

    try:
        act_with_policy(
            cfg=cfg,
            shutdown_event=shutdown_event,
            parameters_queue=parameters_queue,
            transitions_queue=transitions_queue,
            interactions_queue=interactions_queue,
        )
        logging.info("[ACTOR] Policy loop finished")
    except Exception:
        import traceback
        traceback.print_exc()
        logging.exception("[ACTOR] Unhandled exception in act_with_policy")
        shutdown_event.set()
    finally:
        logging.info("[ACTOR] Closing queues")
        transitions_queue.close()
        interactions_queue.close()
        parameters_queue.close()

        transitions_process.join()
        logging.info("[ACTOR] Transitions process joined")
        interactions_process.join()
        logging.info("[ACTOR] Interactions process joined")
        receive_policy_process.join()
        logging.info("[ACTOR] Receive policy process joined")

        transitions_queue.cancel_join_thread()
        interactions_queue.cancel_join_thread()
        parameters_queue.cancel_join_thread()

        logging.info("[ACTOR] Cleanup complete")


# Core algorithm functions


def act_with_policy(
    cfg: TrainRLServerPipelineConfig,
    shutdown_event: Any,  # Event
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
):
    """
    Executes policy interaction within the environment.

    This function rolls out the policy in the environment, collecting interaction data and pushing it to a queue for streaming to the learner.
    Once an episode is completed, updated network parameters received from the learner are retrieved from a queue and loaded into the network.

    Args:
        cfg: Configuration settings for the interaction process.
        shutdown_event: Event to check if the process should shutdown.
        parameters_queue: Queue to receive updated network parameters from the learner.
        transitions_queue: Queue to send transitions to the learner.
        interactions_queue: Queue to send interactions to the learner.
    """
    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_policy_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor policy process logging initialized")

    logging.info("make_env online")

    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    print(f"Environment action space shape: {online_env.action_space.shape}")
    env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, cfg.policy.device)

    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")

    ### Instantiate the policy in both the actor and learner processes
    ### To avoid sending a policy object through the port, we create a policy instance
    ### on both sides, the learner sends the updated parameters every n steps to update the actor's parameters
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy = policy.to(device).eval()
    assert isinstance(policy, nn.Module)

    # Build the algorithm
    algorithm = make_algorithm(cfg=cfg.algorithm, policy=policy)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        dataset_stats=cfg.policy.dataset_stats,
    )

    transition = reset_and_build_transition(online_env, env_processor, action_processor)

    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0
    list_transition_to_send_to_learner = []
    episode_intervention = False
    # Add counters for intervention rate calculation
    episode_intervention_steps = 0
    episode_total_steps = 0

    policy_timer = TimerManager("Policy inference", log=False)

    for interaction_step in range(cfg.policy.online_steps):
        start_time = time.perf_counter()
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down act_with_policy")
            return

        observation = {
            k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features
        }
        # ========== 新增：原始状态NaN兜底 ==========
        state_key = "observation.state"
        if state_key in observation:
            if torch.isnan(observation[state_key]).any():
                logging.warning("环境输出observation.state含NaN，已用0替换")
                observation[state_key] = torch.nan_to_num(
                    observation[state_key], nan=0.0, posinf=10.0, neginf=-10.0
                )
        # ==========================================
        # Time policy inference and check if it meets FPS requirement
        with policy_timer:
            normalized_observation = preprocessor.process_observation(observation)
            action = policy.select_action(batch=normalized_observation)
            # Unnormalize only the continuous part.
            # ========== 修改后（修复代码） ==========
            if cfg.policy.num_discrete_actions is not None:
                
                # 1. 构造与原动作空间维度一致的占位张量，匹配postprocessor统计量维度
                full_action_placeholder = torch.zeros_like(action)
                full_action_placeholder[..., :-1] = action[..., :-1]  # 填入前3维连续动作
                # 2. 完整走一遍反归一化，维度完全匹配，不会报错
                full_unnormalized = postprocessor.process_action(full_action_placeholder)
                # 3. 只提取前3维反归一化后的连续动作
                continuous_action = full_unnormalized[..., :-1]
                
                discrete_action = action[..., -1:].to(
                    device=continuous_action.device, dtype=continuous_action.dtype
                )
                action = torch.cat([continuous_action, discrete_action], dim=-1)
            else:
                action = postprocessor.process_action(action)
            
            # ========== 标签定义：与训练端严格对齐（绝对不能改） ==========
            LABEL_CLOSE = 1.0   # 标签1 = 闭合指令
            LABEL_OPEN  = 0.0   # 标签0 = 张开指令

            # ========== 硬件映射：与实测物理动作严格对齐 ==========
            HARDWARE_CLOSE = 2.0  # 硬件值2.0 = 物理闭合
            HARDWARE_OPEN  = 0.0  # 硬件值0.0 = 物理张开

            # ========== 不对称滞回参数：闭合快、张开慢，优先保持夹紧 ==========
            CLOSE_STABLE_FRAMES = 2   # 闭合指令2帧稳定即执行
            OPEN_STABLE_FRAMES  = 1   # 张开指令需连续2帧稳定才执行

            # ========== 初始化（episode重置时同步清零） ==========
            if not hasattr(act_with_policy, 'gripper_stable_count'):
                act_with_policy.gripper_stable_count = 0
                act_with_policy.current_gripper_label = LABEL_OPEN  # 初始状态：张开

            # ========== 滞回核心逻辑 ==========
            target_label = action[..., 3].item()

            # 指令与当前状态一致，清零计数
            if target_label == act_with_policy.current_gripper_label:
                act_with_policy.gripper_stable_count = 0
            else:
                act_with_policy.gripper_stable_count += 1
                # 目标是闭合 → 低阈值快速响应；目标是张开 → 高阈值防抖
                threshold = CLOSE_STABLE_FRAMES if target_label == LABEL_CLOSE else OPEN_STABLE_FRAMES
                if act_with_policy.gripper_stable_count >= threshold:
                    act_with_policy.current_gripper_label = target_label
                    act_with_policy.gripper_stable_count = 0

            # ========== 最终映射到硬件真实指令 ==========
            if act_with_policy.current_gripper_label == LABEL_CLOSE:
                action[..., 3] = HARDWARE_CLOSE
            else:
                action[..., 3] = HARDWARE_OPEN

            # 调试日志（保留即可）
            #logging.debug(f"[滞回调试] 策略输出={target_label:.0f}, 当前执行标签={act_with_policy.current_gripper_label:.0f}, 计数={act_with_policy.gripper_stable_count}")




        policy_fps = policy_timer.fps_last

        log_policy_frequency_issue(policy_fps=policy_fps, cfg=cfg, interaction_step=interaction_step)
        # 第一次测试：强制发硬件值 0.0，观察夹爪是开还是合
        #action[..., 3] = 0.0

        # 第二次测试：注释上面一行，解开下面一行再测
        #action[..., 3] = 2.0

        # Use the new step function
        new_transition = step_env_and_process_transition(
            env=online_env,
            transition=transition,
            action=action,
            env_processor=env_processor,
            action_processor=action_processor,
        )

        # Extract values from processed transition
        next_observation = {
            k: v
            for k, v in new_transition[TransitionKey.OBSERVATION].items()
            if k in cfg.policy.input_features
        }

        # Teleop action is the action that was executed in the environment
        # It is either the action from the teleop device or the action from the policy
        executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]
        

        
        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)

        sum_reward_episode += float(reward)
        episode_total_steps += 1
        # 在拿到 reward 之后加
        penalty_val = new_transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)
        #logging.debug(f"[奖励调试] 总奖励={reward:.4f}, 夹爪惩罚={penalty_val:.4f}")

        # Check for intervention from transition info
        intervention_info = new_transition[TransitionKey.INFO]
        is_intervention = bool(intervention_info.get(TeleopEvents.IS_INTERVENTION, False))
        
        if is_intervention:
            episode_intervention = True
            episode_intervention_steps += 1
            # ========== 新增：干预期间同步更新策略GRU隐藏态 ==========
            with torch.no_grad():
                # 用当前真实观测跑一次前向，只更新隐藏态，不使用输出动作
                normalized_next_obs = preprocessor.process_observation(next_observation)
                _ = policy.select_action(batch=normalized_next_obs)
            # ========================================================


        # 离散化部分
        # 离散化：与离线数据集规则完全一致，保证分布对齐
        GRIPPER_DIM = 3
        DISCRETE_THRESHOLD = 1.3  # 和buffer.py严格同步
        gripper_val = executed_action[..., GRIPPER_DIM].item()
        discrete_gripper = 1.0 if gripper_val > DISCRETE_THRESHOLD else 0.0

        
        executed_action = executed_action.clone()
        executed_action[..., GRIPPER_DIM] = discrete_gripper

        complementary_info = {
            "discrete_penalty": torch.tensor(
                [new_transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)]
            ),
            TeleopEvents.IS_INTERVENTION.value: is_intervention,
        }
        # Create transition for learner (convert to old format)
        list_transition_to_send_to_learner.append(
            Transition(
                state=observation,
                action=executed_action,
                reward=reward,
                next_state=next_observation,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
        )

        # Update transition for next iteration
        transition = new_transition

        if done or truncated:
            logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}")

            update_policy_parameters(algorithm=algorithm, parameters_queue=parameters_queue, device=device)

            if len(list_transition_to_send_to_learner) > 0:
                push_transitions_to_transport_queue(
                    transitions=list_transition_to_send_to_learner,
                    transitions_queue=transitions_queue,
                )
                list_transition_to_send_to_learner = []

            stats = get_frequency_stats(policy_timer)
            policy_timer.reset()

            # Calculate intervention rate
            intervention_rate = 0.0
            if episode_total_steps > 0:
                intervention_rate = episode_intervention_steps / episode_total_steps

            # Send episodic reward to the learner
            interactions_queue.put(
                python_object_to_bytes(
                    {
                        "Episodic reward": sum_reward_episode,
                        "Interaction step": interaction_step,
                        "Episode intervention": int(episode_intervention),
                        "Intervention rate": intervention_rate,
                        **stats,
                    }
                )
            )

            # Reset intervention counters and environment
            sum_reward_episode = 0.0
            episode_intervention = False
            episode_intervention_steps = 0
            episode_total_steps = 0

            transition = reset_and_build_transition(online_env, env_processor, action_processor)
            policy.reset()  # 清空循环隐藏态，每个episode从零开始记忆
            # 重置夹爪滞回计数器，避免跨episode残留
            act_with_policy.gripper_stable_count = 0
            act_with_policy.current_gripper_label = LABEL_OPEN

        if cfg.env.fps is not None:
            dt_time = time.perf_counter() - start_time
            precise_sleep(max(1 / cfg.env.fps - dt_time, 0.0))


#  Communication Functions - Group all gRPC/messaging functions


def establish_learner_connection(
    stub: "services_pb2_grpc.LearnerServiceStub",
    shutdown_event: Any,  # Event
    attempts: int = 30,
) -> bool:
    """Establish a connection with the learner.

    Args:
        stub (services_pb2_grpc.LearnerServiceStub): The stub to use for the connection.
        shutdown_event (Event): The event to check if the connection should be established.
        attempts (int): The number of attempts to establish the connection.
    Returns:
        bool: True if the connection is established, False otherwise.
    """
    for _ in range(attempts):
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down establish_learner_connection")
            return False

        # Force a connection attempt and check state
        try:
            logging.info("[ACTOR] Send ready message to Learner")
            if stub.Ready(services_pb2.Empty()) == services_pb2.Empty():
                return True
        except grpc.RpcError as e:
            logging.error(f"[ACTOR] Waiting for Learner to be ready... {e}")
            time.sleep(2)
    return False


@lru_cache(maxsize=1)
def learner_service_client(
    host: str = "127.0.0.1",
    port: int = 50051,
) -> "tuple[services_pb2_grpc.LearnerServiceStub, grpc.Channel]":
    """Return a client for the learner service.

    GRPC uses HTTP/2, which is a binary protocol and multiplexes requests over a single connection.
    So we need to create only one client and reuse it.

    Returns:
        tuple[services_pb2_grpc.LearnerServiceStub, grpc.Channel]: The stub and the channel.
    """

    channel = grpc.insecure_channel(
        f"{host}:{port}",
        grpc_channel_options(),
    )
    stub = services_pb2_grpc.LearnerServiceStub(channel)
    logging.info("[ACTOR] Learner service client created")
    return stub, channel


def receive_policy(
    cfg: TrainRLServerPipelineConfig,
    parameters_queue: Queue,
    shutdown_event: Any,  # Event
    learner_client: "services_pb2_grpc.LearnerServiceStub | None" = None,
    grpc_channel: "grpc.Channel | None" = None,
) -> None:
    """Receive parameters from the learner.

    Args:
        cfg (TrainRLServerPipelineConfig): The configuration for the actor.
        parameters_queue (Queue): The queue to receive the parameters.
        shutdown_event (Event): The event to check if the process should shutdown.
        learner_client (services_pb2_grpc.LearnerServiceStub | None): Optional pre-created stub.
        grpc_channel (grpc.Channel | None): Optional pre-created channel.
    """
    logging.info("[ACTOR] Start receiving parameters from the Learner")
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_receive_policy_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor receive policy process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        iterator = learner_client.StreamParameters(services_pb2.Empty())
        receive_bytes_in_chunks(
            iterator,
            parameters_queue,
            shutdown_event,
            log_prefix="[ACTOR] parameters",
        )

    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Received policy loop stopped")


def send_transitions(
    cfg: TrainRLServerPipelineConfig,
    transitions_queue: Queue,
    shutdown_event: Any,  # Event
    learner_client: "services_pb2_grpc.LearnerServiceStub | None" = None,
    grpc_channel: "grpc.Channel | None" = None,
) -> None:
    """Send transitions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - Transition Data:
        - A batch of transitions (observation, action, reward, next observation) is collected.
        - Transitions are moved to the CPU and serialized using PyTorch.
        - The serialized data is wrapped in a `services_pb2.Transition` message and sent to the learner.

    Args:
        cfg (TrainRLServerPipelineConfig): The configuration for the actor.
        transitions_queue (Queue): The queue to receive the transitions.
        shutdown_event (Event): The event to check if the process should shutdown.
        learner_client (services_pb2_grpc.LearnerServiceStub | None): Optional pre-created stub.
        grpc_channel (grpc.Channel | None): Optional pre-created channel.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_transitions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor transitions process logging initialized")

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendTransitions(
            transitions_stream(
                shutdown_event, transitions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming transitions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Transitions process stopped")


def send_interactions(
    cfg: TrainRLServerPipelineConfig,
    interactions_queue: Queue,
    shutdown_event: Any,  # Event
    learner_client: "services_pb2_grpc.LearnerServiceStub | None" = None,
    grpc_channel: "grpc.Channel | None" = None,
) -> None:
    """Send interactions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - Interaction Messages:
        - Contains useful statistics about episodic rewards and policy timings.
        - The message is serialized using `pickle` and sent to the learner.

    Args:
        cfg (TrainRLServerPipelineConfig): The configuration for the actor.
        interactions_queue (Queue): The queue to receive the interactions.
        shutdown_event (Event): The event to check if the process should shutdown.
        learner_client (services_pb2_grpc.LearnerServiceStub | None): Optional pre-created stub.
        grpc_channel (grpc.Channel | None): Optional pre-created channel.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_interactions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor interactions process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendInteractions(
            interactions_stream(
                shutdown_event, interactions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming interactions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Interactions process stopped")


def transitions_stream(
    shutdown_event: Any,  # Event
    transitions_queue: Queue,
    timeout: float,
) -> "Generator[Any, None, services_pb2.Empty]":
    while not shutdown_event.is_set():
        try:
            message = transitions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Transition queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message, services_pb2.Transition, log_prefix="[ACTOR] Send transitions"
        )

    return services_pb2.Empty()


def interactions_stream(
    shutdown_event: Any,  # Event
    interactions_queue: Queue,
    timeout: float,
) -> "Generator[Any, None, services_pb2.Empty]":
    while not shutdown_event.is_set():
        try:
            message = interactions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Interaction queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message,
            services_pb2.InteractionMessage,
            log_prefix="[ACTOR] Send interactions",
        )

    return services_pb2.Empty()


#  Policy functions


def update_policy_parameters(algorithm: RLAlgorithm, parameters_queue: Queue, device):
    """Drain the latest learner-pushed weights into ``algorithm.policy``."""
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
    if bytes_state_dict is not None:
        logging.info("[ACTOR] Load new parameters from Learner.")
        state_dicts = bytes_to_state_dict(bytes_state_dict)

        # TODO: check encoder parameter synchronization possible issues:
        # 1. When shared_encoder=True, we're loading stale encoder params from actor's state_dict
        #    instead of the updated encoder params from critic (which is optimized separately)
        # 2. When freeze_vision_encoder=True, we waste bandwidth sending/loading frozen params
        # 3. Need to handle encoder params correctly for both actor and discrete_critic
        # Potential fixes:
        # - Send critic's encoder state when shared_encoder=True
        # - Skip encoder params entirely when freeze_vision_encoder=True
        # - Ensure discrete_critic gets correct encoder state (currently uses encoder_critic)
        algorithm.load_weights(state_dicts, device=device)


#  Utilities functions


def push_transitions_to_transport_queue(transitions: list, transitions_queue):
    """Send transitions to learner in smaller chunks to avoid network issues.
    Args:
        transitions: List of transitions to send
        message_queue: Queue to send messages to learner
        chunk_size: Size of each chunk to send
    """
    transition_to_send_to_learner = []
    for transition in transitions:
        tr = move_transition_to_device(transition=transition, device="cpu")
        has_nan = False

        # 校验 state 所有观测字段
        for key, value in tr["state"].items():
            if torch.isnan(value).any():
                logging.warning(f"丢弃含NaN的transition: state[{key}]")
                has_nan = True
                break
        if has_nan:
            continue

        # 校验 next_state 所有观测字段
        for key, value in tr["next_state"].items():
            if torch.isnan(value).any():
                logging.warning(f"丢弃含NaN的transition: next_state[{key}]")
                has_nan = True
                break
        if has_nan:
            continue

        # 校验动作张量
        if torch.isnan(tr["action"]).any():
            logging.warning("丢弃含NaN的transition: action")
            continue

        transition_to_send_to_learner.append(tr)

    if len(transition_to_send_to_learner) == 0:
        logging.warning("本批transition全部含NaN，未发送任何数据")
        return
    transitions_queue.put(transitions_to_bytes(transition_to_send_to_learner))



def get_frequency_stats(timer: TimerManager) -> dict[str, float]:
    """Get the frequency statistics of the policy.

    Args:
        timer (TimerManager): The timer with collected metrics.

    Returns:
        dict[str, float]: The frequency statistics of the policy.
    """
    stats = {}
    if timer.count > 1:
        avg_fps = timer.fps_avg
        p90_fps = timer.fps_percentile(90)
        logging.debug(f"[ACTOR] Average policy frame rate: {avg_fps}")
        logging.debug(f"[ACTOR] Policy frame rate 90th percentile: {p90_fps}")
        stats = {
            "Policy frequency [Hz]": avg_fps,
            "Policy frequency 90th-p [Hz]": p90_fps,
        }
    return stats


def log_policy_frequency_issue(policy_fps: float, cfg: TrainRLServerPipelineConfig, interaction_step: int):
    if policy_fps < cfg.env.fps:
        logging.warning(
            f"[ACTOR] Policy FPS {policy_fps:.1f} below required {cfg.env.fps} at step {interaction_step}"
        )


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.actor == "threads"


if __name__ == "__main__":
    actor_cli()
