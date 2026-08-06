# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable, Iterator
from dataclasses import asdict
from typing import Any

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.optim import Optimizer

from lerobot.policies.gaussian_actor.modeling_gaussian_actor import (
    DISCRETE_DIMENSION_INDEX,
    MLP,
    DiscreteCritic,
    GaussianActorObservationEncoder,
    GaussianActorPolicy,
    orthogonal_init,
)
from lerobot.policies.utils import get_device_from_parameters
from lerobot.types import BatchType
from lerobot.utils.constants import ACTION
from lerobot.utils.transition import move_state_dict_to_device
from ..base import RLAlgorithm
from ..configs import TrainingStats
from .configuration_sac import SACAlgorithmConfig


class SACAlgorithm(RLAlgorithm):
    """Soft Actor-Critic. Owns critics, targets, temperature, and loss computation.
    [GRU改造] 支持序列模式：开启 use_sequence 后，Critic 加入 GRU 时序编码，
    支持 [batch, seq_len, *] 形状的序列输入，对齐 sac_v2_gru.py 训练逻辑。
    """
    config_class = SACAlgorithmConfig
    name = "sac"
    # ===== 新增：optimization_step 可读写属性 =====
    @property
    def optimization_step(self) -> int:
        return self._optimization_step

    @optimization_step.setter
    def optimization_step(self, value: int) -> None:
        self._optimization_step = int(value)


    def __init__(
        self,
        policy: GaussianActorPolicy,
        config: SACAlgorithmConfig,
    ):
        self.config = config
        self.policy_config = config.policy_config
        self.policy = policy
        self.optimizers: dict[str, Optimizer] = {}
        self._optimization_step: int = 0

        action_dim = self.policy.config.output_features[ACTION].shape[0]
        self._init_critics(action_dim)
        self._init_temperature(action_dim)
        self._device = torch.device(self.policy.config.device)
        self._move_to_device()

    def _init_critics(self, action_dim) -> None:
        """Build critic ensemble, targets.
        [GRU改造] 从配置读取序列开关，控制 Critic 是否加入 GRU 时序模块
        """
        encoder = self.policy.encoder_critic

        # 读取序列与GRU配置（兼容旧配置，默认关闭）
        use_sequence = getattr(self.config, "use_sequence", False)
        critic_gru_hidden_size = getattr(self.config, "critic_gru_hidden_size", 64)
        num_critic_gru_layers = getattr(self.config, "num_critic_gru_layers", 2)
        critic_gru_dropout = getattr(self.config, "critic_gru_dropout", 0.1)

        heads = [
            CriticHead(
                input_dim=encoder.output_dim + action_dim,
                **asdict(self.config.critic_network_kwargs),
                use_gru=use_sequence,
                gru_hidden_size=critic_gru_hidden_size,
                num_gru_layers=num_critic_gru_layers,
                gru_dropout=critic_gru_dropout,
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(encoder=encoder, ensemble=heads)

        target_heads = [
            CriticHead(
                input_dim=encoder.output_dim + action_dim,
                **asdict(self.config.critic_network_kwargs),
                use_gru=use_sequence,
                gru_hidden_size=critic_gru_hidden_size,
                num_gru_layers=num_critic_gru_layers,
                gru_dropout=critic_gru_dropout,
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(encoder=encoder, ensemble=target_heads)
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        # TODO(Khalil): Investigate and fix torch.compile
        # NOTE: torch.compile is disabled, policy does not converge when enabled.
        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

        self.discrete_critic_target = None
        if self.policy_config.num_discrete_actions is not None:
            self.discrete_critic_target = self._init_discrete_critic_target(encoder)
        if self.policy_config.num_discrete_actions is not None and use_sequence:
            warnings.warn(
                "Sequence mode does not currently support discrete action critic. "
                "Discrete critic will process temporal dimension as batch dimension.",
                UserWarning,
                stacklevel=2
            )

    def _init_discrete_critic_target(self, encoder: GaussianActorObservationEncoder) -> DiscreteCritic:
        """Build target discrete critic (main network is owned by the policy)."""
        discrete_critic_target = DiscreteCritic(
            encoder=encoder,
            input_dim=encoder.output_dim,
            output_dim=self.policy_config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )
        # TODO(Khalil): Compile the discrete critic
        discrete_critic_target.load_state_dict(self.policy.discrete_critic.state_dict())
        return discrete_critic_target

    def _init_temperature(self, continuous_action_dim: int) -> None:
        """Set up temperature parameter (log_alpha) and target entropy."""
        temp_init = self.config.temperature_init
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)]))
        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            total_action_dim = continuous_action_dim + (
                1 if self.policy_config.num_discrete_actions is not None else 0
            )
            self.target_entropy = -total_action_dim / 2

    def _move_to_device(self) -> None:
        self.policy.to(self._device)
        self.critic_ensemble.to(self._device)
        self.critic_target.to(self._device)
        self.log_alpha = nn.Parameter(self.log_alpha.data.to(self._device))
        if self.discrete_critic_target is not None:
            self.discrete_critic_target.to(self._device)

    @property
    def temperature(self) -> float:
        """Return the current temperature value, always in sync with log_alpha."""
        return self.log_alpha.exp().item()

    def _critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
        hidden_in: Tensor | None = None,
        return_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        critics = self.critic_target if use_target else self.critic_ensemble
        return critics(
            observations, actions, observation_features,
            hidden_in=hidden_in, return_hidden=return_hidden
        )


    def _discrete_critic_forward(
        self, observations, use_target=False, observation_features=None
    ) -> torch.Tensor:
        """Forward pass through a discrete critic network
        兼容单步/序列两种输入模式
        """
        discrete_critic = self.discrete_critic_target if use_target else self.policy.discrete_critic
        q_values = discrete_critic(observations, observation_features)
        return q_values

    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        """Run one SAC training step (critic / discrete-critic / actor / temperature).
        Pulls ``utd_ratio`` batches from ``batch_iterator``, computes the relevant
        losses, backpropagates each, and updates target networks.
        """
        # 仅每10步打印一次调试信息
        if self._optimization_step % 10 == 0:
            print(f"[更新调试] 第 {self._optimization_step} 步，开始取batch...")
        clip = self.config.grad_clip_norm
        for _ in range(self.config.utd_ratio - 1):
            batch = next(batch_iterator)
            if self._optimization_step % 10 == 0:
                print("[更新调试] 成功取到batch")
            fb = self._prepare_forward_batch(batch, include_complementary_info=True)
            loss_critic = self._compute_loss_critic(fb)
            self.optimizers["critic"].zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_ensemble.parameters(), max_norm=clip)
            self.optimizers["critic"].step()

            if self.policy_config.num_discrete_actions is not None:
                loss_dc = self._compute_loss_discrete_critic(fb)
                self.optimizers["discrete_critic"].zero_grad()
                loss_dc.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.discrete_critic.parameters(), max_norm=clip)
                self.optimizers["discrete_critic"].step()

            self._update_target_networks()

        batch = next(batch_iterator)
        fb = self._prepare_forward_batch(batch, include_complementary_info=False)
        loss_critic = self._compute_loss_critic(fb)
        self.optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(self.critic_ensemble.parameters(), max_norm=clip).item()
        self.optimizers["critic"].step()

        stats = TrainingStats(
            losses={"loss_critic": loss_critic.item()},
            grad_norms={"critic": critic_grad},
        )

        if self.policy_config.num_discrete_actions is not None:
            loss_dc = self._compute_loss_discrete_critic(fb)
            self.optimizers["discrete_critic"].zero_grad()
            loss_dc.backward()
            dc_grad = torch.nn.utils.clip_grad_norm_(
                self.policy.discrete_critic.parameters(), max_norm=clip
            ).item()
            self.optimizers["discrete_critic"].step()
            stats.losses["loss_discrete_critic"] = loss_dc.item()
            stats.grad_norms["discrete_critic"] = dc_grad

        if self._optimization_step % self.config.policy_update_freq == 0:
            loss_actor = self._compute_loss_actor(fb)
            # 前置校验：损失值异常则直接跳过，不反向传播避免污染参数
            if torch.isnan(loss_actor) or torch.isinf(loss_actor):
                logging.warning("[WARNING] Actor损失出现NaN/Inf，跳过本次更新")
                actor_grad = float('nan')
            else:
                self.optimizers["actor"].zero_grad()
                loss_actor.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=clip).item()
                if torch.isnan(torch.tensor(actor_grad)) or torch.isinf(torch.tensor(actor_grad)):
                    logging.warning("[WARNING] Actor梯度出现NaN/Inf，跳过本次更新")
                    self.optimizers["actor"].zero_grad()
                else:
                    self.optimizers["actor"].step()


            loss_temp = self._compute_loss_temperature(fb)
            self.optimizers["temperature"].zero_grad()
            loss_temp.backward()
            temp_grad = torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=clip).item()
            self.optimizers["temperature"].step()

            stats.losses["loss_actor"] = loss_actor.item()
            stats.losses["loss_temperature"] = loss_temp.item()
            stats.grad_norms["actor"] = actor_grad
            stats.grad_norms["temperature"] = temp_grad
            stats.extra["temperature"] = self.temperature

        self._update_target_networks()
        # ========== 全量参数NaN/Inf检测（每100步执行一次，降低性能开销） ==========
        if self._optimization_step % 100 == 0:
            nan_params = []
            # 检测Actor参数
            for name, param in self.policy.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    nan_params.append(f"policy.{name}")
            # 检测Critic参数
            for name, param in self.critic_ensemble.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    nan_params.append(f"critic_ensemble.{name}")
            # 检测温度参数
            if torch.isnan(self.log_alpha).any() or torch.isinf(self.log_alpha).any():
                nan_params.append("log_alpha")
            if nan_params:
                logging.critical(f"[FATAL] 检测到参数NaN/Inf，异常参数: {nan_params}")
                raise RuntimeError(f"参数NaN检测失败，异常参数: {nan_params}")
        
        self._optimization_step += 1
        return stats
    

    @staticmethod
    def _slice_time_dim(data, start: int, end: int | None):
        """对张量/张量字典在第1维（时间维）切片，兼容None
        Args:
            data: None / 单个张量 / {key: 张量} 字典
            start: 起始索引（包含）
            end: 结束索引（不包含），None表示到末尾
        """
        if data is None:
            return None
        if isinstance(data, dict):
            return {k: v[:, start:end, ...] for k, v in data.items()}
        # 单个张量直接切片
        return data[:, start:end, ...]

    def _compute_loss_critic(self, batch: dict[str, Any]) -> Tensor:
        # 提取通用组件
        observations = batch["state"]
        actions = batch[ACTION]
        observation_features = batch.get("observation_feature")
        rewards = batch["reward"].squeeze(-1)  # [B] 或 [B, L]
        next_observations = batch["next_state"]
        done = batch["done"].squeeze(-1)
        next_observation_features = batch.get("next_observation_feature")

        # ========== 序列预热与截断参数计算 ==========
        if self.config.use_sequence:
            seq_len = observations[next(iter(observations.keys()))].shape[1]
            # 默认预热长度：序列长度1/4，至少5步，不超过序列一半
            default_burn_in = max(5, min(seq_len // 4, seq_len // 2))
            
            if self.config.bptt_truncate_len is not None:
                keep_len = min(self.config.bptt_truncate_len, seq_len)
                burn_in_len = seq_len - keep_len
                # 保证预热长度不低于默认值，不足则压缩有效段
                if burn_in_len < default_burn_in:
                    burn_in_len = default_burn_in
                    keep_len = seq_len - burn_in_len
            else:
                # 非BPTT模式：拆分预热/有效段，预热段不回传梯度
                burn_in_len = default_burn_in
                keep_len = seq_len - burn_in_len

            # 边界保护：序列过短时退化为全序列无预热，并告警
            if keep_len <= 0:
                burn_in_len = 0
                keep_len = seq_len
                if seq_len < default_burn_in:
                    warnings.warn(
                        f"序列长度({seq_len})小于最小预热长度({default_burn_in})，已退化为无预热全序列训练，时序建模效果可能下降。",
                        UserWarning, stacklevel=2
                    )
        else:
            burn_in_len = 0
            keep_len = None

        # ========== 计算Target Q值（无梯度，全序列计算后切片对齐） ==========
        with torch.no_grad():
            # 统一时序起点：Actor与Target Critic均从零隐藏态开始
            # 与当前Critic的预热逻辑完全对齐，消除Bellman目标系统性偏差
            next_action_preds, next_log_probs, _ = self.policy.actor(
                next_observations,
                next_observation_features,
                mode="train",
                hidden_in=None,
            )
            # Target Critic 全序列前向：与当前Critic使用相同的零初始态
            q_targets = self._critic_forward(
                observations=next_observations,
                actions=next_action_preds,
                use_target=True,
                observation_features=next_observation_features,
                hidden_in=None,
            )

            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)
                indices = indices[: self.config.num_subsample_critics]
                q_targets = q_targets[indices]
            min_q_target, _ = q_targets.min(dim=0)
            if self.config.use_backup_entropy:
                min_q_target = min_q_target - (self.temperature * next_log_probs)
            # Bellman目标值
            td_target_full = rewards + (1 - done) * self.config.discount * min_q_target

        # ========== 计算当前Critic Q值（梯度控制核心逻辑） ==========
        if self.policy_config.num_discrete_actions is not None:
            actions: Tensor = actions[..., :DISCRETE_DIMENSION_INDEX]

        if self.config.use_sequence and burn_in_len > 0:
            # --- 预热段：无梯度前向，构建初始上下文，提取截断点隐藏态 ---
            burn_obs = {k: v[:, :burn_in_len, ...] for k, v in observations.items()}
            burn_acts = actions[:, :burn_in_len, ...]
            burn_obs_feat = self._slice_time_dim(observation_features, 0, burn_in_len)
            
            with torch.no_grad():
                _, critic_burn_hidden = self._critic_forward(
                    observations=burn_obs,
                    actions=burn_acts,
                    use_target=False,
                    observation_features=burn_obs_feat,
                    hidden_in=None,
                    return_hidden=True,
                )
            # 预热隐藏态脱梯度，作为有效段初始状态
            critic_trunc_hidden = [h.detach() for h in critic_burn_hidden]

            # --- 有效段：带梯度前向，仅回传有效段梯度 ---
            valid_obs = {k: v[:, burn_in_len:, ...] for k, v in observations.items()}
            valid_acts = actions[:, burn_in_len:, ...]
            valid_obs_feat = self._slice_time_dim(observation_features, burn_in_len, None)
            
            q_preds_valid = self._critic_forward(
                observations=valid_obs,
                actions=valid_acts,
                use_target=False,
                observation_features=valid_obs_feat,
                hidden_in=critic_trunc_hidden,
            )
            # 截取对应有效段的Target值
            td_target_valid = td_target_full[:, burn_in_len:]
        else:
            # 无预热/单步模式：全量前向
            q_preds_valid = self._critic_forward(
                observations=observations,
                actions=actions,
                use_target=False,
                observation_features=observation_features,
                hidden_in=None,
            )
            td_target_valid = td_target_full

        # ========== 统一损失计算 ==========
        td_target_dup = einops.repeat(td_target_valid, "... -> e ...", e=q_preds_valid.shape[0])
        critics_loss = (
            F.mse_loss(
                input=q_preds_valid,
                target=td_target_dup,
                reduction="none",
            ).mean(dim=tuple(range(1, q_preds_valid.ndim)))
        ).sum()

        return critics_loss


    def _compute_loss_discrete_critic(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        actions = batch[ACTION]
        rewards = batch["reward"].squeeze(-1)
        next_observations = batch["next_state"]
        done = batch["done"].squeeze(-1)
        observation_features = batch.get("observation_feature")
        next_observation_features = batch.get("next_observation_feature")
        complementary_info = batch.get("complementary_info")

        actions_discrete: Tensor = actions[..., DISCRETE_DIMENSION_INDEX:].clone()
        actions_discrete = torch.round(actions_discrete)
        actions_discrete = actions_discrete.long()

        # ========== 【修复】序列维度适配：压平时间维度，与Critic输出对齐 ==========
        has_seq_dim = actions_discrete.ndim == 3  # [B, L, 1]
        if has_seq_dim:
            batch_size, seq_len, _ = actions_discrete.shape
            actions_discrete = actions_discrete.reshape(batch_size * seq_len, 1)
            rewards = rewards.reshape(batch_size * seq_len)
            done = done.reshape(batch_size * seq_len)
            if complementary_info is not None and "discrete_penalty" in complementary_info:
                complementary_info["discrete_penalty"] = complementary_info["discrete_penalty"].reshape(batch_size * seq_len)

        discrete_penalties: Tensor | None = None
        if complementary_info is not None:
            discrete_penalties = complementary_info.get("discrete_penalty")

        with torch.no_grad():
            next_discrete_qs = self._discrete_critic_forward(
                next_observations, use_target=False, observation_features=next_observation_features
            )
            best_next_discrete_action = torch.argmax(next_discrete_qs, dim=-1, keepdim=True)
            target_next_discrete_qs = self._discrete_critic_forward(
                observations=next_observations,
                use_target=True,
                observation_features=next_observation_features,
            )
            target_next_discrete_q = torch.gather(
                target_next_discrete_qs, dim=-1, index=best_next_discrete_action
            ).squeeze(-1)

            rewards_discrete = rewards
            if discrete_penalties is not None:
                rewards_discrete = rewards + discrete_penalties
            target_discrete_q = rewards_discrete + (1 - done) * self.config.discount * target_next_discrete_q

        predicted_discrete_qs = self._discrete_critic_forward(
            observations=observations, use_target=False, observation_features=observation_features
        )
        predicted_discrete_q = torch.gather(predicted_discrete_qs, dim=-1, index=actions_discrete).squeeze(-1)

        discrete_critic_loss = F.mse_loss(input=predicted_discrete_q, target=target_discrete_q)
        return discrete_critic_loss
    
    # ========== 【新增：补回缺失的actor损失方法】==========
    def _compute_loss_actor(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        observation_features = batch.get("observation_feature")
        initial_hidden = batch.get("initial_hidden", None)

        if self.config.use_sequence and self.config.bptt_truncate_len is not None:
            seq_len = observations[next(iter(observations.keys()))].shape[1]
            keep_len = min(self.config.bptt_truncate_len, seq_len)
            burn_in_len = seq_len - keep_len

            # ========== Actor 侧：预热 + 截断（使用存储的初始隐藏态）==========
            if burn_in_len > 0:
                # 预热段：前向得到截断点隐藏态，detach 断开梯度
                burn_obs = {k: v[:, :burn_in_len, ...] for k, v in observations.items()}
                burn_obs_feat = self._slice_time_dim(observation_features, 0, burn_in_len)
                _, _, _, burn_hidden = self.policy.actor(
                    burn_obs, burn_obs_feat,
                    mode="train", hidden_in=initial_hidden,
                    return_hidden=True,
                )
                actor_trunc_hidden = burn_hidden.detach()
            else:
                # 无预热则直接使用序列初始隐藏态
                actor_trunc_hidden = initial_hidden

            # 有效段：计算动作与对数概率（梯度仅在有效段回传）
            valid_obs = {k: v[:, burn_in_len:, ...] for k, v in observations.items()}
            valid_obs_feat = self._slice_time_dim(observation_features, burn_in_len, None)
            actions_pi, log_probs, _ = self.policy.actor(
                valid_obs, valid_obs_feat,
                mode="train", hidden_in=actor_trunc_hidden,
            )

            # ========== Critic 侧：预热 + 截断（使用真实动作预热，对齐Critic训练分布）==========
            if burn_in_len > 0:
                critic_burn_obs = {k: v[:, :burn_in_len, ...] for k, v in observations.items()}
                critic_burn_obs_feat = self._slice_time_dim(observation_features, 0, burn_in_len)
                # 修复：使用真实动作预热Critic，而非策略生成动作，对齐Critic训练时的输入分布
                critic_burn_actions = batch[ACTION][:, :burn_in_len, ...]
                if self.policy_config.num_discrete_actions is not None:
                    critic_burn_actions = critic_burn_actions[..., :DISCRETE_DIMENSION_INDEX]
                
                _, critic_burn_hidden = self._critic_forward(
                    observations=critic_burn_obs,
                    actions=critic_burn_actions,
                    use_target=False,
                    observation_features=critic_burn_obs_feat,
                    hidden_in=None,
                    return_hidden=True,
                )
                critic_trunc_hidden = [h.detach() for h in critic_burn_hidden]
            else:
                critic_trunc_hidden = None



            # 有效段 Critic 前向
            q_preds = self._critic_forward(
                observations=valid_obs,
                actions=actions_pi,
                use_target=False,
                observation_features=valid_obs_feat,
                hidden_in=critic_trunc_hidden,
            )
            min_q_preds = q_preds.min(dim=0)[0]
            actor_loss = ((self.temperature * log_probs) - min_q_preds).mean()
        else:
            # 全序列模式：统一预热规则，与Critic损失计算逻辑完全对齐
            seq_len = observations[next(iter(observations.keys()))].shape[1]
            default_burn_in = max(5, min(seq_len // 4, seq_len // 2))
            burn_in_len = default_burn_in
            keep_len = seq_len - burn_in_len
            if keep_len <= 0:
                burn_in_len = 0
                keep_len = seq_len

            # Actor侧：预热段无梯度构建上下文
            if burn_in_len > 0:
                burn_obs = {k: v[:, :burn_in_len, ...] for k, v in observations.items()}
                burn_obs_feat = self._slice_time_dim(observation_features, 0, burn_in_len)
                _, _, _, burn_hidden = self.policy.actor(
                    burn_obs, burn_obs_feat,
                    mode="train", hidden_in=None,
                    return_hidden=True,
                )
                actor_trunc_hidden = burn_hidden.detach()
            else:
                actor_trunc_hidden = None

            # Actor有效段前向
            valid_obs = {k: v[:, burn_in_len:, ...] for k, v in observations.items()}
            valid_obs_feat = self._slice_time_dim(observation_features, burn_in_len, None)
            actions_pi, log_probs, _ = self.policy.actor(
                valid_obs, valid_obs_feat,
                mode="train", hidden_in=actor_trunc_hidden,
            )

            # Critic侧：用真实动作预热，与Critic损失计算分布一致
            if burn_in_len > 0:
                critic_burn_obs = {k: v[:, :burn_in_len, ...] for k, v in observations.items()}
                critic_burn_obs_feat = self._slice_time_dim(observation_features, 0, burn_in_len)
                critic_burn_actions = batch[ACTION][:, :burn_in_len, ...]
                _, critic_burn_hidden = self._critic_forward(
                    observations=critic_burn_obs,
                    actions=critic_burn_actions,
                    use_target=False,
                    observation_features=critic_burn_obs_feat,
                    hidden_in=None,
                    return_hidden=True,
                )
                critic_trunc_hidden = [h.detach() for h in critic_burn_hidden]
            else:
                critic_trunc_hidden = None

            # Critic有效段前向
            q_preds = self._critic_forward(
                observations=valid_obs,
                actions=actions_pi,
                use_target=False,
                observation_features=valid_obs_feat,
                hidden_in=critic_trunc_hidden,
            )
            min_q_preds = q_preds.min(dim=0)[0]
            actor_loss = ((self.temperature * log_probs) - min_q_preds).mean()

        return actor_loss



    # ======================================================
      
    def _compute_loss_temperature(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        observation_features = batch.get("observation_feature")
        initial_hidden = batch.get("initial_hidden", None)

        with torch.no_grad():
            if self.config.use_sequence and self.config.bptt_truncate_len is not None:
                seq_len = observations[next(iter(observations.keys()))].shape[1]
                keep_len = min(self.config.bptt_truncate_len, seq_len)
                burn_in_len = seq_len - keep_len

                if burn_in_len > 0:
                    burn_obs = {k: v[:, :burn_in_len, ...] for k, v in observations.items()}
                    burn_obs_feat = self._slice_time_dim(observation_features, 0, burn_in_len)
                    _, _, _, burn_hidden = self.policy.actor(
                        burn_obs, burn_obs_feat,
                        mode="train", hidden_in=initial_hidden,
                        return_hidden=True,
                    )
                    trunc_hidden = burn_hidden.detach()
                else:
                    trunc_hidden = initial_hidden

                valid_obs = {k: v[:, burn_in_len:, ...] for k, v in observations.items()}
                valid_obs_feat = self._slice_time_dim(observation_features, burn_in_len, None)
                _, log_probs, _ = self.policy.actor(
                    valid_obs, valid_obs_feat,
                    mode="train", hidden_in=trunc_hidden,
                )
            else:
                _, log_probs, _ = self.policy.actor(
                    observations, observation_features,
                    mode="train", hidden_in=initial_hidden,
                )

        temperature_loss = (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()
        return temperature_loss




    def _update_target_networks(self) -> None:
        """Update target networks with exponential moving average"""
        for target_p, p in zip(
            self.critic_target.parameters(), self.critic_ensemble.parameters(), strict=True
        ):
            target_p.data.copy_(
                p.data * self.config.critic_target_update_weight
                + target_p.data * (1.0 - self.config.critic_target_update_weight)
            )
        if self.policy_config.num_discrete_actions is not None:
            for target_p, p in zip(
                self.discrete_critic_target.parameters(),
                self.policy.discrete_critic.parameters(),
                strict=True,
            ):
                target_p.data.copy_(
                    p.data * self.config.critic_target_update_weight
                    + target_p.data * (1.0 - self.config.critic_target_update_weight)
                )

    def _prepare_forward_batch(
        self, batch: BatchType, *, include_complementary_info: bool = True
    ) -> dict[str, Any]:
        observations = batch["state"]
        next_observations = batch["next_state"]
        observation_features, next_observation_features = self.get_observation_features(
            observations, next_observations
        )
        forward_batch: dict[str, Any] = {
            ACTION: batch[ACTION],
            "reward": batch["reward"],
            "state": observations,
            "next_state": next_observations,
            "done": batch["done"],
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
            # 透传序列初始隐藏态，供 Actor 时序网络使用
            "initial_hidden": batch.get("initial_hidden"),
            "next_initial_hidden": batch.get("next_initial_hidden"),
        }
        if include_complementary_info and "complementary_info" in batch:
            forward_batch["complementary_info"] = batch["complementary_info"]
        return forward_batch

    def make_optimizers_and_scheduler(self) -> dict[str, Optimizer]:
        """
        Creates and returns optimizers for the actor, critic, and temperature components of a reinforcement learning policy.

        This function sets up Adam optimizers for:
        - The **actor network**, ensuring that only relevant parameters are optimized.
        - The **critic ensemble**, which evaluates the value function.
        - The **temperature parameter**, which controls the entropy in soft actor-critic (SAC)-like methods.

        It also initializes a learning rate scheduler, though currently, it is set to `None`.

        NOTE:
        - If the encoder is shared, its parameters are excluded from the actor's optimization process.
        - The policy's log temperature (`log_alpha`) is wrapped in a list to ensure proper optimization as a standalone tensor.

        Args:
            cfg: Configuration object containing hyperparameters.
            policy (nn.Module): The policy model containing the actor, critic, and temperature components.

        Returns:
            A dictionary mapping component names ("actor", "critic", "temperature")
            to their respective Adam optimizers.
        """
        actor_params = self.policy.get_optim_params()["actor"]
        self.optimizers = {
            "actor": torch.optim.Adam(actor_params, lr=self.config.actor_lr),
            "critic": torch.optim.Adam(self.critic_ensemble.parameters(), lr=self.config.critic_lr),
            "temperature": torch.optim.Adam([self.log_alpha], lr=self.config.temperature_lr),
        }
        if self.policy_config.num_discrete_actions is not None:
            self.optimizers["discrete_critic"] = torch.optim.Adam(
                self.policy.discrete_critic.parameters(), lr=self.config.critic_lr
            )
        return self.optimizers

    def get_optimizers(self) -> dict[str, Optimizer]:
        return self.optimizers

    def get_weights(self) -> dict[str, Any]:
        """Send actor + discrete-critic state dicts."""
        state_dicts: dict[str, Any] = {
            "policy": move_state_dict_to_device(self.policy.actor.state_dict(), device="cpu"),
        }
        if self.policy_config.num_discrete_actions is not None:
            state_dicts["discrete_critic"] = move_state_dict_to_device(
                self.policy.discrete_critic.state_dict(), device="cpu"
            )
        return state_dicts

    # 在 sac_algorithm.py 的 SACAlgorithm 类中修改 load_weights 方法
    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        """Load actor + discrete-critic weights into the policy."""
        actor_sd = move_state_dict_to_device(weights["policy"], device=device)
        
        # ========== 新增：旧版权重键名兼容映射 ==========
        remapped_actor_sd = {}
        for key, value in actor_sd.items():
            # 旧版 encoder_actor.xxx -> 新版 encoder.xxx（对应 self.actor.encoder）
            if key.startswith("encoder_actor."):
                new_key = key.replace("encoder_actor.", "encoder.", 1)
                remapped_actor_sd[new_key] = value
            # 其余键保持不变（如 GRU、MLP、mean_layer 等）
            else:
                remapped_actor_sd[key] = value
        # =================================================
        
        # 使用映射后的权重加载，strict=True 可验证是否完全匹配
        self.policy.actor.load_state_dict(remapped_actor_sd, strict=True)
        
        # 同步加载 critic 侧编码器权重（shared_encoder=False 时生效）
        if "policy" in weights and not self.policy_config.shared_encoder:
            critic_enc_sd = {}
            for key, value in weights["policy"].items():
                if key.startswith("encoder_critic."):
                    new_key = key.replace("encoder_critic.", "", 1)
                    critic_enc_sd[new_key] = value
            if critic_enc_sd:
                self.policy.encoder_critic.load_state_dict(critic_enc_sd, strict=True)

        if "discrete_critic" in weights and self.policy.discrete_critic is not None:
            discrete_sd = move_state_dict_to_device(weights["discrete_critic"], device=device)
            self.policy.discrete_critic.load_state_dict(discrete_sd)


    def state_dict(self) -> dict[str, torch.Tensor]:
        """Algorithm-owned trainable tensors.

        Encoder weights are stripped because they are owned by the policy
        (``policy.encoder_critic``) and already saved via ``policy.save_pretrained``.
        """
        bundle: dict[str, torch.Tensor] = {}
        for k, v in _strip_encoder_keys(self.critic_ensemble.state_dict()).items():
            bundle[f"critic_ensemble.{k}"] = v
        for k, v in _strip_encoder_keys(self.critic_target.state_dict()).items():
            bundle[f"critic_target.{k}"] = v
        if self.discrete_critic_target is not None:
            for k, v in _strip_encoder_keys(self.discrete_critic_target.state_dict()).items():
                bundle[f"discrete_critic_target.{k}"] = v
        bundle["log_alpha"] = self.log_alpha.detach()
        return bundle

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
    ) -> None:
        """In-place load of algorithm-owned tensors.

        ``log_alpha`` is restored via ``Parameter.data.copy_`` so the
        ``temperature`` optimizer's reference to the parameter object stays
        valid after resume.
        """
        critic_ensemble_state = _split_prefix(state_dict, "critic_ensemble.")
        critic_target_state = _split_prefix(state_dict, "critic_target.")
        self.critic_ensemble.load_state_dict(critic_ensemble_state, strict=False)
        self.critic_target.load_state_dict(critic_target_state, strict=False)

        if self.discrete_critic_target is not None:
            discrete_target_state = _split_prefix(state_dict, "discrete_critic_target.")
            self.discrete_critic_target.load_state_dict(discrete_target_state, strict=False)

        if "log_alpha" in state_dict:
            self.log_alpha.data.copy_(state_dict["log_alpha"].to(self.log_alpha.device))

    def get_observation_features(
        self, observations: Tensor, next_observations: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        """
        Get observation features from the policy encoder. It act as cache for the observation features.
        when the encoder is frozen, the observation features are not updated.
        We can save compute by caching the observation features.

        Args:
            policy: The policy model
            observations: The current observations
            next_observations: The next observations

        Returns:
            tuple: observation_features, next_observation_features
        """

        if self.policy.config.vision_encoder_name is None or not self.policy.config.freeze_vision_encoder:
            return None, None

        with torch.no_grad():
            observation_features = self.policy.actor.encoder.get_cached_image_features(observations)
            next_observation_features = self.policy.actor.encoder.get_cached_image_features(next_observations)

        return observation_features, next_observation_features


def _strip_encoder_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop ``encoder.*`` keys from a critic-module state dict."""
    return {k: v for k, v in state.items() if not k.startswith("encoder.")}


def _split_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Return the subset of ``state`` whose keys start with ``prefix``, prefix-stripped."""
    return {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}

class CriticHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
        use_gru: bool = False,
        gru_hidden_size: int = 64,
        num_gru_layers: int = 2,
        gru_dropout: float = 0.1,
    ):
        super().__init__()
        self.use_gru = use_gru
        # GRU层：放在MLP之前，做时序建模
        if self.use_gru:
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=gru_hidden_size,
                num_layers=num_gru_layers,
                batch_first=True,
                dropout=gru_dropout,
            )
            # 正交初始化GRU权重，提升初始数值稳定性
            # 权重正交初始化 + 偏置分段初始化
            for name, param in self.gru.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif 'bias' in name:
                    # GRU 偏置按 [重置门r, 更新门z, 候选状态n] 三段拼接
                    bias_size = param.shape[0]
                    n = bias_size // 3
                    nn.init.constant_(param[:n], 1.0)      # 重置门偏置为正，提升梯度流通
                    nn.init.constant_(param[n:2*n], 0.0)   # 更新门偏置为0，初始保留历史信息
                    nn.init.constant_(param[2*n:], 0.0)    # 候选状态偏置为0

            mlp_input_dim = gru_hidden_size
        else:
            self.gru = None
            mlp_input_dim = input_dim
        # MLP主干
        self.net = MLP(
            input_dim=mlp_input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )
        # Q值输出层
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(
        self,
        x: torch.Tensor,
        hidden_in: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入特征 [B, D] 或 [B, L, D]
            hidden_in: GRU初始隐藏态
            return_hidden: 是否返回最终隐藏态，用于BPTT截断
        Returns:
            q_values: Q值张量
            hidden_out: 最终隐藏态（仅return_hidden=True时返回）
        """
        if self.use_gru:
            if not hasattr(self, '_flattened'):
                self.gru.flatten_parameters()
                setattr(self, '_flattened', True)
            x, hidden_out = self.gru(x, hidden_in)
        else:
            hidden_out = None

        q = self.output_layer(self.net(x))
        q = q.squeeze(-1)

        if return_hidden:
            return q, hidden_out
        return q


class CriticEnsemble(nn.Module):
    def __init__(
        self,
        encoder: GaussianActorObservationEncoder,
        ensemble: list[CriticHead],
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.init_final = init_final
        self.critics = nn.ModuleList(ensemble)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        observation_features: torch.Tensor | None = None,
        hidden_in: torch.Tensor | list[torch.Tensor] | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        device = get_device_from_parameters(self)
        observations = {k: v.to(device) for k, v in observations.items()}
        obs_enc = self.encoder(observations, cache=observation_features)
        inputs = torch.cat([obs_enc, actions], dim=-1)
        q_values = []
        hidden_states = []

        # 修复：单张量输入时，为每个Critic Head复制独立副本，避免共享隐藏态
        if hidden_in is not None and not isinstance(hidden_in, list):
            hidden_in = [hidden_in.clone() for _ in range(len(self.critics))]

        for i, critic in enumerate(self.critics):
            curr_hidden = hidden_in[i] if hidden_in is not None else None
            
            if return_hidden:
                q, h = critic(inputs, hidden_in=curr_hidden, return_hidden=True)
                q_values.append(q)
                hidden_states.append(h)
            else:
                q_values.append(critic(inputs, hidden_in=curr_hidden))
        
        q_values = torch.stack(q_values, dim=0)
        if return_hidden:
            return q_values, hidden_states
        return q_values


