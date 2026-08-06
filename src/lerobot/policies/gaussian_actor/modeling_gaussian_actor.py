#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Callable
from dataclasses import asdict
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import MultivariateNormal, TanhTransform, Transform, TransformedDistribution
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from ..pretrained import PreTrainedPolicy
from ..utils import get_device_from_parameters, get_dtype_from_parameters
from .configuration_gaussian_actor import GaussianActorConfig, is_image_feature
from torch.distributions import Normal, Independent
DISCRETE_DIMENSION_INDEX = -1  # Gripper is always the last dimension
import math
from typing import Optional


class GaussianActorPolicy(
    PreTrainedPolicy,
):
    config_class = GaussianActorConfig
    name = "gaussian_actor"

    def __init__(
        self,
        config: GaussianActorConfig | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        # Determine action dimension and initialize all components
        continuous_action_dim = config.output_features[ACTION].shape[0]
        self._init_encoders()
        self._init_actor(continuous_action_dim)
        self._init_discrete_critic()

    def get_optim_params(self) -> dict:
        optim_params = {
            "actor": [
                p
                for n, p in self.actor.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
        }
        return optim_params

    def reset(self, batch_size: int = 1):
        """Reset the policy: clear GRU hidden state and encoder caches
        Called at the beginning of each episode during environment interaction.
        Args:
            batch_size: 并行环境数量，单环境默认为1
        """
        # 重置策略网络的GRU隐藏态
        self.actor.reset_hidden(batch_size=batch_size)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError(
            "GaussianActorPolicy does not support action chunking. It returns single actions!"
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation
        Maintains internal GRU hidden state across timesteps. Call reset() at episode start.
        接口签名与原版完全一致，GRU模式下内部自动维护隐藏态
        """

        observations_features = None
        if self.shared_encoder and self.actor.encoder.has_images:
            observations_features = self.actor.encoder.get_cached_image_features(batch)
        # 推理模式：内部维护隐藏态，对外仅返回动作
        actions, _, _ = self.actor(batch, observations_features, mode="inference")

        if self.config.num_discrete_actions is not None:
            if self.discrete_critic is not None:
                discrete_action_value = self.discrete_critic(batch, observations_features)
                discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)
            else:
                discrete_action = torch.ones(
                    (*actions.shape[:-1], 1), device=actions.device, dtype=actions.dtype
                )
            actions = torch.cat([actions, discrete_action], dim=-1)
        return actions

    def forward(self, batch: dict[str, Tensor | dict[str, Tensor]]) -> dict[str, Tensor]:
        """Actor forward pass: sample actions and return log-probabilities.
        Training mode: accepts full sequence input [batch, seq_len, ...]
        Args:
            batch: A flat observation dict, or a training dict containing
                ``"state"`` (observations) and optionally ``"observation_feature"``
                (pre-computed encoder features).
        Returns:
            Dict with ``"action"``, ``"log_prob"``, and ``"action_mean"`` tensors.
        """
        observations = batch.get("state", batch)
        observation_features = batch.get("observation_feature") if isinstance(batch, dict) else None
        # 【GRU改造】训练模式：输入完整序列，不更新内部隐藏态
        actions, log_probs, means = self.actor(observations, observation_features, mode="train")
        return {"action": actions, "log_prob": log_probs, "action_mean": means}

    def _init_encoders(self):
        """Initialize shared or separate encoders for actor and critic."""
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = GaussianActorObservationEncoder(self.config)
        self.encoder_actor = (
            self.encoder_critic if self.shared_encoder else GaussianActorObservationEncoder(self.config)
        )

    def _init_actor(self, continuous_action_dim):
        """Initialize policy actor network with optional GRU temporal module."""
        # NOTE: The actor select only the continuous action part
        self.actor = Policy(
            encoder=self.encoder_actor,
            mlp_kwargs=asdict(self.config.actor_network_kwargs),  # 传入MLP配置，而非实例
            action_dim=continuous_action_dim,
            encoder_is_shared=self.shared_encoder,
            # 【GRU参数】从配置读取，缺失时使用默认值
            use_gru=getattr(self.config, "use_gru", False),
            gru_hidden_size=getattr(self.config, "gru_hidden_size", 64),
            num_gru_layers=getattr(self.config, "num_gru_layers", 2),
            gru_dropout=getattr(self.config, "gru_dropout", 0.1),
            **asdict(self.config.policy_kwargs),
        )

    def _init_discrete_critic(self) -> None:
        """Initialize discrete critic network."""
        if self.config.num_discrete_actions is None:
            self.discrete_critic = None
            return
        # TODO(Khalil): Compile the discrete critic
        self.discrete_critic = DiscreteCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )

    def update_hidden(self, batch: dict[str, Tensor]) -> None:
        """对外暴露的隐藏态更新接口，干预场景调用"""
        if not self.config.use_gru:
            return
        self.actor.update_hidden(batch)

    def load_state_dict(self, state_dict, strict=False):
        # 旧版键名自动映射兼容
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("encoder_actor."):
                new_key = k.replace("encoder_actor.", "actor.encoder.", 1)
                remapped[new_key] = v
            elif k.startswith("encoder_critic."):
                new_key = k.replace("encoder_critic.", "", 1)
                remapped[new_key] = v
            else:
                remapped[k] = v
        
        missing_keys, unexpected_keys = super().load_state_dict(remapped, strict=strict)
        # GRU参数缺失属于正常兼容场景，仅提示不告警
        gru_missing = [k for k in missing_keys if "gru" in k]
        other_missing = [k for k in missing_keys if "gru" not in k]
        if gru_missing:
            logging.info(f"检测到旧版无GRU权重，已自动兼容，缺失GRU参数共{len(gru_missing)}个")
        if other_missing:
            logging.warning(f"加载权重存在非预期缺失键：{other_missing[:5]}")
        if unexpected_keys:
            logging.warning(f"加载权重存在多余键：{unexpected_keys[:5]}")
        return missing_keys, unexpected_keys


class GaussianActorObservationEncoder(nn.Module):
    """Encode image and/or state vector observations.
    [GRU改造] 自动适配序列输入：输入带 seq_len 维度时，自动合并 batch 与时间维度编码后拆分
    """
    def __init__(self, config: GaussianActorConfig) -> None:
        super().__init__()
        self.config = config
        self._init_image_layers()
        self._init_state_layers()
        self._compute_output_dim()


    def _init_image_layers(self) -> None:
        self.image_keys = [k for k in self.config.input_features if is_image_feature(k)]
        self.has_images = bool(self.image_keys)
        if not self.has_images:
            return
        if self.config.vision_encoder_name is not None:
            self.image_encoder = PretrainedImageEncoder(self.config)
        else:
            self.image_encoder = DefaultImageEncoder(self.config)
        if self.config.freeze_vision_encoder:
            freeze_image_encoder(self.image_encoder)
        dummy = torch.zeros(1, *self.config.input_features[self.image_keys[0]].shape)
        with torch.no_grad():
            _, channels, height, width = self.image_encoder(dummy).shape
        self.spatial_embeddings = nn.ModuleDict()
        self.post_encoders = nn.ModuleDict()
        for key in self.image_keys:
            name = key.replace(".", "_")
            self.spatial_embeddings[name] = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channels,
                num_features=self.config.image_embedding_pooling_dim,
            )
            self.post_encoders[name] = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(
                    in_features=channels * self.config.image_embedding_pooling_dim,
                    out_features=self.config.latent_dim,
                ),
                nn.LayerNorm(normalized_shape=self.config.latent_dim),
                nn.Tanh(),
            )

    def _init_state_layers(self) -> None:
        self.has_env = OBS_ENV_STATE in self.config.input_features
        self.has_state = OBS_STATE in self.config.input_features
        if self.has_env:
            dim = self.config.input_features[OBS_ENV_STATE].shape[0]
            self.env_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )
        if self.has_state:
            dim = self.config.input_features[OBS_STATE].shape[0]
            self.state_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )

    def _compute_output_dim(self) -> None:
        out = 0
        if self.has_images:
            out += len(self.image_keys) * self.config.latent_dim
        if self.has_env:
            out += self.config.latent_dim
        if self.has_state:
            out += self.config.latent_dim
        self._out_dim = out

    def forward(
        self, obs: dict[str, Tensor], cache: dict[str, Tensor] | None = None, detach: bool = False
    ) -> Tensor:
        # ========== 【修复】统一校验所有可用观测的序列维度一致性 ==========
        has_seq_dim = None
        batch_shape = None

        # 1. 收集所有存在的观测键与对应维度约定
        check_pairs = []
        if self.has_images:
            for key in self.image_keys:
                if key in obs:
                    check_pairs.append((key, 5))  # 图像带序列维时为 5 维 [B,L,C,H,W]
        if self.has_env and OBS_ENV_STATE in obs:
            check_pairs.append((OBS_ENV_STATE, 3))  # 状态带序列维时为 3 维 [B,L,D]
        if self.has_state and OBS_STATE in obs:
            check_pairs.append((OBS_STATE, 3))

        # 2. 遍历校验所有观测维度是否一致
        for key, expected_seq_ndim in check_pairs:
            cur_ndim = obs[key].ndim
            cur_has_seq = cur_ndim == expected_seq_ndim
            if has_seq_dim is None:
                has_seq_dim = cur_has_seq
            elif has_seq_dim != cur_has_seq:
                raise ValueError(
                    f"观测维度不一致：{key} 的 ndim={cur_ndim}，"
                    f"是否带序列维={cur_has_seq}，与其他观测不匹配。请确保所有输入同时带/不带时间维度。"
                )

        # 兜底：无任何观测时默认无序列维
        if has_seq_dim is None:
            has_seq_dim = False

        # 3. 序列模式：合并 batch 与时间维度
        if has_seq_dim:
            first_key = next(iter(obs.keys()))
            batch_size, seq_len = obs[first_key].shape[:2]
            batch_shape = (batch_size, seq_len)
            obs = self._flatten_seq_dim(obs)
            if cache is not None:
                cache = {k: v.flatten(0, 1) for k, v in cache.items()}

        # ========== 以下原有编码逻辑保持不变 ==========
        parts = []
        if self.has_images:
            if cache is None:
                cache = self.get_cached_image_features(obs)
            image_feat = self._encode_images(cache, detach)
            parts.append(image_feat)
        
        if self.has_env:
            parts.append(self.env_encoder(obs[OBS_ENV_STATE]))
        if self.has_state:
            state_feat = self.state_encoder(obs[OBS_STATE])
            parts.append(state_feat)
        
        if parts:
            out = torch.cat(parts, dim=-1)
        else:
            raise ValueError("No parts to concatenate")

        # 数值兜底
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        out = torch.clamp(out, -50.0, 50.0)

        # 恢复序列维度
        if has_seq_dim and batch_shape is not None:
            out = out.view(*batch_shape, -1)
        return out



    # 【GRU改造新增】辅助方法：合并batch与seq维度
    def _flatten_seq_dim(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        flat_obs = {}
        for key, val in obs.items():
            # val shape: [B, L, ...] -> [B*L, ...]
            flat_obs[key] = val.flatten(0, 1)
        return flat_obs

    def get_cached_image_features(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Extract and optionally cache image features from observations.
        [GRU 修复] 兼容序列模式 5D 输入：自动合并 batch 与 seq_len 维度，编码后恢复
        """
        # 检测是否携带序列维度（图像为 5 维：B, L, C, H, W）
        first_img = obs[self.image_keys[0]]
        has_seq_dim = first_img.ndim == 5
        batch_shape = None

        if has_seq_dim:
            batch_size, seq_len = first_img.shape[:2]
            batch_shape = (batch_size, seq_len)
            # 合并 batch 与时间维度，转为 Conv2d 支持的 4D 格式
            obs = {k: v.flatten(0, 1) for k, v in obs.items()}

        # 原有的编码逻辑保持不变
        batched = torch.cat([obs[k] for k in self.image_keys], dim=0)
        out = self.image_encoder(batched)
        chunks = torch.chunk(out, len(self.image_keys), dim=0)
        features = dict(zip(self.image_keys, chunks, strict=False))

        # 恢复序列维度，与输入形状对齐
        if has_seq_dim and batch_shape is not None:
            features = {
                k: v.view(*batch_shape, *v.shape[1:])
                for k, v in features.items()
            }

        return features

    def _encode_images(self, cache: dict[str, Tensor], detach: bool) -> Tensor:
        """Encode image features from cached observations.
        This function takes pre-encoded image features from the cache and applies spatial embeddings and post-encoders.
        It also supports detaching the encoded features if specified.
        Args:
            cache (dict[str, Tensor]): The cached image features.
            detach (bool): Usually when the encoder is shared between actor and critics,
            we want to detach the encoded features on the policy side to avoid backprop through the encoder.
            More detail here `https://cdn.aaai.org/ojs/17276/17276-13-20770-1-2-20210518.pdf`
        Returns:
            Tensor: The encoded image features.
        """
        feats = []
        for k, feat in cache.items():
            safe_key = k.replace(".", "_")
            x = self.spatial_embeddings[safe_key](feat)
            x = self.post_encoders[safe_key](x)
            if detach:
                x = x.detach()
            feats.append(x)
        return torch.cat(feats, dim=-1)

    @property
    def output_dim(self) -> int:
        return self._out_dim


class MLP(nn.Module):
    """Multi-layer perceptron builder.
    Dynamically constructs a sequence of layers based on `hidden_dims`:
      1) Linear (in_dim -> out_dim)
      2) Optional Dropout if `dropout_rate` > 0 and (not final layer or `activate_final`)
      3) LayerNorm on the output features
      4) Activation (standard for intermediate layers, `final_activation` for last layer if `activate_final`)
    Arguments:
        input_dim (int): Size of input feature dimension.
        hidden_dims (list[int]): Sizes for each hidden layer.
        activations (Callable[[torch.Tensor], torch.Tensor] | str): Activation to apply between layers.
        activate_final (bool): Whether to apply activation at the final layer.
        dropout_rate (Optional[float]): Dropout probability applied before normalization and activation.
        final_activation (Optional[Callable[[torch.Tensor], torch.Tensor] | str]): Activation for the final layer when `activate_final` is True.
    For each layer, `in_dim` is updated to the previous `out_dim`. All constructed modules are
    stored in `self.net` as an `nn.Sequential` container.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        total = len(hidden_dims)
        for idx, out_dim in enumerate(hidden_dims):
            # 1) linear transform
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = idx == total - 1
            # 2-4) optionally add dropout, normalization, and activation
            if not is_last or activate_final:
                if dropout_rate and dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.LayerNorm(out_dim))
                act_cls = final_activation if is_last and final_activation else activations
                act = act_cls if isinstance(act_cls, nn.Module) else getattr(nn, act_cls)()
                layers.append(act)
            in_dim = out_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiscreteCritic(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 3,
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=self.output_dim)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(
        self, observations: torch.Tensor, observation_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        observations = {k: v.to(device) for k, v in observations.items()}
        obs_enc = self.encoder(observations, cache=observation_features)
        return self.output_layer(self.net(obs_enc))


class Policy(nn.Module):
    """Gaussian policy head with optional GRU temporal encoder
    对齐 sac_v2_gru.py 时序逻辑：观测特征 -> GRU时序建模 -> MLP -> 高斯动作分布
    支持两种模式：
    - train: 输入完整序列 [B, L, D]，支持传入初始隐藏态，默认零初始化
    - inference: 输入单步 [B, D]，维护内部隐藏态递推，用于环境交互
    """
    def __init__(
        self,
        encoder: GaussianActorObservationEncoder,
        mlp_kwargs: dict,  # MLP构造参数，替代原network参数
        action_dim: int,
        std_min: float = 1e-6,
        std_max: float = 10.0,

        fixed_std: torch.Tensor | None = None,
        init_final: float | None = None,
        use_tanh_squash: bool = False,
        encoder_is_shared: bool = False,
        # GRU时序模块参数
        use_gru: bool = False,
        gru_hidden_size: int = 64,
        num_gru_layers: int = 2,
        gru_dropout: float = 0.1,
        gripper_std_max: float | None = None,  # None表示不限制
    ):
        super().__init__()
        self.encoder: GaussianActorObservationEncoder = encoder
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max
        self.fixed_std = fixed_std
        self.use_tanh_squash = use_tanh_squash
        self.encoder_is_shared = encoder_is_shared

        # GRU时序模块配置
        self.use_gru = use_gru
        self.gru_hidden_size = gru_hidden_size
        self.num_gru_layers = num_gru_layers
        self._hidden_state: Tensor | None = None  # 推理时维护的内部隐藏态

        # 计算MLP输入维度：启用GRU则为GRU隐藏维度，否则为编码器输出维度
        if self.use_gru:
            self.gru = nn.GRU(
                input_size=encoder.output_dim,
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
            mlp_input_dim = encoder.output_dim

        # 【修复】直接构造MLP，不再先建后改第一层
        self.network = MLP(input_dim=mlp_input_dim, **mlp_kwargs)

        # 获取MLP最后一层输出维度
        for layer in reversed(self.network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break

        # Mean 输出层
        self.mean_layer = nn.Linear(out_features, action_dim)
        if init_final is not None:
            nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.mean_layer.weight)

        # Std 输出层
        if fixed_std is None:
            self.std_layer = nn.Linear(out_features, action_dim)
            if init_final is not None:
                nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
                nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
            else:
                orthogonal_init()(self.std_layer.weight)
        self.gripper_std_max = gripper_std_max
    @property
    def hidden_state(self) -> Tensor | None:
        """获取当前推理模式下的GRU隐藏态"""
        return self._hidden_state

    @hidden_state.setter
    def hidden_state(self, value: Tensor) -> None:
        """设置GRU隐藏态，用于干预、权重更新等外部场景"""
        if not self.use_gru:
            return
        self._hidden_state = value

    def reset_hidden(self, batch_size: int = 1):
        """Reset GRU hidden state to zero. Called at episode start for inference.
        Args:
            batch_size: 并行推理的环境数量，单环境默认1
        """
        if not self.use_gru:
            return
        device = get_device_from_parameters(self)
        self._hidden_state = torch.zeros(
            self.num_gru_layers, batch_size, self.gru_hidden_size, device=device
        )
        # 强制与模型参数同 dtype，避免类型不匹配导致的数值异常
        self._hidden_state = self._hidden_state.to(dtype=get_dtype_from_parameters(self))

    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None = None,
        mode: str = "train",
        hidden_in: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 编码器前向：共享编码器时detach梯度
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
        # 编码器输出数值兜底：过滤NaN/Inf，限制范围
        # ===== 新增：编码器输出异常统计 =====
        #self.enc_out_tracker.track(obs_enc, desc=f"mode={mode}")
        obs_enc = torch.nan_to_num(obs_enc, nan=0.0, posinf=10.0, neginf=-10.0)

        #print(f"[逐层调试] 编码器输出 | min={obs_enc.min():.4f} max={obs_enc.max():.4f} mean={obs_enc.mean():.4f}")
        # GRU时序处理分支
        if self.use_gru:
            if mode == "train":
                if not hasattr(self, '_flattened'):
                    self.gru.flatten_parameters()
                    setattr(self, '_flattened', True)
                init_hidden = hidden_in if hidden_in is not None else None
                # 训练模式：GRU输入统计
                #self.gru_in_tracker.track(obs_enc, desc="train_input")
                gru_out, hidden_out = self.gru(obs_enc, init_hidden)
                
                # 训练模式：GRU输出与隐藏态统计
                #self.gru_out_tracker.track(gru_out, desc="train_output")
                #self.hidden_tracker.track(hidden_out, desc="train_hidden")
                net_in = gru_out
            elif mode == "inference":
                # 隐藏态异常时自动重置
                # 隐藏态异常检测保留并增强
                # if self._hidden_state is not None:
                #     self.hidden_tracker.track(self._hidden_state, desc="inference_hidden_before")
                #     if torch.isnan(self._hidden_state).any() or torch.isinf(self._hidden_state).any():
                #         print("[WARNING] GRU hidden state has NaN/Inf, resetting...")
                #         self.reset_hidden(batch_size=obs_enc.shape[0])

                # # 新增：打印隐藏态与输入特征的范数，判断是否坍缩
                # #print(f"[GRU调试] 输入特征范数: {obs_enc.norm():.4f}")
                # #print(f"[GRU调试] 隐藏态范数: {self._hidden_state.norm():.4f}")
                # # ========== 新增：GRU 输入统计 ==========
                # self.gru_in_tracker.track(obs_enc, desc="inference_input")
                # ============================================================
                obs_enc = obs_enc.unsqueeze(1)  # [B, 1, D]
                gru_out, new_hidden = self.gru(obs_enc, self._hidden_state)
                # 新增：打印GRU输出范数
                #print(f"[GRU调试] GRU输出范数: {gru_out.norm():.4f}")
                # GRU 输出与新隐藏态统计
                # self.gru_out_tracker.track(gru_out, desc="inference_output")
                # self.hidden_tracker.track(new_hidden, desc="inference_hidden_after")
                # ========== 新增：NaN/Inf 二次校验，异常则用零替换 ==========
                new_hidden = torch.nan_to_num(new_hidden, nan=0.0, posinf=20.0, neginf=-20.0)
                gru_out = torch.nan_to_num(gru_out, nan=0.0, posinf=20.0, neginf=-20.0)
                # ============================================================
                self._hidden_state = new_hidden
                net_in = gru_out.squeeze(1)  # [B, H]
                hidden_out = new_hidden
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'inference'.")
            #print(f"[调试] GRU输出 | min={gru_out.min():.4f} max={gru_out.max():.4f} mean={gru_out.mean():.4f}")
        else:
            net_in = obs_enc
            hidden_out = None

        # MLP + 高斯分布头
        outputs = self.network(net_in)
        #print(f"[调试] MLP输出 | min={outputs.min():.4f} max={outputs.max():.4f} mean={outputs.mean():.4f}")
        # 新增：MLP输出数值裁剪
        # 新增：MLP输出异常统计
        #self.action_head_tracker.track(outputs, desc="mlp_output")
        
        means = self.mean_layer(outputs)
        #print(f"[调试] 动作均值 | min={means.min():.4f} max={means.max():.4f} mean={means.mean():.4f}")
        # 均值裁剪到合理动作范围，避免输出极端动作导致环境发散
        #self.action_head_tracker.track(means, desc="action_mean")

        # 计算标准差
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            #self.action_head_tracker.track(log_std, desc="log_std")
            std = torch.exp(log_std)
            # 在原始空间做范围限制，与配置参数语义完全对齐
            std = torch.clamp(std, self.std_min, self.std_max)
            # ========== 新增：单独限制夹爪维度的标准差上限 ==========
            # 夹爪为第4维（索引3），降低其探索噪声，抑制随机开合
            # 修改后
            # 连续动作的最后一维为夹爪维度，动态推导
            # ========== 替换后（非原地，保留完整计算图）==========
            # 仅当配置了夹爪标准差上限时才执行裁剪
            if self.gripper_std_max is not None and self.action_dim > 0:
                gripper_dim_idx = self.action_dim - 1
                std_other = std[..., :gripper_dim_idx]
                std_gripper = std[..., gripper_dim_idx:]
                std_gripper = torch.clamp(std_gripper, max=self.gripper_std_max)
                std = torch.cat([std_other, std_gripper], dim=-1)
        else:
            std = self.fixed_std.expand_as(means)
        # 分布输入异常统计（复用动作头追踪器，长效累积）
        # self.action_head_tracker.track(means, desc="dist_loc")
        # self.action_head_tracker.track(std, desc="dist_scale")
        # 构建高斯分布：根据开关决定是否启用tanh压缩
        if self.use_tanh_squash:
            dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)
        else:
            dist = Independent(Normal(loc=means, scale=std), reinterpreted_batch_ndims=1)
        actions = dist.rsample()  # 重参数化采样
        log_probs = dist.log_prob(actions)

        if return_hidden:
            return actions, log_probs, means, hidden_out
        return actions, log_probs, means

    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get encoded features from observations"""
        device = get_device_from_parameters(self)
        observations = observations.to(device)
        if self.encoder is not None:
            with torch.inference_mode():
                return self.encoder(observations)
        return observations

    @torch.no_grad()
    def update_hidden(self, observations: dict[str, Tensor]) -> None:
        """
        仅更新 GRU 隐藏态，不输出动作。
        数值处理与推理模式完全一致，保证隐藏态分布对齐。
        """
        if not self.use_gru:
            return
        
        # 与推理路径完全一致的编码 + 数值兜底
        obs_enc = self.encoder(observations, detach=self.encoder_is_shared)
        obs_enc = torch.nan_to_num(obs_enc, nan=0.0, posinf=10.0, neginf=-10.0)

        # 隐藏态异常自动重置（与推理路径一致）
        if self._hidden_state is not None and (torch.isnan(self._hidden_state).any() or torch.isinf(self._hidden_state).any()):
            self.reset_hidden(batch_size=obs_enc.shape[0])
        if self._hidden_state is None:
            self.reset_hidden(batch_size=obs_enc.shape[0])

        # 单步 GRU 前向，与推理路径处理完全一致
        obs_enc = obs_enc.unsqueeze(1)
        _, new_hidden = self.gru(obs_enc, self._hidden_state)
        
        # 与推理路径对齐的 NaN/Inf 兜底，不额外强裁剪
        new_hidden = torch.nan_to_num(new_hidden, nan=0.0, posinf=20.0, neginf=-20.0)
        self._hidden_state = new_hidden



class DefaultImageEncoder(nn.Module):
    def __init__(self, config: GaussianActorConfig):
        super().__init__()
        image_key = next(key for key in config.input_features if is_image_feature(key))
        self.image_enc_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=config.input_features[image_key].shape[0],
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=7,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=5,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.image_enc_layers(x)
        return x


def freeze_image_encoder(image_encoder: nn.Module):
    """Freeze all parameters in the encoder"""
    for param in image_encoder.parameters():
        param.requires_grad = False


class PretrainedImageEncoder(nn.Module):
    def __init__(self, config: GaussianActorConfig):
        super().__init__()
        self.image_enc_layers, self.image_enc_out_shape = self._load_pretrained_vision_encoder(config)

    def _load_pretrained_vision_encoder(self, config: GaussianActorConfig):
        """Set up CNN encoder"""
        from transformers import AutoModel
        self.image_enc_layers = AutoModel.from_pretrained(config.vision_encoder_name, trust_remote_code=True)
        if hasattr(self.image_enc_layers.config, "hidden_sizes"):
            self.image_enc_out_shape = self.image_enc_layers.config.hidden_sizes[-1]  # Last channel dimension
        elif hasattr(self.image_enc_layers, "fc"):
            self.image_enc_out_shape = self.image_enc_layers.fc.in_features
        else:
            raise ValueError("Unsupported vision encoder architecture, make sure you are using a CNN")
        return self.image_enc_layers, self.image_enc_out_shape

    def forward(self, x):
        enc_feat = self.image_enc_layers(x).last_hidden_state
        return enc_feat


def orthogonal_init():
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        PyTorch implementation of learned spatial embeddings
        Args:
            height: Spatial height of input features
            width: Spatial width of input features
            channel: Number of input channels
            num_features: Number of output embedding dimensions
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features
        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))
        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        """
        Forward pass for spatial embedding
        【GRU 修复】适配任意前导 batch 维度（支持序列模式 [B, L, C, H, W]）
        Args:
            features: Input tensor of shape [..., C, H, W]
                    单步模式: [B, C, H, W]
                    序列模式: [B, L, C, H, W]
        Returns:
            Output tensor of shape [..., C*F]
        """
        features_expanded = features.unsqueeze(-1)  # [..., C, H, W, 1]
        kernel_expanded = self.kernel.unsqueeze(0)  # [1, C, H, W, F]
        
        # Element-wise multiplication and spatial reduction
        # 对倒数第3、倒数第2维（H、W）求和，自动兼容单步/序列模式
        output = (features_expanded * kernel_expanded).sum(dim=(-3, -2))  # [..., C, F]
        
        # 保留所有前导维度，展平通道与特征维度
        output = output.view(*output.shape[:-2], -1)  # [..., C*F]
        return output



class RescaleFromTanh(Transform):
    def __init__(self, low: float = -1, high: float = 1):
        super().__init__()
        self.low = low
        self.high = high

    def _call(self, x):
        # Rescale from (-1, 1) to (low, high)
        return 0.5 * (x + 1.0) * (self.high - self.low) + self.low

    def _inverse(self, y):
        # Rescale from (low, high) back to (-1, 1)
        return 2.0 * (y - self.low) / (self.high - self.low) - 1.0

    def log_abs_det_jacobian(self, x, y):
        # log|d(rescale)/dx| = sum(log(0.5 * (high - low)))
        scale = 0.5 * (self.high - self.low)
        return torch.sum(torch.log(scale), dim=-1)


class TanhMultivariateNormalDiag(TransformedDistribution):
    def __init__(self, loc, scale_diag, low=None, high=None):
        # 仅做 NaN/Inf 兜底，不截断分布均值，保证熵与 log_prob 计算准确
        loc = torch.nan_to_num(loc, nan=0.0)
        scale_diag = torch.nan_to_num(scale_diag, nan=1e-2)
        scale_diag = torch.clamp_min(scale_diag, 1e-4)
        
        base_dist = Independent(Normal(loc=loc, scale=scale_diag), reinterpreted_batch_ndims=1)
        transforms = [TanhTransform(cache_size=0)]  # 关闭缓存避免数值累积误差
        if low is not None and high is not None:
            low = torch.as_tensor(low)
            high = torch.as_tensor(high)
            transforms.insert(0, RescaleFromTanh(low, high))
        super().__init__(base_dist, transforms)

    def log_prob(self, value):
        """数值稳定实现：用cosh公式替代1-tanh²，避免饱和时出现-inf"""
        # 逆变换回高斯空间
        x = value
        for transform in reversed(self.transforms):
            x = transform.inv(x)
        
        # 基础高斯对数概率
        base_log_prob = self.base_dist.log_prob(x)
        
        # 数值稳定计算Tanh雅可比行列式：log(1-tanh²x) = -2*log(coshx)
        tanh_jacobian = -2.0 * torch.log(torch.cosh(x) + 1e-8)
        tanh_jacobian = tanh_jacobian.sum(dim=-1)
        
        # 缩放变换雅可比（如有）
        rescale_log_det = 0.0
        for transform in self.transforms:
            if isinstance(transform, RescaleFromTanh):
                rescale_log_det = transform.log_abs_det_jacobian(x, value)
                break
        
        return base_log_prob + tanh_jacobian + rescale_log_det




    def mode(self):
        x = self.base_dist.base_dist.mean
        for transform in self.transforms:
            x = transform(x)
        return x

    def stddev(self, num_samples: int = 10000):
        """
        Tanh变换后的分布无解析标准差，通过采样近似计算。
        Args:
            num_samples: 采样数量，越大精度越高
        """
        # 从基础高斯分布采样
        samples = self.base_dist.sample(torch.Size([num_samples]))
        # 执行全部变换（tanh + 缩放）
        for transform in self.transforms:
            samples = transform(samples)
        # 样本标准差
        return samples.std(dim=0)

