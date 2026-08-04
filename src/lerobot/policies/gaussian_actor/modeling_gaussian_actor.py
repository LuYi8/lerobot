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
        # print("=== 观测输入校验 ===")
        # for k, v in batch.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"{k}: shape={list(v.shape)}, min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")
                # ===== 新增：观测NaN清洗 =====
        for key in batch:
            if torch.isnan(batch[key]).any():
                # 打印异常维度信息，方便后续定位
                nan_mask = torch.isnan(batch[key])
                print(f"[WARNING] NaN detected in obs key: {key}, count: {nan_mask.sum().item()}")
                # 用0填充NaN，避免后续计算全崩
                batch[key] = torch.nan_to_num(batch[key], nan=0.0, posinf=1e3, neginf=-1e3)
        # ==========================
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

    def load_state_dict(self, state_dict, strict=True):
        # 旧版键名自动映射兼容
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("encoder_actor."):
                # 旧版顶层actor编码器 -> 映射到actor模块内部的encoder
                new_key = k.replace("encoder_actor.", "actor.encoder.", 1)
                remapped[new_key] = v
            elif k.startswith("encoder_critic."):
                # 旧版顶层critic编码器 -> 映射到当前顶层encoder_critic
                new_key = k.replace("encoder_critic.", "encoder_critic.", 1)
                remapped[new_key] = v
            else:
                remapped[k] = v
        
        return super().load_state_dict(remapped, strict=strict)

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
        # 保留原有维度检测、flatten 逻辑
        has_seq_dim = False
        batch_shape = None
        if self.has_state and OBS_STATE in obs:
            has_seq_dim = obs[OBS_STATE].ndim == 3
        elif self.has_env and OBS_ENV_STATE in obs:
            has_seq_dim = obs[OBS_ENV_STATE].ndim == 3
        elif self.has_images:
            first_key = self.image_keys[0]
            has_seq_dim = obs[first_key].ndim == 5
        if has_seq_dim:
            batch_size, seq_len = obs[next(iter(obs.keys()))].shape[:2]
            batch_shape = (batch_size, seq_len)
            obs = self._flatten_seq_dim(obs)
            if cache is not None:
                cache = {k: v.flatten(0, 1) for k, v in cache.items()}

        parts = []
        if self.has_images:
            if cache is None:
                cache = self.get_cached_image_features(obs)
            
            # 新增：打印图像编码器原始输出
            first_img_feat = next(iter(cache.values()))
            #print(f"[编码器调试] ResNet原始输出 | min={first_img_feat.min():.4f} max={first_img_feat.max():.4f} mean={first_img_feat.mean():.4f}")
            
            image_feat = self._encode_images(cache, detach)
            #print(f"[编码器调试] 图像后处理输出 | min={image_feat.min():.4f} max={image_feat.max():.4f} mean={image_feat.mean():.4f}")
            parts.append(image_feat)
        
        if self.has_env:
            parts.append(self.env_encoder(obs[OBS_ENV_STATE]))
        if self.has_state:
            state_feat = self.state_encoder(obs[OBS_STATE])
            #print(f"[编码器调试] 状态编码器输出 | min={state_feat.min():.4f} max={state_feat.max():.4f} mean={state_feat.mean():.4f}")
            parts.append(state_feat)
        
        if parts:
            out = torch.cat(parts, dim=-1)
        else:
            raise ValueError("No parts to concatenate")
        # ========== 新增：编码器输出数值兜底，训练/推理全链路生效 ==========
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        out = torch.clamp(out, -50.0, 50.0)
        # ================================================================
        #print(f"[编码器调试] 最终拼接输出 | min={out.min():.4f} max={out.max():.4f} mean={out.mean():.4f}")
        
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
        std_min: float = -5,
        std_max: float = 2,
        fixed_std: torch.Tensor | None = None,
        init_final: float | None = None,
        use_tanh_squash: bool = False,
        encoder_is_shared: bool = False,
        # GRU时序模块参数
        use_gru: bool = False,
        gru_hidden_size: int = 64,
        num_gru_layers: int = 2,
        gru_dropout: float = 0.1,
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
            for name, param in self.gru.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=0.5)
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
        mode: str = "train",  # "train" | "inference"
        hidden_in: torch.Tensor | None = None,  # 训练模式可选初始隐藏态
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 编码器前向：共享编码器时detach梯度
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
        # 编码器输出数值兜底：过滤NaN/Inf，限制范围
        obs_enc = torch.nan_to_num(obs_enc, nan=0.0, posinf=10.0, neginf=-10.0)
        obs_enc = torch.clamp(obs_enc, -100.0, 100.0)
        #print(f"[逐层调试] 编码器输出 | min={obs_enc.min():.4f} max={obs_enc.max():.4f} mean={obs_enc.mean():.4f}")
        # GRU时序处理分支
        if self.use_gru:
            if mode == "train":
                # 训练模式：输入完整序列 [B, L, D]
                if not hasattr(self, '_flattened'):
                    self.gru.flatten_parameters()
                    setattr(self, '_flattened', True)
                # 未传入初始隐藏态则零初始化
                init_hidden = hidden_in if hidden_in is not None else None
                gru_out, _ = self.gru(obs_enc, init_hidden)  # [B, L, H]
                net_in = gru_out
            elif mode == "inference":
                # 隐藏态异常时自动重置
                if self._hidden_state is not None and (torch.isnan(self._hidden_state).any() or torch.isinf(self._hidden_state).any()):
                    print("[WARNING] GRU hidden state has NaN/Inf, resetting...")
                    self.reset_hidden(batch_size=obs_enc.shape[0])

                if self._hidden_state is None:
                    batch_size = obs_enc.shape[0]
                    self.reset_hidden(batch_size=batch_size)
                # 新增：打印隐藏态与输入特征的范数，判断是否坍缩
                #print(f"[GRU调试] 输入特征范数: {obs_enc.norm().item():.4f}")
                #print(f"[GRU调试] 隐藏态范数: {self._hidden_state.norm().item():.4f}")
                # ========== 新增：GRU 输入强裁剪，从源头抑制数值爆炸 ==========
                obs_enc = torch.clamp(obs_enc, -10.0, 10.0)
                # ============================================================
                obs_enc = obs_enc.unsqueeze(1)  # [B, 1, D]
                gru_out, new_hidden = self.gru(obs_enc, self._hidden_state)
                # 新增：打印GRU输出范数
                #print(f"[GRU调试] GRU输出范数: {gru_out.norm().item():.4f}")
                # 新增：强制裁剪隐藏态与输出，彻底杜绝数值爆炸
                new_hidden = torch.clamp(new_hidden, -50.0, 50.0)
                gru_out = torch.clamp(gru_out, -50.0, 50.0)
                # ========== 新增：NaN/Inf 二次校验，异常则用零替换 ==========
                new_hidden = torch.nan_to_num(new_hidden, nan=0.0, posinf=20.0, neginf=-20.0)
                gru_out = torch.nan_to_num(gru_out, nan=0.0, posinf=20.0, neginf=-20.0)
                # ============================================================
                self._hidden_state = new_hidden
                net_in = gru_out.squeeze(1)  # [B, H]
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'inference'.")
            #print(f"[调试] GRU输出 | min={gru_out.min():.4f} max={gru_out.max():.4f} mean={gru_out.mean():.4f}")
        else:
            net_in = obs_enc

        # MLP + 高斯分布头
        outputs = self.network(net_in)
        #print(f"[调试] MLP输出 | min={outputs.min():.4f} max={outputs.max():.4f} mean={outputs.mean():.4f}")
        # 新增：MLP输出数值裁剪
        outputs = torch.clamp(outputs, -100.0, 100.0)
        
        means = self.mean_layer(outputs)
        #print(f"[调试] 动作均值 | min={means.min():.4f} max={means.max():.4f} mean={means.mean():.4f}")
        # 均值裁剪到合理动作范围，避免输出极端动作导致环境发散
        means = torch.clamp(means, -5.0, 5.0)

        # 计算标准差
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            # 先在对数空间做 clamp，保证 exp 后标准差严格为正，对齐参数原本的语义
            log_std = torch.clamp(log_std, self.std_min, self.std_max)
            std = torch.exp(log_std)
            # 双重保险：强制加极小下限，彻底避免极端数值下溢
            std = torch.clamp_min(std, 1e-6)
        else:
            std = self.fixed_std.expand_as(means)

        # 构建tanh压缩的高斯分布
        dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)
        actions = dist.rsample()  # 重参数化采样
        log_probs = dist.log_prob(actions)

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
        用于人类干预、外部状态同步等场景，保证隐藏态与真实轨迹对齐。
        Args:
            observations: 经过归一化预处理的观测字典，与 select_action 输入格式一致
        """
        if not self.use_gru:
            return
        
        # 与推理模式完全一致的编码流程，保证数值分布对齐
        obs_enc = self.encoder(observations, detach=self.encoder_is_shared)
        obs_enc = torch.nan_to_num(obs_enc, nan=0.0, posinf=10.0, neginf=-10.0)
        obs_enc = torch.clamp(obs_enc, -100.0, 100.0)

        # 隐藏态异常自动重置
        if self._hidden_state is not None and (torch.isnan(self._hidden_state).any() or torch.isinf(self._hidden_state).any()):
            self.reset_hidden(batch_size=obs_enc.shape[0])
        if self._hidden_state is None:
            self.reset_hidden(batch_size=obs_enc.shape[0])

        # 单步 GRU 前向，只更新隐藏态，不后续计算动作
        obs_enc = obs_enc.unsqueeze(1)
        _, new_hidden = self.gru(obs_enc, self._hidden_state)
        new_hidden = torch.clamp(new_hidden, -50.0, 50.0)
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
        # 数值兜底：过滤NaN/Inf，保证标准差严格为正
        loc = torch.nan_to_num(loc, nan=0.0, posinf=1.0, neginf=-1.0)
        scale_diag = torch.nan_to_num(scale_diag, nan=1e-2, posinf=1.0, neginf=1e-2)
        scale_diag = torch.clamp_min(scale_diag, 1e-4)
        
        # 对角高斯用Independent+Normal，无矩阵分解，数值更稳定
        base_dist = Independent(Normal(loc=loc, scale=scale_diag), reinterpreted_batch_ndims=1)
        transforms = [TanhTransform(cache_size=1)]
        if low is not None and high is not None:
            low = torch.as_tensor(low)
            high = torch.as_tensor(high)
            transforms.insert(0, RescaleFromTanh(low, high))
        super().__init__(base_dist, transforms)

    def mode(self):
        x = self.base_dist.base_dist.mean
        for transform in self.transforms:
            x = transform(x)
        return x

    def stddev(self):
        std = self.base_dist.base_dist.stddev
        x = std
        for transform in self.transforms:
            x = transform(x)
        return x
