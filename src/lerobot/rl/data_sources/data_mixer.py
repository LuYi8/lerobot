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

import abc

from lerobot.types import BatchType

from ..buffer import ReplayBuffer, concatenate_batch_transitions


class DataMixer(abc.ABC):
    """Abstract interface for all data mixing strategies."""

    @abc.abstractmethod
    def sample(self, batch_size: int, sequence_length: int = 1) -> BatchType:
        """Draw one batch of ``batch_size`` transitions."""
        raise NotImplementedError

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
        sequence_length: int = 1,
    ):
        """Infinite iterator that yields batches."""
        while True:
            yield self.sample(batch_size, sequence_length)


class OnlineOfflineMixer(DataMixer):
    """Mixes transitions from an online and an offline replay buffer."""

    def __init__(
        self,
        online_buffer: ReplayBuffer,
        offline_buffer: ReplayBuffer | None = None,
        online_ratio: float = 1.0,
    ):
        if not 0.0 <= online_ratio <= 1.0:
            raise ValueError(f"online_ratio must be in [0, 1], got {online_ratio}")
        self.online_buffer = online_buffer
        self.offline_buffer = offline_buffer
        self.online_ratio = online_ratio

    #修改 ============ GRU：sequence_length 透传 ============
    # 序列采样时两个 buffer 各采样 n_online/n_offline 条序列（每条 T 帧），
    # concatenate 沿 dim0 拼接 (B, T, ...) 天然有效，总帧数 = batch_size × T 不变。
    #结束 ============================================
    def sample(self, batch_size: int, sequence_length: int = 1) -> BatchType:
        if self.offline_buffer is None:
            return self.online_buffer.sample(batch_size, sequence_length)

        n_online = max(1, int(batch_size * self.online_ratio))
        n_offline = batch_size - n_online

        online_batch = self.online_buffer.sample(n_online, sequence_length)
        offline_batch = self.offline_buffer.sample(n_offline, sequence_length)
        return concatenate_batch_transitions(online_batch, offline_batch)

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
        sequence_length: int = 1,
    ):
        """Yield batches by composing buffer async iterators."""

        n_online = max(1, int(batch_size * self.online_ratio))

        online_iter = self.online_buffer.get_iterator(
            batch_size=n_online,
            async_prefetch=async_prefetch,
            queue_size=queue_size,
            sequence_length=sequence_length,
        )

        if self.offline_buffer is None:
            yield from online_iter
            return

        n_offline = batch_size - n_online
        offline_iter = self.offline_buffer.get_iterator(
            batch_size=n_offline,
            async_prefetch=async_prefetch,
            queue_size=queue_size,
            sequence_length=sequence_length,
        )

        while True:
            yield concatenate_batch_transitions(next(online_iter), next(offline_iter))
