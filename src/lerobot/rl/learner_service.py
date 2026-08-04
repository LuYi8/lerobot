# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
import hashlib

import logging
import time
from multiprocessing import Event, Queue
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _grpc_available

from .queue import get_last_item_from_queue

if TYPE_CHECKING or _grpc_available:
    import grpc

    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import receive_bytes_in_chunks, send_bytes_in_chunks

    _ServicerBase = services_pb2_grpc.LearnerServiceServicer
else:
    grpc = None
    services_pb2 = None
    services_pb2_grpc = None
    receive_bytes_in_chunks = None
    send_bytes_in_chunks = None
    _ServicerBase = object

MAX_WORKERS = 3  # Stream parameters, send transitions and interactions
SHUTDOWN_TIMEOUT = 10


class LearnerService(_ServicerBase):
    """
    Implementation of the LearnerService gRPC service
    This service is used to send parameters to the Actor and receive transitions and interactions from the Actor
    check transport.proto for the gRPC service definition
    """

    def __init__(
        self,
        shutdown_event: Event,
        parameters_queue: Queue,
        seconds_between_pushes: float,
        transition_queue: Queue,
        interaction_message_queue: Queue,
        queue_get_timeout: float = 0.001,
    ):
        self.shutdown_event = shutdown_event
        self.parameters_queue = parameters_queue
        self.seconds_between_pushes = seconds_between_pushes
        self.transition_queue = transition_queue
        self.interaction_message_queue = interaction_message_queue
        self.queue_get_timeout = queue_get_timeout
        # 【新增】缓存上一次发送的权重哈希，用于去重
        self._last_params_hash = None


    def StreamParameters(self, request: "services_pb2.Empty", context: "grpc.ServicerContext"):
        logging.info("[LEARNER] Received request to stream parameters from the Actor")
        last_push_time = 0
        while not self.shutdown_event.is_set():
            time_since_last_push = time.time() - last_push_time
            if time_since_last_push < self.seconds_between_pushes:
                self.shutdown_event.wait(self.seconds_between_pushes - time_since_last_push)
                continue

            # 从队列取最新权重
            buffer = get_last_item_from_queue(
                self.parameters_queue, block=True, timeout=self.queue_get_timeout
            )
            if buffer is None:
                continue

            # 计算当前权重哈希，和上一次对比
            current_hash = hashlib.md5(buffer).hexdigest()
            if current_hash == self._last_params_hash:
                # 权重未变化，跳过发送，不打日志
                last_push_time = time.time()
                continue

            # 权重有更新，才执行推送
            logging.debug("[LEARNER] Push parameters to the Actor")
            yield from send_bytes_in_chunks(
                buffer,
                services_pb2.Parameters,
                log_prefix="[LEARNER] Sending parameters",
                silent=True,
            )
            self._last_params_hash = current_hash
            last_push_time = time.time()
            logging.debug("[LEARNER] Parameters sent")

        logging.info("[LEARNER] Stream parameters finished")
        return services_pb2.Empty()


    def SendTransitions(self, request_iterator, _context: "grpc.ServicerContext"):  # noqa: N802
        # TODO: authorize the request
        logging.info("[LEARNER] Received request to receive transitions from the Actor")

        receive_bytes_in_chunks(
            request_iterator,
            self.transition_queue,
            self.shutdown_event,
            log_prefix="[LEARNER] transitions",
        )

        logging.debug("[LEARNER] Finished receiving transitions")
        return services_pb2.Empty()

    def SendInteractions(self, request_iterator, _context: "grpc.ServicerContext"):  # noqa: N802
        # TODO: authorize the request
        logging.info("[LEARNER] Received request to receive interactions from the Actor")

        receive_bytes_in_chunks(
            request_iterator,
            self.interaction_message_queue,
            self.shutdown_event,
            log_prefix="[LEARNER] interactions",
        )

        logging.debug("[LEARNER] Finished receiving interactions")
        return services_pb2.Empty()

    def Ready(self, request: "services_pb2.Empty", context: "grpc.ServicerContext"):  # noqa: N802
        return services_pb2.Empty()
