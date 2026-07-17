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

import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

import gym_hil  # noqa: F401


@pytest.mark.parametrize(
    "env_task, image_obs",
    [
        ("PandaPickCubeBase-v0", False),
        ("PandaPickCubeBase-v0", True),
    ],
)
def test_hil(env_task, image_obs):
    env = gym.make(f"gym_hil/{env_task}", image_obs=image_obs)
    check_env(env.unwrapped)
