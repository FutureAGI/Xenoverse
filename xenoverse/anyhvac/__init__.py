#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from gym.envs.registration import register
from xenoverse.anyhvac.anyhvac_env import HVACEnv
from xenoverse.anyhvac.anyhvac_env_vis import HVACEnvVisible
from xenoverse.anyhvac.anyhvac_solver import HVACSolverGTPID

register(
    id='anyhvac-v0',
    entry_point='xenoverse.anyhvac.anyhvac_env:HVACEnv',
    kwargs={"max_steps": 86400,
            "target_temperature": 28,
            "upper_limit": 80,
            "iter_per_step": 600},
)

register(
    id='anyhvac-visualizer-v0',
    entry_point='xenoverse.anyhvac.anyhvac_env:HVACEnvVisible',
    kwargs={"max_steps": 86400,
            "target_temperature": 28,
            "upper_limit": 80,
            "iter_per_step": 600},
)