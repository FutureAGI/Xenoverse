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

from gymnasium.envs.registration import register
from xenoverse.linds.linds_env import LinearDSEnv
from xenoverse.linds.task_sampler import LinearDSSampler
from xenoverse.linds.visualizer import LinearDSVisualizer

register(
    id='linear-dynamics-v0',
    entry_point='xenoverse.linds:LinearDSEnv',
    order_enforce=False,
    disable_env_checker=True,
    kwargs={"max_steps": 5000},
)

register(
    id='linear-dynamics-v0-visualizer',
    entry_point='xenoverse.linds:LinearDSVisualizer',
    order_enforce=False,
    disable_env_checker=True,
    kwargs={"max_steps": 5000},
)