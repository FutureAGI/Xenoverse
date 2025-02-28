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
from xenoverse.mazeworld.envs import MazeWorldContinuous3D
from xenoverse.mazeworld.envs import MazeTaskSampler, Resampler

register(
    id='mazeworld-v2',
    entry_point='xenoverse.mazeworld:MazeWorldContinuous3D',
    kwargs={
        "enable_render": True,
        "render_scale": 480,
        "resolution": (256, 256),
        "max_steps": 5000,
        "visibility_3D": 12.0,
        "command_in_observation": False,
        "action_space_type": "Discrete16" ,
    }
)
