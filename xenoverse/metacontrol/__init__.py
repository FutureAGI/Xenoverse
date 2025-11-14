#   Copyright (c) Xenoverse. All Rights Reserved.
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
from xenoverse.metacontrol.random_cartpole import sample_cartpole,RandomCartPoleEnv
from xenoverse.metacontrol.random_acrobot import sample_acrobot, RandomAcrobotEnv
from xenoverse.metacontrol.random_humanoid import RandomHumanoidEnv, sample_humanoid, get_humanoid_tasks

register(
    id='random-cartpole-v0',
    entry_point='xenoverse.metacontrol.random_cartpole:RandomCartPoleEnv',
    order_enforce=False,
    disable_env_checker=True,
    kwargs={"frameskip":1, "reset_bounds_scale":[0.45, 0.90, 0.13, 1.0]}
)
register(
    id='random-acrobot-v0',
    entry_point='xenoverse.metacontrol.random_acrobot:RandomAcrobotEnv',
    order_enforce=False,
    disable_env_checker=True,
    kwargs={"frameskip":1, "reset_bounds_scale":0.10}
)
register(
    id='random-humanoid-v0',
    entry_point='xenoverse.metacontrol.random_humanoid:RandomHumanoidEnv',
    order_enforce=False,
    disable_env_checker=True,
    kwargs={}
)
