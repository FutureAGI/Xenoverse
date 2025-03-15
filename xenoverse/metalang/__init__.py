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
from xenoverse.metalang.metalangv1 import MetaLangV1
from xenoverse.metalang.metalangv2 import MetaLangV2
from xenoverse.metalang.metalangv3 import MetaLMV3Env
from xenoverse.metalang.task_sampler import TaskSamplerV1, TaskSamplerV2, TaskSamplerV3
from xenoverse.metalang.generator import metalang_generator
from xenoverse.metalang.generator_v3 import metalang_generator_v3
register(
    id='meta-language-v1',
    entry_point='xenoverse.metalang:MetaLangV1',
    kwargs={"L": 2048}
)

register(
    id='meta-language-v2',
    entry_point='xenoverse.metalang:MetaLangV2',
    kwargs={"L": 2048}
)

register(
    id='meta-language-v3',
    entry_point='xenoverse.metalang:MetaLMV3Env',
    kwargs={}
)