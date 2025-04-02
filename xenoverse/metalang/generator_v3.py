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
import sys
import argparse
import numpy
import time
import pickle
import random

from xenoverse.metalang.metalangv3 import MetaLMV3Env
from xenoverse.metalang.task_sampler import TaskSamplerV1, TaskSamplerV2, TaskSamplerV3
from xenoverse.metalang.generator import _text_io

def sample_and_check_task(vocab_size=32,
                          embedding_size=16,
                          hidden_size=32,
                          function_token_number=6,
                          seed=None):
    env = MetaLMV3Env()
    reward_high = 0.0
    reward_low = 0.0

    while(reward_high < 2.0 or reward_low > 0.0):
        task = TaskSamplerV3(vocab_size=vocab_size,
                            embedding_size=embedding_size,
                            hidden_size=hidden_size,
                            seed=seed)
        env.set_task(task)
        obs = env.reset()
        next_obs, reward_low, _, _, info = env.step(env.policy(T=10000), 
                                                 cached=True)
        next_obs, reward_high, _, _, info = env.step(env.policy(T=1.0e-6),
                                                  cached=True)

    return task

def generate_data_v3_single_task_qar(task, T_choices=None, L=10000):
    env = MetaLMV3Env()
    vocab = task["function_vocabulary"]

    def reward_token(r):
        if(r < 0):
            return vocab['r1']
        elif(r < 0.5):
            return vocab['r2']
        elif(r < 1.0):
            return vocab['r3']
        elif(r < 2.0):
            return vocab['r4']
        else:
            return vocab['r5']

    list_data = []
    label = []
    if(T_choices is None):
        T_choices = numpy.logspace(-1, 4, num=20)
    
    env.set_task(task)
    obs = env.reset()

    while len(list_data) < L:
        act = env.policy(T=numpy.random.choice(T_choices))
        next_obs, reward, _, _, info = env.step(act)
        r_token = reward_token(reward)
        list_data.append(vocab['q'])
        list_data.extend(obs)
        list_data.append(vocab['s'])
        list_data.append(vocab['a'])
        list_data.extend(act)
        list_data.append(vocab['s'])
        list_data.append(r_token)
        label.extend(obs)
        label.append(vocab['s'])
        label.append(vocab['a'])
        label.extend(info['label'])
        label.append(vocab['s'])
        label.append(r_token)
        label.append(vocab['q'])
        obs = next_obs
    return numpy.array(list_data[:L]), numpy.array(label[:L])

def generate_data_v3_single_task_qara(task, T_choices=None, L=10000):
    env = MetaLMV3Env()
    vocab = task["function_vocabulary"]

    def reward_token(r1, r2):
        deta = r1 - r2
        if(deta > 0.20):
            return vocab['r>']
        elif(deta < 0.5):
            return vocab['r<']
        else:
            return vocab['r=']

    list_data = []
    label = []
    if(T_choices is None):
        T_choices = numpy.logspace(-1, 4, num=20)
    
    env.set_task(task)
    obs = env.reset()

    while len(list_data) < L:
        act1 = env.policy(T=numpy.random.choice(T_choices))
        _, r1, _, _, info1 = env.step(act1, cached=True)
        act2 = env.policy(T=numpy.random.choice(T_choices))
        next_obs2, r2, _, _, info2 = env.step(act2)
        r_token = reward_token(r1, r2)

        list_data.append(vocab['q'])
        list_data.extend(obs)
        list_data.append(vocab['s'])
        list_data.append(vocab['a'])
        list_data.extend(act1)
        list_data.append(vocab['s'])
        list_data.append(r_token)
        list_data.append(vocab['a'])
        list_data.extend(act2)
        list_data.append(vocab['s'])
        label.extend(obs)
        label.append(vocab['s'])
        label.append(vocab['a'])
        label.extend(info1['label'])
        label.append(vocab['s'])
        label.append(r_token)
        label.append(vocab['a'])
        label.extend(info2['label'])
        label.append(vocab['s'])
        label.append(vocab['q'])
        obs = next_obs2
    return numpy.array(list_data[:L]), numpy.array(label[:L])

def generate_data_v3_single_task_qa(task, T_choices=None, L=10000):
    env = MetaLMV3Env()
    vocab = task["function_vocabulary"]

    list_data = []
    label = []
    if(T_choices is None):
        T_choices = numpy.logspace(-1, 4, num=20)
    
    env.set_task(task)
    obs = env.reset()

    while len(list_data) < L:
        act = env.policy(T=1.0e-3)
        next_obs, reward, _, _, info = env.step(act)
        list_data.append(vocab['q'])
        list_data.extend(obs)
        list_data.append(vocab['s'])
        list_data.append(vocab['a'])
        list_data.extend(act)
        list_data.append(vocab['s'])
        label.extend(obs)
        label.append(vocab['s'])
        label.append(vocab['a'])
        label.extend(act)
        label.append(vocab['s'])
        label.append(vocab['q'])

        obs = next_obs
    return numpy.array(list_data[:L]), numpy.array(label[:L])

def metalang_generator_v3(datatype='QAR',   #choices ['QAR', 'QA', 'QARA']
                       sample_type='sequences',
                       vocab_size=32,
                       embedding_size=16,
                       hidden_size=32,
                       samples=1000,
                       sequence_length=16000,
                       output_type='txt',
                       task_file=None,
                       output=None):
    seed_base = int(time.time()*1000 % 1000000)

    assert datatype in ['QAR', 'QA', 'QARA'], 'datatype must be one of QAR, QA, QARA'
    if(datatype == 'QAR'):
        gen_func = generate_data_v3_single_task_qar
    elif(datatype == 'QA'):
        gen_func = generate_data_v3_single_task_qa
    elif(datatype == 'QARA'):
        gen_func = generate_data_v3_single_task_qara

    if(sample_type == 'tasks'):
        if(output is None):
            raise Exception("Must specify --output when sample_type is tasks")
        
        tasks = [sample_and_check_task(vocab_size=vocab_size,
                                     embedding_size=embedding_size,
                                     hidden_size=hidden_size,
                                     function_token_number=6,
                                     seed=seed_base) for _ in range(samples)]

        output_file_name = output
        if(not output_file_name.endswith('.pkl')):
            output_file_name += '.pkl'
        pickle.dump(tasks, open(output_file_name, 'wb'))
    else:
        env = MetaLMV3Env()
        if(task_file is None):
            tasks = None
        else:
            tasks = pickle.load(open(task_file, 'rb'))
        if(tasks is None):
            # Generate a unique task for each sample
            tasks = [sample_and_check_task(vocab_size=vocab_size,
                                          embedding_size=embedding_size,
                                          hidden_size=hidden_size,
                                          function_token_number=6) for _ in range(samples)]
        random.shuffle(tasks)
        data = []
        for t in tasks:
            x, y = gen_func(t, L=sequence_length)
            data.append([x,y])
        data = numpy.array(data)
        print(data.shape)
        if(output_type == 'npy'):
            numpy.save(output, data)
        elif(output_type == 'txt'):
            _text_io(data, output)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generating Meta Language Tasks or Sequences')
    parser.add_argument('--sample_type', type=str, choices=['tasks', 'sequences'], default='sequences', help='Generate tasks or sequences')
    parser.add_argument('--datatype', type=str, choices=['QAR', 'QA', 'QARA'], default='QAR', help='Query-Answer, Query-Answer-Reward, or Query-Answer-Reward-Answer')
    parser.add_argument('--task_file', type=str, default=None, help='Specify task file to generate from if the sample_type is sequences. Default will generate task on the fly.')
    parser.add_argument('--vocab_size', type=int, default=32)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--sequence_length', type=int, default=16000)
    parser.add_argument('--samples', type=int, default=10, help='number of sequences / tasks to generate')
    parser.add_argument('--output_type', type=str, choices=['txt', 'npy'], default='txt')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()
    metalang_generator_v3(**vars(args))