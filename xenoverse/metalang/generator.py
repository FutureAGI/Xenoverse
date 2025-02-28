#   Copyright (c) 2021 DeepEvolution Authors. All Rights Reserved.
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

# This file is used to generate data for meta language models

import sys
import argparse
import random
import time
import _io
import numpy
import pickle
from xenoverse.metalang import MetaLangV1
from xenoverse.metalang import MetaLangV2
from xenoverse.metalang import TaskSamplerV1, TaskSamplerV2

def _text_io(tokens, output_stream):
    if(isinstance(output_stream, _io.TextIOWrapper)):
        need_close = False
    elif(isinstance(output_stream, str)):
        output_stream = open(output_stream, "w")
        need_close = True
    else:
        output_stream = sys.stdout
        need_close = False
    for i in range(tokens.shape[0]):
        output_stream.write("\t".join(map(str, tokens[i].tolist())))
        output_stream.write("\n")
    if(need_close):
        output_stream.close()

def metalang_generator(sample_type='sequences',
                       version='v1',
                       vocab_size=64,
                       patterns_number=10,
                       n_gram=3,
                       error_rate=0.15,
                       embedding_size=16,
                       hidden_size=64,
                       samples=1000,
                       lambda_weight=5.0,
                       batch_size=1,
                       task_file=None,
                       sequence_length=4096,
                       output_type='txt',
                       output=None
                       ):
    seed_base = int(time.time()*1000 % 1000000)

    if(sample_type == 'tasks'):
        if(output is None):
            raise Exception("Must specify --output when sample_type is tasks")
        if(version=='v1'):
            tasks = [TaskSamplerV1(seed=i+seed_base, 
                                   n_vocab=vocab_size, 
                                   n_patterns=patterns_number, 
                                   n_gram=n_gram, 
                                   error_ratio=error_rate) for i in range(samples)]
        elif(version=='v2'):
            tasks = [TaskSamplerV2(seed=i+seed_base,
                                   n_vocab=vocab_size,
                                   n_emb=embedding_size,
                                   n_hidden=hidden_size,
                                   n_gram=n_gram,
                                   _lambda=lambda_weight
                                   ) for i in range(samples)]
        output_file_name = output
        if(not output_file_name.endswith('.pkl')):
            output_file_name += '.pkl'
        pickle.dump(tasks, open(output_file_name, 'wb'))
    else:
        if(version == 'v1'):
            env = MetaLangV1(L=sequence_length)
        elif(version == 'v2'):
            env = MetaLangV2(L=sequence_length)

        if(task_file is None):
            tasks = None
        else:
            tasks = pickle.load(open(task_file, 'rb'))

        batch_size = batch_size
        if(tasks is None):
            # Generate a unique task for each sample
            batch_size = 1
            if(version=='v1'):
                tasks = [TaskSamplerV1(seed=i+seed_base, 
                                       n_vocab=vocab_size, 
                                       n_patterns=patterns_number, 
                                       n_gram=n_gram, 
                                       error_ratio=error_rate) for i in range(samples)]
            elif(version=='v2'):
                tasks = [TaskSamplerV2(seed=i+seed_base,
                                       n_vocab=vocab_size,
                                       n_emb=embedding_size,
                                       n_hidden=hidden_size,
                                       n_gram=n_gram,
                                       _lambda=lambda_weight
                                       ) for i in range(samples)]
        batch_number = (samples - 1) // batch_size + 1
        tokens = []
        if(len(tasks) < batch_number):
            tasks = tasks * ((batch_number - 1)//len(tasks) + 1)
        random.shuffle(tasks)
        for i in range(batch_number):
            env.set_task(tasks[i])
            seed_base = int(time.time()*1000 % 1000000)
            token = env.batch_generator(batch_size, seed=i+seed_base)
            tokens.append(token)
        tokens=numpy.concatenate(tokens, axis=0)

        if(output_type == 'npy'):
            numpy.save(output, tokens)
        elif(output_type == 'txt'):
            _text_io(tokens, output)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generating Meta Language Tasks or Sequences')
    parser.add_argument('--version', type=str, choices=['v1', 'v2'], default='v2')
    parser.add_argument('--sample_type', type=str, choices=['tasks', 'sequences'], default='sequences', help='Generate tasks or sequences')
    parser.add_argument('--task_file', type=str, default=None, help='Specify task file to generate from if the sample_type is sequences. Default will generate task on the fly.')
    parser.add_argument('--vocab_size', type=int, default=64)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--patterns_number', type=int, default=10)
    parser.add_argument('--error_rate', type=float, default=0.20)
    parser.add_argument('--n_gram', nargs='+', type=int, default=[3,4,5,6], help="A [List of] length n used for generating tasks")
    parser.add_argument('--lambda_weight', type=float, default=5.0, help="Lambda weight multiplied for softmax sampling in MetaLangV2")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sequence_length', type=int, default=4096)
    parser.add_argument('--samples', type=int, default=100, help='number of sequences / tasks to generate')
    parser.add_argument('--output_type', type=str, choices=['txt', 'npy'], default='txt')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()
    metalang_generator(**vars(args))
