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

import sys
import _io
import numpy
import gym
from numpy import random
from xenoverse.utils import pseudo_random_seed
from xenoverse.utils import RandomLM

def TaskSamplerV2(seed=None,
                n_emb=16,
                n_hidden=64,
                n_vocab=256,
                n_gram=3,
                _lambda=5.0):
    if(seed is not None):
        numpy.random.seed(seed)
    else:
        numpy.random.seed(pseudo_random_seed())
    if(isinstance(n_gram, list)):
        n_gram = random.choice(n_gram)
    word_emb = numpy.random.normal(0, 1.0, size=(n_vocab, n_emb))
    weights_inputlayer = numpy.random.normal(0, 1.0, size=(n_gram, n_emb, n_hidden))
    bias_inputlayer = numpy.random.normal(0, 1.0, size=(n_gram, 1, n_hidden))
    weights_outputlayer = numpy.random.normal(0, 1.0, size=(n_hidden, n_vocab))
    bias_outputlayer = numpy.random.normal(0, 1.0, size=(1, n_vocab))
    return {
                'word_emb': word_emb,
                'weights_inputlayer': weights_inputlayer,
                'bias_inputlayer': bias_inputlayer,
                'weights_outputlayer': weights_outputlayer,
                'bias_outputlayer': bias_outputlayer,
                '_lambda': _lambda,
                'n_emb': n_emb,
                'n_hidden': n_hidden,
                'n_vocab': n_vocab,
                'n_gram': n_gram
            }

def TaskSamplerV1(seed=None,
                n_vocab=64, 
                n_patterns=10, 
                n_gram=64,
                error_ratio=0.1):
    patterns = []
    if(seed is not None):
        numpy.random.seed(seed)
    else:
        numpy.random.seed(pseudo_random_seed())
    if(isinstance(n_gram, list)):
        n_gram = random.choice(n_gram)
    for _ in range(n_patterns):
        l_r = max(3, numpy.random.poisson(n_gram))
        patterns.append(random.randint(0, n_vocab, size=(l_r), dtype="int32"))
    return {
            'patterns': patterns,
            'n_vocab': n_vocab,
            'n_patterns': n_patterns,
            'error_ratio': error_ratio,
            'n_gram': n_gram
        }

function_vocabulary = {'s':0,
    'q':1,
    'a':2,
    'r1':3,
    'r2':4,
    'r3':5,
    'r4':6,
    'r5':7,
    'r>':8,
    'r=':9,
    'r<':10}

def TaskSamplerV3(vocab_size=32,
                 embedding_size=16,
                 hidden_size=32,
                 seed=None):
    return {"vocabulary": vocab_size,
            "embedding": embedding_size,
            "hidden": hidden_size,
            "function_vocabulary": function_vocabulary,
            "lm": RandomLM(n_vocab=vocab_size, 
                           function_vocab=function_vocabulary,
                           n_emb=embedding_size, 
                           n_hidden=hidden_size, 
                           seed=seed)}