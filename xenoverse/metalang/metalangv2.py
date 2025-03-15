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
import numpy
import gym
from numpy import random


class RandomNGram(object):
    def __init__(self, task):
        for key, val in task.items():
            self.__dict__[key] = val
        self.s_tok = 0
        self.w_arr = numpy.expand_dims(numpy.arange(self.n_gram), axis=[0, 2, 3])

    def softmax(self, x):
        e_x = numpy.exp(x - numpy.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def forward(self, l, batch=1, seed=None):
        #Generate n=batch sequences of length l under current task
        ind = 0
        if(seed is not None):
            numpy.random.seed(seed)

        def mean_var_norm(i):
            m_i = numpy.mean(i)
            m_ii = numpy.mean(i * i)
            std = numpy.sqrt(m_ii - m_i * m_i)
            return (1.0 / std) * (i - m_i)

        cur_tok = numpy.full((batch,), self.s_tok)
        idxes = numpy.arange(batch)
        pad_emb = numpy.expand_dims(self.word_emb[cur_tok], axis=1)
 
        h = numpy.zeros((batch, self.n_hidden))

        # mark whether there is end token
        seqs = []
        seqs.append(cur_tok)
        ppl = 0
        tok_cnt = 0
        tok_embs = [pad_emb for _ in range(self.n_gram)]
        while ind < l:
            ind += 1
            tok_emb = numpy.expand_dims(self.word_emb[cur_tok], axis=1)
            tok_embs.append(tok_emb)
            del tok_embs[0]
            tok_emb = numpy.expand_dims(numpy.concatenate(tok_embs[-self.n_gram:], axis=1), axis=2)

            h = numpy.tanh(numpy.matmul(tok_emb, self.weights_inputlayer) + self.bias_inputlayer)
            h = numpy.mean(self.w_arr * h, axis=1)
            o = numpy.matmul(h, self.weights_outputlayer) + self.bias_outputlayer
            o = numpy.squeeze(o, axis=1)
            o = self._lambda * mean_var_norm(o)
            exp_o = numpy.exp(o)
            prob = exp_o / numpy.sum(exp_o, axis=-1, keepdims=True)
            cur_tok = (prob.cumsum(1) > numpy.random.rand(prob.shape[0])[:,None]).argmax(1)
            cur_prob = prob[idxes, cur_tok]
            ppl -= numpy.sum(numpy.log(cur_prob))
            tok_cnt += cur_prob.shape[0]

            seqs.append(cur_tok)
        print("Ground Truth Sequence Perplexity: %f" % (ppl / tok_cnt))

        return numpy.transpose(numpy.asarray(seqs, dtype="int32"))

class MetaLangV2(gym.Env):
    """
    Pseudo Langauge Generated from RNN models
    V: vocabulary size
    d: embedding size (input size)
    n: n-gram
    N: hidden size
    e: inverse of softmax - temporature
    L: maximum length
    """
    def __init__(self, L=4096):
        self.L = int(L)
        assert L>1
        self.task_set = False

    def set_task(self, task):
        self.nn = RandomNGram(task)
        self.task_set = True

    def data_generator(self, seed=None):
        if(self.task_set):
            tokens = self.nn.forward(self.L, seed=seed)[0]
        else:
            raise Exception("Please set task before using data generator")
        return tokens

    def batch_generator(self, batch_size, seed=None):
        if(self.task_set):
            tokens = self.nn.forward(self.L, batch=batch_size, seed=seed)
        else:
            raise Exception("Please set task before using data generator")
        return tokens