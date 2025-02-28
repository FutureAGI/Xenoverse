#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

class MetaLangV1(gym.Env):
    """
    Meta Language Model
    MetaLM(V, n, l, e, L) generates n Exponential(1/l)-length sequences, which appears repeatedly until reaching length L
    Each time it repeats, we add a noise by replacing the number with e probablity 
    V: Vocabulary Size
    l: mean repeating length
    e: noise ratio
    L: Sequence Length

    The task require the agent to recover the correct sequence.
    """

    def __init__(self, L=2048):
        self.L = int(L)
        self.mask_ratio = 0.30
        self.task_set = False
        assert self.L > 1

    def add_noise(self, seq):
        """
        Add noise to a sequence, return the new sequence
        """
        noise_value = random.randint(0, self.n_vocab, size=(numpy.shape(seq)), dtype="int32")
        noise_ratio = (random.random(size=(numpy.shape(seq))) < self.error_ratio).astype("int32")
        mask_ratio = (random.random(size=(numpy.shape(seq))) < self.mask_ratio).astype("int32")
        diff = noise_value - seq
        diff = diff * (noise_ratio!=0).astype("int32")
        new_seq = seq + diff
        new_seq = new_seq * (1 -  mask_ratio * noise_ratio)
        return new_seq

    def set_task(self, task):
        for key, val in task.items():
            self.__dict__[key] = val
        self.task_set = True
        
    def data_generator(self, seed=None):
        if(not self.task_set):
            raise Exception("Please set task before using data generator")
        features = []
        labels = []
        cur_l = 0
        while cur_l < self.L + 1:
            seq = self.patterns[random.randint(0, self.n_patterns)]
            fea = self.add_noise(seq)
            sep = numpy.array([self.SepID], dtype="int32")
            features.append(fea)
            labels.append(seq)
            features.append(sep)
            labels.append(sep)
            cur_l += len(seq) + 1
        features = numpy.concatenate(features, axis=0).astype("int32")
        labels = numpy.concatenate(labels, axis=0).astype("int32")
        return features[:self.L], labels[1:(self.L+1)]

    def batch_generator(self, batch_size, seed=None):
        features = []
        labels = []
        for _ in range(batch_size):
            fea, lab = self.data_generator(seed=seed)
            features.append(fea.tolist())
            labels.append(lab.tolist())
        features = numpy.asarray(features)
        labels = numpy.asarray(labels)
        return features, labels

    def generate_text(self, size, output_stream):
        features, labels = self.batch_generator(size)
        if(isinstance(output_stream, _io.TextIOWrapper)):
            need_close = False
        elif(isinstance(output_stream, str)):
            output_stream = open(output_stream, "w")
            need_close = True
        for i in range(features.shape[0]):
            output_stream.write("\t".join(map(lambda x: "%d,%d"%(x[0],x[1]), zip(features[i].tolist(), labels[i].tolist()))))
            output_stream.write("\n")
        if(need_close):
            output_stream.close()

    def generate_npy(self, size, file_name):
        feas, labs = self.batch_generator(size)
        numpy.save(file, numpy.stack([feas, labs], axis=0))

    @property
    def VocabSize(self):
        return self.n_vocab

    @property
    def SepID(self):
        return 0

    @property
    def MaskID(self):
        return 0

    @property
    def PaddingID(self):
        return 0
