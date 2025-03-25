### A random MLP generator

import time
import numpy
import re
from numpy import random

def pseudo_random_seed(hyperseed=0):
    '''
    Generate a pseudo random seed based on current time and system random number
    '''
    timestamp = time.time_ns()
    system_random = int(random.random() * 100000000)
    pseudo_random = timestamp + system_random + hyperseed
    
    return pseudo_random % (4294967296)
    
def gen_uniform_matrix(n_in, n_out):
    w = random.normal(size=[n_out, n_in])
    u, s, vt = numpy.linalg.svd(w)
    s = numpy.diag(numpy.ones_like(s) * random.uniform(low=0.5, high=3))

    sm = numpy.zeros((n_out, n_in))
    numpy.fill_diagonal(sm, s)
    return u @ sm @ vt

def weights_and_biases(n_in, n_out, need_bias=False):
    weights = gen_uniform_matrix(n_in, n_out)
    if(need_bias):
        bias = 0.1 * random.normal(size=[n_out])
    else:
        bias = numpy.zeros(shape=[n_out])
    return weights, bias

def actfunc(raw_name):
    if(raw_name is None):
        name = 'none'
    else:
        name = raw_name.lower()
    if(name=='sigmoid'):
        return lambda x: 1/(1+numpy.exp(-x))
    elif(name=='tanh'):
        return numpy.tanh
    elif(name.find('leakyrelu') >= 0):
        return lambda x: numpy.maximum(0.01*x, x)
    elif(name.find('bounded') >= 0):
        pattern = r"bounded\(([-+]?\d*\.\d+|[-+]?\d+),\s*([-+]?\d*\.\d+|[-+]?\d+)\)"
        match = re.match(pattern, name)
        if match:
            B = float(match.group(1).strip())
            T = float(match.group(2).strip())
        else:
            raise ValueError("Bounded support only BOUNDED(min,max) type")
        k = (T - B) / 2
        return lambda x: k*numpy.tanh(x/k) + k + B
    elif(name == 'sin'):
        return lambda x: numpy.concat([numpy.sin(x[:len(x)//2]), numpy.cos(x[len(x)//2:])], axis=-1)
    elif(name == 'none'):
        return lambda x:x
    else:
        raise ValueError(f"Invalid activation function name: {name}")

class RandomMLP(object):
    '''
    A class for generating random MLPs with given parameters
    '''
    def __init__(self, n_inputs, n_outputs, 
                 n_hidden_layers=None, 
                 activation=None, 
                 biases=False,
                 seed=None):
        # Set the seed for the random number generator
        if seed is None:
            seed = pseudo_random_seed()
        random.seed(seed)

        # Set the number of hidden units and activation function
        self.hidden_units = [n_inputs]
        if n_hidden_layers is not None:
            if(isinstance(n_hidden_layers, list)):
                self.hidden_units += n_hidden_layers
            elif(isinstance(n_hidden_layers, numpy.ndarray)):
                self.hidden_units += n_hidden_layers.tolist()
            elif(isinstance(n_hidden_layers, tuple)):
                self.hidden_units += list(n_hidden_layers)
            elif(isinstance(n_hidden_layers, int)):
                self.hidden_units.append(n_hidden_layers)
            else:
                raise TypeError(f"Invalid input type of n_hidden_layers: {type(n_hidden_layers)}")
        self.hidden_units.append(n_outputs)
        
        self.activation = []

        if activation is None:
            for _ in range(len(self.hidden_units)-1):
                self.activation.append(actfunc(None))
        elif isinstance(activation, list):
            assert len(activation) == len(self.hidden_units) - 1
            for hidden_act in activation:
                self.activation.append(actfunc(hidden_act))
        elif isinstance(activation, str):
            for _ in range(len(self.hidden_units)-1):
                self.activation.append(actfunc(activation))
        
        # Initialize weights and biases to random values
        self.weights = []
        self.biases = []
        for i in range(len(self.hidden_units)-1):
            if(isinstance(biases, list)):
                assert len(biases) == len(self.hidden_units) - 1
                w, b = weights_and_biases(self.hidden_units[i], self.hidden_units[i+1], need_bias=biases[i])
            else:
                w, b = weights_and_biases(self.hidden_units[i], self.hidden_units[i+1], need_bias=biases)
            self.weights.append(w)
            self.biases.append(b)
            
    def forward(self, inputs):
        outputs = inputs
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            outputs = self.activation[i](weight @ outputs + bias)
        if(numpy.size(outputs) > 1):
            return outputs
        else:
            return outputs[0]
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

class RandomRNN(object):
    '''
    A class for generating random MLPs with given parameters
    '''
    def __init__(self, n_inputs, n_hidden,
                 activation='tanh',
                 seed=None):
        # Set the seed for the random number generator
        if seed is None:
            seed = pseudo_random_seed()
        random.seed(seed)
        
        # Initialize weights and biases to random values
        n_inner = n_inputs + n_hidden
        self.wh, self.bh = weights_and_biases(n_inner, n_hidden, need_bias=True)
        self.hidden_states = numpy.zeros(shape=[n_hidden])

        self.reset()
        self.act_func = actfunc(activation)

    def reset(self):
        self.hidden_states.fill(0.)

    def restore(self):
        self.hidden_states = self.cache_states.copy()

    def cache(self):
        self.cache_states = self.hidden_states.copy()

    def forward(self, inputs):
        outputs = inputs
        n_hidden = numpy.concat([self.hidden_states, inputs], axis=0)
        self.hidden_states = self.act_func(self.wh @ n_hidden + self.bh)
        return self.hidden_states.copy()

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
def layer_norm(x, eps=1e-8):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    normed = (x - mean) / (std + eps)
    return normed

def softmax_sampling(logits, temperature=1.0):
    logits -= numpy.max(logits)
    probs = numpy.exp(logits / temperature)
    probs /= numpy.sum(probs, axis=-1, keepdims=True)
    symbol = numpy.random.choice(numpy.arange(len(probs)), p=probs)
    return int(symbol), - numpy.log(max(probs[symbol], 1.0e-10))

def rnd_sampling(logits, temperature=1.0):
    logits[1:] -= logits[1:] * (logits[1:] < -1.0e+5).astype(numpy.float32)
    probs = numpy.exp(logits)
    probs /= numpy.sum(probs, axis=-1, keepdims=True)
    symbol = numpy.random.choice(numpy.arange(len(probs)), p=probs)

    return int(symbol), - numpy.log(max(probs[symbol], 1.0e-10))

def high_sampling(logits, temperature=1.0):
    logits -= numpy.max(logits)
    probs = numpy.exp(logits / temperature)
    probs /= numpy.sum(probs, axis=-1, keepdims=True)
    symbol = numpy.random.choice(numpy.arange(len(probs)), p=probs)
    if(symbol != 0):
        symbol = numpy.argmax(probs)

    return int(symbol), - numpy.log(max(probs[symbol], 1.0e-10))
    
class RandomLM(object):
    '''
    A class for generating random GRUs with given parameters
    '''
    def __init__(self, n_vocab, n_emb, n_hidden, 
                 function_token_number=10,
                 seed=None):
        # Set the seed for the random number generator
        self.n_vocab = n_vocab
        self.function_token_number = function_token_number
        self.enc = RandomMLP(n_vocab, n_emb, seed=seed)
        self.dec = RandomMLP(n_hidden, n_vocab, seed=seed)
        self.rnn = RandomRNN(n_emb, n_hidden, seed=seed)
        self.stop_inc = 0.02
        self.echo_punish = 0.05

    def reset(self):
        self.rnn.reset()
        self.stop_bias = -1.0e+6 # Stop first token to be 0
        self.echo_bias = numpy.zeros((self.n_vocab,))
    
    def cache(self):
        self.rnn.cache()
        self.stop_bias = -1.0e+6 # Stop first token to be 0
        self.echo_bias.fill(0.0)

    def restore(self):
        self.rnn.restore()
        self.stop_bias = -1.0e+6 # Stop first token to be 0
        self.echo_bias.fill(0.0)

    def forward(self, inputs):
        emb = numpy.zeros(shape=[self.n_vocab])
        emb[inputs] = 1

        encodings = layer_norm(self.enc(emb))
        hiddens = self.rnn(encodings)
        decodings = self.dec(hiddens)

        logits = decodings + self.echo_bias
        logits[0] += self.stop_bias
        logits[1:self.function_token_number] = -1.0e+6
        if(self.stop_bias < 0):
            self.stop_bias = self.stop_inc  # avoid stop from the beginning
        else:
            self.stop_bias += self.stop_inc # increase probability of stop token
        self.echo_bias[inputs] -= self.echo_punish

        return logits

    def generate_one_step(self, inputs, temperature=1, decode_type='softmax'):
        logits = self.forward(inputs)
        if(decode_type == 'softmax'):
            tok = softmax_sampling(logits, temperature=temperature)
        elif(decode_type == 'rnd'):
            tok = rnd_sampling(logits, temperature=temperature)
        elif(decode_type == 'greedy'):
            tok = high_sampling(logits, temperature=temperature)
        else:
            raise NotImplementedError(f"Unknown sampling method: {decode_type}")
        
        return tok
        
    def generate_sequence(self, inputs, T_s=1.0, T_c=1.0, decode_type='softmax'):
        output = []
        ppls = []
        done = False
        T = T_s
        step = 0
        while not done:
            next_token, ppl = self.generate_one_step(inputs, temperature=T, decode_type=decode_type)
            ppls.append(ppl)
            if(next_token == 0):
                done=True
            else:
                output.append(next_token)
                inputs = next_token
            T = T_c
            step += 1
        return output, ppls
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def generate_query(self):
        self.reset()
        query, ppls = self.generate_sequence(0, decode_type='softmax', T_s=10.0, T_c=1.0)
        return query
    
    def generate_answer_greedy(self):
        self.cache()
        ans, ppls = self.generate_sequence(0, decode_type='greedy')
        self.restore()
        return ans, numpy.mean(ppls)
    
    def generate_answer_softmax(self, T=1.0):
        self.cache()
        ans, ppls = self.generate_sequence(0, decode_type='softmax', T_s=T, T_c=T)
        self.restore()
        return ans, numpy.mean(ppls)
    
    def generate_answer_low(self):
        self.cache()
        ans, ppls = self.generate_sequence(0, decode_type='rnd')
        self.restore()
        return ans, numpy.mean(ppls)

    def label_answer(self, ans):
        self.cache()
        ppls = []
        label_toks = []
        prev_token = 0

        for i,tok in enumerate(ans+[0]):  # consider also the stop token
            logits = self.forward(prev_token)
            probs = numpy.exp(logits)
            probs /= numpy.sum(probs)
            label_toks.append(int(numpy.argmax(probs)))
            ppl = - numpy.log(max(1.0e-10, probs[tok]))
            ppls.append(ppl)
            prev_token = tok
        self.restore()
        return label_toks, numpy.mean(ppls)

class RandomFourier(object):
    def __init__(self,
                 ndim,
                 max_order=16,
                 max_item=5,
                 max_steps=1000,
                 box_size=2):
        n_items = random.randint(1, max_item + 1)
        self.coeffs = [(0, random.normal(size=(ndim, 2)) * random.exponential(scale=box_size / numpy.sqrt(n_items), size=(ndim, 2)))]
        self.max_steps = max_steps
        for j in range(n_items):
            # Sample a cos nx + b cos ny
            order = random.randint(1, max_order + 1) + random.normal(scale=1.0)
            factor = random.normal(size=(ndim, 2)) * random.exponential(scale=box_size / numpy.sqrt(n_items), size=(ndim, 2))
            self.coeffs.append((order, factor))

    def __call__(self, t):
        # calculate a cos nx + b cos ny with elements of [t, [a, b]]
        x = t / self.max_steps
        y = 0
        for order, coeff in self.coeffs:
            y += coeff[:,0] * numpy.sin(order * x) + coeff[:,1] * numpy.cos(order * x)
        return y

class RandomGoal(object):
    def __init__(self,
                 ndim,
                 type='static',
                 reward_type='p',
                 repetitive_position=None,
                 repetitive_distance=0.2,
                 is_pitfall=False,
                 max_try=10000,
                 box_size=2):
        # Type: static, fourier
        # Reward type: field (f), trigger (t), potential (p) or combination (e.g., `ft`, `pt`)
        # Pitfall: if True, the goal is a pitfall, otherwise it is a goal
        eff_factor = numpy.sqrt(ndim)
        eff_rd = repetitive_distance * eff_factor
        self.reward_type = reward_type
        self.is_pitfall = is_pitfall
        if(type == 'static'):
            overlapped = True
            ntry = 0
            while overlapped and ntry < max_try:
                position = random.uniform(low=-box_size, high=box_size, size=(ndim, ))

                overlapped = False
                
                if(repetitive_position is None):
                    break

                for pos in repetitive_position:
                    dist = numpy.linalg.norm(pos - position)
                    if(dist < eff_rd):
                        overlapped = True
                        break
                ntry += 1
            if(ntry >= max_try):
                raise RuntimeError(f"Failed to generate goal position after {max_try} tries.")
            self.position = lambda t:position
        elif(type == 'fourier'):
            self.position = RandomFourier(ndim)
        else:
            raise ValueError(f"Invalid goal type: {type}")
        self.activate()

        self.has_field_reward=False
        self.has_trigger_reward=False
        self.has_potential_reward=False

        if('f' in self.reward_type): # Field Rewards
            self.field_reward = random.uniform(0.2, 0.8)
            self.field_threshold = random.exponential(box_size / 2) * eff_factor
            self.has_field_reward = True
        if('t' in self.reward_type): # Trigger Rewards
            self.trigger_reward = max(random.exponential(5.0), 1.0)
            self.trigger_threshold = random.uniform(0.20, 0.50) * eff_factor
            if(is_pitfall):
                self.trigger_threshold += box_size / 4
            self.trigger_rs_terminal = self.trigger_reward
            self.trigger_rs_threshold = 3 * box_size * eff_factor
            self.trigger_rs_potential = self.trigger_reward * self.trigger_rs_threshold / box_size
            self.has_trigger_reward = True
        if('p' in self.reward_type): # Potential Rewards
            self.potential_reward = max(random.exponential(2.0), 0.5)
            self.potential_threshold = random.uniform(box_size/2, box_size) * eff_factor
            self.has_potential_reward = True

    def activate(self):
        self.is_activated = True

    def deactivate(self):
        self.is_activated = False

    def __call__(self, sp, sn, t=0, need_reward_shaping=False):
        # input previous state, next state        
        # output reward, done
        if(not self.is_activated):
            return 0.0, False, {}
        reward = 0
        shaped_reward = 0
        done = False
        cur_pos = self.position(t)
        dist = numpy.linalg.norm(sn - cur_pos)
        distp = numpy.linalg.norm(sp - cur_pos)
        if(self.has_field_reward):
            if(dist <= 3.0 * self.field_threshold):
                k = dist / self.field_threshold
                reward += self.field_reward * numpy.exp(- k ** 2)
        if(self.has_trigger_reward):
            if(dist <= self.trigger_threshold):
                reward += self.trigger_reward
                if(need_reward_shaping):
                    shaped_reward += self.trigger_rs_terminal - self.trigger_reward
                done = True
            if(need_reward_shaping):
                if(dist <= self.trigger_rs_threshold):
                    shaped_reward += self.trigger_rs_potential * (min(distp, self.trigger_rs_threshold) - dist) / self.trigger_rs_threshold
            #print(f"dist: {dist}, distp: {distp}, reward: {shaped_reward}, \
            #      trigger_rs_threshold: {self.trigger_rs_threshold}")
        if(self.has_potential_reward):
            if(dist <= self.potential_threshold):
                reward += self.potential_reward * (min(distp, self.potential_threshold) - dist) / self.potential_threshold
        shaped_reward += reward
        if(self.is_pitfall):
            reward *= -1
            shaped_reward = 0
        return reward, done, {'shaped_reward':shaped_reward}