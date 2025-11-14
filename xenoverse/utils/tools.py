import numpy
from numba import njit
from numpy import random
import secrets
import string

@njit(cache=True)
def conv2d_numpy(input_data:numpy.ndarray, 
                 kernel:numpy.ndarray, 
                 stride=(1,1), padding=0):
    input_height, input_width = input_data.shape
    kernel_height, kernel_width = kernel.shape
    output_height = (input_height - kernel_height + 2 * padding) // stride[0] + 1
    output_width = (input_width - kernel_width + 2 * padding) // stride[1] + 1
    
    output_data = numpy.zeros((output_height, output_width))
    ni = 0
    for i in range(-padding, input_height - kernel_height + padding + 1, stride[0]):
        ib = max(0, i)
        ie = min(input_height, i + kernel_height)
        _ib = ib - i
        _ie = ie - i
        nj = 0
        for j in range(-padding, input_width - kernel_width + padding + 1, stride[1]):
            jb = max(0, j)
            je = min(input_width, j + kernel_width)
            _jb = jb - j
            _je = je - j
            output_data[ni, nj] = numpy.sum(input_data[ib:ie, jb:je] * kernel[_ib:_ie, _jb:_je])
            nj += 1
        ni += 1
    
    return output_data

def random_partition(num_parts:int):
    # Generate a random partition of 1 into num_parts parts
    if num_parts <= 0:
        raise ValueError("Number of parts must be greater than 0")
    if num_parts == 1:
        return [sum_value]
    partitions = numpy.random.random(num_parts - 1)
    partitions.sort()
    partitions = numpy.concatenate(([0], partitions, [1]))
    return partitions[1:] - partitions[:-1]

def versatile_sample(setting, default_range, default_value):
    if(isinstance(setting, tuple) or isinstance(setting, list)):
        assert len(setting) == 2, f"Setting must be a tuple or list of length 2, got {len(setting)}"
        return random.uniform(setting[0], setting[1])
    elif(setting):
        return random.uniform(default_range[0], default_range[1])
    else:
        return default_value
    
def generate_secure_strings(count, length=16):
    alphabet = string.ascii_letters + string.digits  # 62个字符
    return [''.join(secrets.choice(alphabet) for _ in range(length)) 
            for _ in range(count)]