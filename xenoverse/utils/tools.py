import numpy as np
from numba import njit

@njit(cache=True)
def conv2d_numpy(input_data:np.ndarray, 
                 kernel:np.ndarray, 
                 stride=(1,1), padding=0):
    input_height, input_width = input_data.shape
    kernel_height, kernel_width = kernel.shape
    output_height = (input_height - kernel_height + 2 * padding) // stride[0] + 1
    output_width = (input_width - kernel_width + 2 * padding) // stride[1] + 1
    
    output_data = np.zeros((output_height, output_width))
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
            output_data[ni, nj] = np.sum(input_data[ib:ie, jb:je] * kernel[_ib:_ie, _jb:_je])
            nj += 1
        ni += 1
    
    return output_data