# MaxPool
import math
import numpy as np
import common_def

def name():
    print('LRN')


def kernel_LRN_naive(inputs:dict, alpha:float, beta:float, bias:float, size:float):
    input0 = inputs[0]
    n,c,h,w = input0.shape

    res = np.zeros_like(input0)

    input_sq = input0 ** 2
    for bn in range(n):
        for ch in range(c):
            for y in range(h):
                for x in range(w):
                    denominator = (bias+alpha*np.sum(input_sq[bn, max(0,ch-size//2):min(c, ch+size//2+1), y, x])) ** beta
                    res[bn, ch, y, x] = input0[bn, ch, y, x] / denominator
    return res


def kernel_LRN_numpy(inputs:dict, alpha:float, beta:float, bias:float, size:float):
    input0 = inputs[0]
    n,c,h,w = input0.shape

    res = np.zeros_like(input0)

    input_sq = input0 ** 2
    denominator = np.zeros_like(input0)
    for ch in range(c):
        denominator[:,ch,:,:] = (bias+alpha*np.sum(input_sq[:, max(0,ch-size//2):min(c, ch+size//2+1), :, :], axis=1)) ** beta

    res = input0 / denominator
    return res


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        assert data.dtype == common_def.type_convert_tbl[input_port['precision']]
        assert data.shape == input_port['dims']

    alpha = float(node['data']['alpha'])
    beta  = float(node['data']['beta'])
    bias  = float(node['data']['bias'])
    size  = int(node['data']['size'])

    if kernel_type == 'naive':
        res = kernel_LRN_naive(inputs, alpha, beta, bias, size)
    else:
        res = kernel_LRN_numpy(inputs, alpha, beta, bias, size)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# LRN {'name': 'pool1/norm16325', 'type': 'LRN', 'version': 'opset1', 
# 'data': {'alpha': '9.9999997473787516e-05', 'beta': '0.75', 'bias': '1', 'size': '5'}, 
# 'input': {0: {'precision': 'FP16', 'dims': (1, 64, 56, 56)}, 1: {'precision': 'I64', 'dims': (1,)}}, 
# 'output': {2: {'precision': 'FP16', 'dims': (1, 64, 56, 56)}}}
