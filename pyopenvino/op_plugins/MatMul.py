# MatMul
import numpy as np
import common_def

def name():
    print('MatMul')


def kernel_MatMul_numpy(inputs, data):
    input0 = inputs[0]
    input1 = inputs[1]
    if data['transpose_a'] == 'true':
        input0 = input0.T
    if data['transpose_b'] == 'true':
        input1 = input1.T
    res = np.matmul(input0, input1)
    return res


def transpose_naive(input:np.array):
    ax0, ax1 = input.shape
    output = np.zeros((ax1, ax0), dtype=input.dtype)
    for i0 in range(ax0):
        for i1 in range(ax1):
            output[i1, i0] = input[i0, i1]
    return output

def kernel_MatMul_naive(inputs, data):
    input0 = inputs[0]
    input1 = inputs[1]
    if data['transpose_a'] == 'true':
        input0 = transpose_naive(input0)
    if data['transpose_b'] == 'true':
        input1 = transpose_naive(input1)
    i00, i01 = input0.shape
    i10, i11 = input1.shape
    output = np.zeros((i00, i11), dtype=input0.dtype)
    for i0 in range(i00):
        for i1 in range(i01):
            for i2 in range(i11):
                output[i0, i2] += input0[i0, i1] * input1[i1, i2]
    return output


def compute(node:dict, inputs:dict=None, debug:bool=False):
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        if data.dtype != common_def.type_convert_tbl[input_port['precision']]:
            print('input data precision mismatch')
            return None
        if data.shape != input_port['dims']:
            print('input data shape mismatch')
            return None
    data = node['data']

    #res = kernel_MatMul_numpy(inputs, data)
    res = kernel_MatMul_naive(inputs, data)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'StatefulPartitionedCall/sequential/dense/MatMul', 'type': 'MatMul', 'version': 'opset1', 
# 'data': {'transpose_a': 'false', 'transpose_b': 'true'}, 
# 'input': {0: {'precision': 'FP32', 'dims': (1, 576)}, 
#           1: {'precision': 'FP32', 'dims': (64, 576)}}, 
# 'output': {2: {'precision': 'FP32', 'dims': (1, 64)}}}
