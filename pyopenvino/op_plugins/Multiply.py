# Multiply
import numpy as np
import common_def

def name():
    print('Multiply')


def kernel_Multiply_numpy(inputs:dict):
    input0 = inputs[0]
    input1 = inputs[1]
    if input0.size > input1.size:
        input1 = np.broadcast_to(input1, input0.shape)
    else:
        input0 = np.broadcast_to(input0, input1.shape)
    res = input0 * input1
    return res


def kernel_Multiply_naive(inputs:dict):
    input0 = inputs[0]
    input1 = inputs[1]
    if input0.size > input1.size:
        input1 = np.broadcast_to(input1, input0.shape)
    else:
        input0 = np.broadcast_to(input0, input1.shape)

    assert input0.ndim <= 4
    shape_orig = input0.shape
    if input0.ndim < 4:         # Convert tensors to rank-4 tensors
        input0 = np.expand_dims(input0, tuple(range(4-input0.ndim)))
        input1 = np.expand_dims(input0, tuple(range(4-input0.ndim)))

    output = np.zeros_like(input0)
    shape = input0.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    output[i,j,k,l] = input0[i,j,k,l] * input1[i,j,k,l]

    return output.reshape(shape_orig)


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        assert data.dtype == common_def.type_convert_tbl[input_port['precision']]
        assert data.shape == input_port['dims']

    if kernel_type == 'naive':
        res = kernel_Multiply_naive(inputs)
    else:
        res = kernel_Multiply_numpy(inputs)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'StatefulPartitionedCall/sequential/batch_normalization/FusedBatchNormV3/mean/Fused_Mul_', 'type': 'Multiply', 'version': 'opset1', 
# 'data': {'auto_broadcast': 'numpy'}, 
# 'input': {0: {'precision': 'FP32', 'dims': (1, 64, 14, 14)}, 
#           1: {'precision': 'FP32', 'dims': (1, 64, 1, 1)}}, 
# 'output': {2: {'precision': 'FP32', 'dims': (1, 64, 14, 14)}}}
