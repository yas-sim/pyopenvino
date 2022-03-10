# Clamp
import numpy as np
import common_def

def name():
    print('Clamp')


def kernel_Clamp_numpy(inputs:dict, min_val, max_val):
    input0 = inputs[0]
    res = np.clip(input0, min_val, max_val)
    return res


def kernel_Clamp_naive(inputs:dict, min_val, max_val):
    input0 = inputs[0]

    assert input0.ndim <= 4
    shape_orig = input0.shape
    if input0.ndim < 4:         # Convert tensor to rank-4 tensor
        input0 = np.expand_dims(input0, tuple(range(4-input0.ndim)))

    output = np.zeros_like(input0)
    shape = input0.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    output[i,j,k,l] = max(min_val, min(max_val, input0[i,j,k,l]))

    return output.reshape(shape_orig)


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        assert data.dtype == common_def.type_convert_tbl[input_port['precision']]
        assert data.shape == input_port['dims']

    max_val = float(node['data']['max'])
    min_val = float(node['data']['min'])

    if kernel_type == 'naive':
        res = kernel_Clamp_naive(inputs, min_val, max_val)
    else:
        res = kernel_Clamp_numpy(inputs, min_val, max_val)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Relu6', 'type': 'Clamp', 'version': 'opset1', 
# 'data': {'max': '6', 'min': '0'}, 
# 'input': {0: {'precision': 'FP16', 'dims': (1, 32, 150, 150)}}, 
# 'output': {1: {'precision': 'FP16', 'dims': (1, 32, 150, 150)}}}
