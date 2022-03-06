# Sigmoid
import math
import numpy as np
import common_def

def name():
    print('Sigmoid')


def kernel_Sigmoid_numpy(inputs:dict):
    input0 = inputs[0]
    res = 1/(1+np.exp(-input0))
    return res


def kernel_Sigmoid_naive(inputs:dict):
    input0 = inputs[0].ravel()
    output = np.zeros((input0.size), dtype=input0.dtype)
    for i in range(output.size):
        output[i] = 1/(1+math.exp(-input0[i]))
    return output.reshape(inputs[0].shape)


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        assert data.dtype == common_def.type_convert_tbl[input_port['precision']]
        assert data.shape == input_port['dims']

    if kernel_type == 'naive':
        res = kernel_Sigmoid_naive(inputs)
    else:
        res = kernel_Sigmoid_numpy(inputs)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'do_ExpandDims_conf/sigmoid', 'type': 'Sigmoid', 'version': 'opset1', 
# 'input': {0: {'precision': 'FP16', 'dims': (1, 1, 1917, 91)}}, 
# 'output': {1: {'precision': 'FP16', 'dims': (1, 1, 1917, 91)}}}
