# ReLU
import numpy as np
import common_def

def name():
    print('ReLU')


def kernel_ReLU_numpy(inputs:dict):
    input0 = inputs[0]
    res = np.where(input0<0, 0, input0)      # ReLU
    return res


def kernel_ReLU_naive(inputs:dict):
    input0 = inputs[0].ravel()
    output = np.zeros((input0.size), dtype=input0.dtype)
    for i in range(output.size):
        output[i] = 0 if input0[i]<0 else input0[i]
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
        res = kernel_ReLU_naive(inputs)
    else:
        res = kernel_ReLU_numpy(inputs)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'StatefulPartitionedCall/sequential/conv2d/Relu', 'type': 'ReLU', 'version': 'opset1', 
#  'input': {0: {'precision': 'FP32', 'dims': (1, 32, 26, 26)}}, 
#  'output': {1: {'precision': 'FP32', 'dims': (1, 32, 26, 26)}}}
