# Add
import numpy as np
import common_def

def name():
    print('Add')


def kernel_Add_numpy(inputs:dict):
    input0 = inputs[0]
    input1 = inputs[1]
    input1 = np.broadcast_to(input1, input0.shape)
    res = input0 + input1
    return res


def kernel_Add_naive(inputs:dict):
    input0 = inputs[0]
    input1 = inputs[1]
    input1 = np.broadcast_to(input1, input0.shape)

    assert input0.ndim == 4
    shape = input0.shape
    output = np.zeros_like(input0)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    output[i,j,k,l] = input0[i,j,k,l] + input1[i,j,k,l]
    return output


def kernel_Add_naive2(inputs:dict):
    input0 = inputs[0]
    input1 = inputs[1]
    input1 = np.broadcast_to(input1, input0.shape)

    assert input0.ndim <= 4
    shape_orig = input0.shape
    if input0.ndim < 4:         # Convert tensors to rank-4 tensors
        input0 = np.expand_dims(input0, tuple(range(4-input0.ndim)))
        input1 = np.expand_dims(input1, tuple(range(4-input1.ndim)))

    output = np.zeros_like(input0)
    shape = input0.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    output[i,j,k,l] = input0[i,j,k,l] + input1[i,j,k,l]

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
        res = kernel_Add_naive2(inputs)
    else:
        res = kernel_Add_numpy(inputs)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'StatefulPartitionedCall/sequential/conv2d/BiasAdd/Add', 'type': 'Add', 'version': 'opset1', 
#   'data': {'auto_broadcast': 'numpy'}, 
#   'input': {0: {'precision': 'FP32', 'dims': (1, 32, 26, 26)}, 
#             1: {'precision': 'FP32', 'dims': (1, 32, 1, 1)}}, 
#   'output': {2: {'precision': 'FP32', 'dims': (1, 32, 26, 26)}}}
