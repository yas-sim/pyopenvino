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


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        assert data.dtype == common_def.type_convert_tbl[input_port['precision']]
        assert data.shape == input_port['dims']

    res = kernel_Add_numpy(inputs)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'StatefulPartitionedCall/sequential/conv2d/BiasAdd/Add', 'type': 'Add', 'version': 'opset1', 
#   'data': {'auto_broadcast': 'numpy'}, 
#   'input': {0: {'precision': 'FP32', 'dims': (1, 32, 26, 26)}, 
#             1: {'precision': 'FP32', 'dims': (1, 32, 1, 1)}}, 
#   'output': {2: {'precision': 'FP32', 'dims': (1, 32, 26, 26)}}}
