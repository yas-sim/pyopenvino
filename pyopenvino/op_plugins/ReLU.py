# ReLU
import numpy as np
import common_def

def name():
    print('ReLU')

def ReLU(inputs:dict):
    input0 = inputs[0]
    res = np.where(input0<0, 0, input0)      # ReLU
    return res

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

    res = ReLU(inputs)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'StatefulPartitionedCall/sequential/conv2d/Relu', 'type': 'ReLU', 'version': 'opset1', 
#  'input': {0: {'precision': 'FP32', 'dims': (1, 32, 26, 26)}}, 
#  'output': {1: {'precision': 'FP32', 'dims': (1, 32, 26, 26)}}}
