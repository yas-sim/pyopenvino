# SoftMax
import numpy as np
import common_def

def name():
    print('SoftMax')

def SoftMax(inputs):
    input0 = inputs[0]
    u = np.sum(np.exp(input0))
    res = np.exp(input0)/u
    return res

def compute(node:dict, inputs:dict=None, debug:bool=False):
    debug = True
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

    res = SoftMax(inputs)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'StatefulPartitionedCall/sequential/dense_1/Softmax', 'type': 'SoftMax', 'version': 'opset1', 
#   'data': {'axis': '1'}, 
#   'input': {0: {'precision': 'FP32', 'dims': (1, 10)}}, 
#   'output': {1: {'precision': 'FP32', 'dims': (1, 10)}}}
