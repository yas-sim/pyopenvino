# MatMul
import numpy as np
import common_def

def name():
    print('MatMul')

def MatMul(inputs):
    input0 = inputs[0]
    input1 = inputs[1]
    res = np.matmul(input0, input1.transpose((1,0)))
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

    res = MatMul(inputs)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'StatefulPartitionedCall/sequential/dense/MatMul', 'type': 'MatMul', 'version': 'opset1', 
# 'data': {'transpose_a': 'false', 'transpose_b': 'true'}, 
# 'input': {0: {'precision': 'FP32', 'dims': (1, 576)}, 
#           1: {'precision': 'FP32', 'dims': (64, 576)}}, 
# 'output': {2: {'precision': 'FP32', 'dims': (1, 64)}}}
