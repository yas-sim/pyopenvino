# Transpose
import numpy as np
import common_def

def name():
    print('Transpose')

def Transpose(inputs):
    input0 = inputs[0]
    input1 = inputs[1]   # axes
    res = input0.transpose(input1)
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

    res = Transpose(inputs)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'StatefulPartitionedCall/sequential/target_conv_layer/Relu/Transpose', 'type': 'Transpose', 'version': 'opset1', 
#  'input': {0: {'precision': 'FP32', 'dims': (1, 64, 3, 3)}, 
#            1: {'precision': 'I64', 'dims': (4,)}}, 
# 'output': {2: {'precision': 'FP32', 'dims': (1, 3, 3, 64)}}}
