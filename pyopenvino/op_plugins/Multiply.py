# Multiply
import numpy as np
import common_def

def name():
    print('Multiply')


def kernel_Multiply_numpy(inputs:dict):
    input0 = inputs[0]
    input1 = inputs[1]
    input1 = np.broadcast_to(input1, input0.shape)
    res = input0 * input1
    return res


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
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

    res = kernel_Multiply_numpy(inputs)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'StatefulPartitionedCall/sequential/batch_normalization/FusedBatchNormV3/mean/Fused_Mul_', 'type': 'Multiply', 'version': 'opset1', 
# 'data': {'auto_broadcast': 'numpy'}, 
# 'input': {0: {'precision': 'FP32', 'dims': (1, 64, 14, 14)}, 
#           1: {'precision': 'FP32', 'dims': (1, 64, 1, 1)}}, 
# 'output': {2: {'precision': 'FP32', 'dims': (1, 64, 14, 14)}}}
