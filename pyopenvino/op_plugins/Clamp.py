# Clamp
import numpy as np
import common_def

def name():
    print('Clamp')


def kernel_Clamp_numpy(inputs:dict, min_val, max_val):
    input0 = inputs[0]
    res = np.clip(input0, min_val, max_val)
    return res


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    debug = True
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        assert data.dtype == common_def.type_convert_tbl[input_port['precision']]
        assert data.shape == input_port['dims']

    max_val = float(node['data']['max'])
    min_val = float(node['data']['min'])

    res = kernel_Clamp_numpy(inputs, min_val, max_val)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Relu6', 'type': 'Clamp', 'version': 'opset1', 
# 'data': {'max': '6', 'min': '0'}, 
# 'input': {0: {'precision': 'FP16', 'dims': (1, 32, 150, 150)}}, 
# 'output': {1: {'precision': 'FP16', 'dims': (1, 32, 150, 150)}}}
