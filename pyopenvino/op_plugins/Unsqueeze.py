# Unsqueeze
import numpy as np
import common_def

def name():
    print('Unsqueeze')


def kernel_Unsqueeze_numpy(inputs):
    input0 = inputs[0]
    input1 = inputs[1]   # dims

    res = np.expand_dims(input0, list(input1))   # np.expand_dims([3,3], (0,3)) -> [1,3,3,1]
    return res


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        assert data.dtype == common_def.type_convert_tbl[input_port['precision']]
        assert data.shape == input_port['dims']

    res = kernel_Unsqueeze_numpy(inputs)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'PriorBoxClustered_0', 'type': 'Unsqueeze', 'version': 'opset1', 
# 'input': {0: {'precision': 'FP32', 'dims': (2, 4332)}, 
#           1: {'precision': 'I64', 'dims': (1,)}}, 
# 'output': {2: {'precision': 'FP32', 'dims': (1, 2, 4332)}}}
