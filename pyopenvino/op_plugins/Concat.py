# Concat
import numpy as np
import common_def

def name():
    print('Concat')

# Concat can take arbitary number of inputs
def kernel_Concat_numpy(inputs:dict, axis):
    assert len(inputs)>1
    tmp = [ data for data in inputs.values() ]
    res = np.concatenate(tmp, axis = axis)
    return res


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        assert data.dtype == common_def.type_convert_tbl[input_port['precision']]
        assert data.shape == input_port['dims']

    axis = int(node['data']['axis'])
    assert axis <= inputs[0].ndim

    res = kernel_Concat_numpy(inputs, axis)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'inception_3a/output', 'type': 'Concat', 'version': 'opset1', 
# 'data': {'axis': '1'}, 
# 'input': {0: {'precision': 'FP16', 'dims': (1, 64, 28, 28)}, 1: {'precision': 'FP16', 'dims': (1, 128, 28, 28)}, 2: {'precision': 'FP16', 'dims': (1, 32, 28, 28)}, 3: {'precision': 'FP16', 'dims': (1, 32, 28, 28)}}, 
# 'output': {4: {'precision': 'FP16', 'dims': (1, 256, 28, 28)}}}
