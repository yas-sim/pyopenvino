# ShapeOf
import numpy as np
import common_def

def name():
    print('ShapeOf')


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        input_dtype = common_def.type_convert_tbl[input_port['precision']]
        input_shape = input_port['dims']
        assert data.dtype == input_dtype
        assert data.shape == input_shape

    res = np.array(input_shape, dtype=common_def.type_convert_tbl[node['output'][1]['precision']])

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'PriorBoxClustered_0/1_port', 'type': 'ShapeOf', 'version': 'opset3', 
# 'data': {'output_type': 'i64'}, 
# 'input': {0: {'precision': 'FP16', 'dims': (1, 3, 300, 300)}}, 
# 'output': {1: {'precision': 'I64', 'dims': (4,)}}}
