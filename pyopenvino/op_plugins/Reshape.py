# Reshape
import numpy as np
import common_def

def name():
    print('Reshape')

'''
Reshape takes two input tensors: data to be resized and shape of the new output. The values in the shape could be -1, 0 and any positive integer number. The two special values -1 and 0 :
0 means “copy the respective dimension *(left aligned)* of the input tensor” if special_zero is set to true; otherwise it is a normal dimension and is applicable to empty tensors.
-1 means that this dimension is calculated to keep the overall elements count the same as in the input tensor. Not more than one -1 can be used in a reshape operation.
'''

def kernel_Reshape_numpy(inputs):
    input0 = inputs[0]
    input1 = inputs[1]   # dims
    size = input0.size

    # Process special dimension value '0' and '-1'
    zero_flag = True    # '0' must be left aligned. Once non-zero appeared, no more 0 can exist. OK:[0,0,1,9000] Not allowed:[0,1,0,9000] 
    adjusted_dims = []
    deferred_dim = -1   # the axis (dim) number deferred to detemine the value due to '-1'. Differed value will be calculated later
    for idx, dim in enumerate(input1):
        if dim == 0:
            assert zero_flag == True
            i0_dim = input0.shape[idx]
            assert (size / i0_dim) == (size // i0_dim)    # i0_dim must be divisor of the size
            adjusted_dims.append(int(i0_dim))
            size /= i0_dim
        else:
            zero_flag = False
            if dim == -1:
                assert deferred_dim == -1                 # Multiple -1 is not allowed
                deferred_dim = idx
                adjusted_dims.append(-1)
            else:
                assert (size / dim) == (size // dim)
                adjusted_dims.append(int(dim))
                size /= dim
    if deferred_dim != -1:
        adjusted_dims[deferred_dim] = int(size)

    res = input0.reshape(adjusted_dims)
    return res


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        assert data.dtype == common_def.type_convert_tbl[input_port['precision']]
        assert data.shape == input_port['dims']

    res = kernel_Reshape_numpy(inputs)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'StatefulPartitionedCall/sequential/flatten/Reshape', 'type': 'Reshape', 'version': 'opset1', 'data': {'special_zero': 'false'}, 
# 'input': {0: {'precision': 'FP32', 'dims': (1, 3, 3, 64)}, 
#           1: {'precision': 'I64', 'dims': (2,)}}, 
# 'output': {2: {'precision': 'FP32', 'dims': (1, 576)}}}
