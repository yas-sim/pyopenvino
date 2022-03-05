# StridedSlice
import numpy as np
import common_def

def name():
    print('Reshape')

def kernel_StridedSlice_naive(inputs, begin_mask, end_mask, new_axis_mask, shrink_axis_mask, ellipsis_mask):
    input  = inputs[0]
    begin  = inputs[1]
    end    = inputs[2]
    stride = inputs[3]

    # Generate array slicing formula dynamically (e.g. input[2:3:1])
    f = 'input['
    for dim in range(input.ndim):
        i_begin = begin[dim]
        i_end = end[dim]
        i_stride = stride[dim]
        f += '{}:{}:{},'.format(i_begin, i_end, i_stride)
    f = f[:-1] + ']'

    res = eval(f)
    return res


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        assert data.dtype == common_def.type_convert_tbl[input_port['precision']]
        assert data.shape == input_port['dims']

    # data
    begin_mask       = node['data']['begin_mask']
    end_mask         = node['data']['end_mask']
    new_axis_mask    = node['data']['new_axis_mask']
    shrink_axis_mask = node['data']['shrink_axis_mask']
    ellipsis_mask    = node['data']['ellipsis_mask']

    res = kernel_StridedSlice_naive(inputs, begin_mask, end_mask, new_axis_mask, shrink_axis_mask, ellipsis_mask)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# StridedSlice,  {'name': 'PriorBoxClustered_0/ss_1_port', 'type': 'StridedSlice', 'version': 'opset1', 
# 'data': {'begin_mask': '0', 'ellipsis_mask': '0', 'end_mask': '1', 'new_axis_mask': '0', 'shrink_axis_mask': '0'}, 
# 'input': {0: {'precision': 'I64', 'dims': (4,)}, 
#           1: {'precision': 'I64', 'dims': (1,)}, 
#           2: {'precision': 'I64', 'dims': (1,)}, 
#           3: {'precision': 'I64', 'dims': (1,)}}, 
# 'output': {4: {'precision': 'I64', 'dims': (2,)}}}

'''
<layer id="294" name="PriorBoxClustered_0/ss_1_port" type="StridedSlice" version="opset1">
    <data begin_mask="0" ellipsis_mask="0" end_mask="1" new_axis_mask="0" shrink_axis_mask="0"/>
    <input>
        <port id="0" precision="I64">
            <dim>4</dim>
        </port>
        <port id="1" precision="I64">
            <dim>1</dim>
        </port>
        <port id="2" precision="I64">
            <dim>1</dim>
        </port>
        <port id="3" precision="I64">
            <dim>1</dim>
        </port>
    </input>
    <output>
        <port id="4" precision="I64">
            <dim>2</dim>
        </port>
    </output>
</layer>
'''