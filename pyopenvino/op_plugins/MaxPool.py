# MaxPool
import math
import numpy as np
import common_def

def name():
    print('MaxPool')


def calc_output_shape(input_dim:tuple, kernel_dim:tuple, strides:tuple, pads_begin:tuple, pads_end:tuple, rounding_type:str, auto_pad:str):
    h, w     = input_dim
    kh, kw   = kernel_dim
    sh, sw   = strides
    pb0, pb1 = pads_begin
    pe0, pe1 = pads_end

    # output feature map size
    assert auto_pad in [ 'explicit', 'valid', 'same_upper', 'same_lower' ]
    assert rounding_type in [ 'floor', 'ceil' ]
    if auto_pad == 'explicit':
        if rounding_type == 'floor':
            oh = math.floor((h + pb0 + pe0 - kh)/sh) + 1
            ow = math.floor((w + pb1 + pe1 - kw)/sw) + 1
        elif rounding_type == 'ceil':
            oh = math.ceil((h + pb0 + pe0 - kh)/sh) + 1
            ow = math.ceil((w + pb1 + pe1 - kw)/sw) + 1
    elif auto_pad == 'valid':
        if rounding_type == 'floor':
            oh = math.floor((h - kh)/sh) + 1
            ow = math.floor((w - kw)/sw) + 1
        if rounding_type == 'ceil':
            oh = math.ceil((h - kh)/sh) + 1
            ow = math.ceil((w - kw)/sw) + 1
    elif auto_pad == 'same_upper' or auto_pad == 'same_lower':
            oh = h
            ow = w
    
    return (oh, ow)


def kernel_MaxPool_numpy(inputs:dict, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad):
    input0   = inputs[0]
    n,c,h,w  = input0.shape
    sh, sw   = strides
    kh, kw   = kernel
    pb0, pb1 = pads_begin
    pe0, pe1 = pads_end

    # output feature map size
    oh, ow = common_def.calc_output_shape((h, w), (kh, kw), (sh, sw), pads_begin, pads_end, rounding_type, auto_pad)

    res = np.zeros((n, c, oh, ow), dtype=input0.dtype)
    input0 = np.pad(input0, [(0,0), (0,0), (pb0, pe0), (pb1, pe1)], 'constant')

    n,c,h,w  = input0.shape

    """
    # A bit naive version of MaxPool-Numpy
    for bn in range(n):
        for ch in range(c):
            for y in range(oh):
                for x in range(ow):
                    patch = input0[bn, ch, y*sh:min(h, y*sh+kh), x*sw:min(w, x*sw+kw)]
                    max_val = np.max(patch)
                    res[bn, ch, y, x] = max_val
    """
    for y in range(oh):
        for x in range(ow):
            patch = input0[:, :, y*sh:min(h, y*sh+kh), x*sw:min(w, x*sw+kw)]
            max_val = np.max(patch, axis=(2,3))
            res[:, :, y, x] = max_val
    return res


def kernel_MaxPool_naive(inputs:dict, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad):
    input0 = inputs[0]
    n,c,h,w = input0.shape
    sh, sw  = strides
    kh, kw  = kernel
    pb0, pb1 = pads_begin
    pe0, pe1 = pads_end

    # output feature map size
    oh, ow = common_def.calc_output_shape((h, w), (kh, kw), (sh, sw), pads_begin, pads_end, rounding_type, auto_pad)

    res = np.zeros((n, c, oh, ow), dtype=input0.dtype)
    input0 = np.pad(input0, [(0,0), (0,0), (pb0, pe0), (pb1, pe1)], 'constant')

    n,c,h,w  = input0.shape

    for bn in range(n):
        for ch in range(c):
            for y in range(oh):
                for x in range(ow):
                    max_val = 0
                    for ky in range(kh):
                        iy = y*sh+ky
                        if iy>=h:
                            continue
                        for kx in range(kw):
                            ix = x*sw+kx
                            if ix>=w:
                                continue
                            val = input0[bn, ch, iy, ix]
                            if max_val < val:
                                max_val = val
                    res[bn, ch, y, x] = max_val
    return res


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        assert data.dtype == common_def.type_convert_tbl[input_port['precision']]
        assert data.shape == input_port['dims']

    strides       = common_def.string_to_tuple(node['data']['strides'])
    pads_begin    = common_def.string_to_tuple(node['data']['pads_begin'])
    pads_end      = common_def.string_to_tuple(node['data']['pads_end'])
    kernel        = common_def.string_to_tuple(node['data']['kernel'])
    rounding_type = node['data']['rounding_type']
    auto_pad      = node['data']['auto_pad']

    if kernel_type == 'naive':
        res = kernel_MaxPool_naive(inputs, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad)
    else:
        res = kernel_MaxPool_numpy(inputs, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'StatefulPartitionedCall/sequential/max_pooling2d/MaxPool', 'type': 'MaxPool', 'version': 'opset1', 
#  'data': {'strides': '2, 2', 'pads_begin': '0, 0', 'pads_end': '0, 0', 'kernel': '2, 2', 'rounding_type': 'floor', 'auto_pad': 'valid'}, 
#  'input': {0: {'precision': 'FP32', 'dims': (1, 32, 26, 26)}}, 
#  'output': {1: {'precision': 'FP32', 'dims': (1, 32, 13, 13)}}}
