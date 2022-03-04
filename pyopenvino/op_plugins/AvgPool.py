# AvgPool
import math
import numpy as np
import common_def

def name():
    print('AvgPool')


def kernel_AvgPool_numpy(inputs:dict, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad):
    input0   = inputs[0]
    n,c,h,w  = input0.shape
    sh, sw   = strides
    kh, kw   = kernel

    # output feature map size
    oh, ow = common_def.calc_output_shape((h, w), (kh, kw), (sh, sw), pads_begin, pads_end, rounding_type, auto_pad)

    res = np.zeros((n, c, oh, ow), dtype=input0.dtype)

    for bn in range(n):
        for ch in range(c):
            for y in range(oh):
                for x in range(ow):
                    patch = input0[bn, ch, y*sh:min(h-1, y*sh+kh), x*sw:min(w-1, x*sw+kw)]
                    avg_val = np.average(patch)
                    res[bn, ch, y, x] = avg_val
    return res


def kernel_AvgPool_naive(inputs:dict, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad):
    input0 = inputs[0]
    n,c,h,w = input0.shape
    sh, sw  = strides
    kh, kw  = kernel

    # output feature map size
    oh, ow = common_def.calc_output_shape((h, w), (kh, kw), (sh, sw), pads_begin, pads_end, rounding_type, auto_pad)

    res = np.zeros((n, c, oh, ow), dtype=input0.dtype)

    for bn in range(n):
        for ch in range(c):
            for y in range(oh):
                for x in range(ow):
                    total_val = 0
                    count = 0
                    for ky in range(kh):
                        iy = y*sh+ky
                        if iy>=h:
                            continue
                        for kx in range(kw):
                            ix = x*sw+kx
                            if ix>=w:
                                continue
                            total_val += input0[bn, ch, y*sh + ky, x*sw + kx]
                            count += 1
                    assert count > 0
                    res[bn, ch, y, x] = total_val / count
    return res


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        assert data.dtype == common_def.type_convert_tbl[input_port['precision']]
        assert data.shape == input_port['dims']

    strides = common_def.string_to_tuple(node['data']['strides'])
    pads_begin = common_def.string_to_tuple(node['data']['pads_begin'])
    pads_end = common_def.string_to_tuple(node['data']['pads_end'])
    kernel = common_def.string_to_tuple(node['data']['kernel'])
    rounding_type = node['data']['rounding_type']
    auto_pad = node['data']['auto_pad']

    if kernel_type == 'naive':
        res = kernel_AvgPool_naive(inputs, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad)
    else:
        res = kernel_AvgPool_numpy(inputs, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'pool5/7x7_s1', 'type': 'AvgPool', 'version': 'opset1', 
# 'data': {'auto_pad': 'explicit', 'exclude-pad': 'false', 'kernel': '7, 7', 'pads_begin': '0, 0', 'pads_end': '0, 0', 'rounding_type': 'ceil', 'strides': '1, 1'}, 
# 'input': {0: {'precision': 'FP16', 'dims': (1, 1024, 7, 7)}}, 
# 'output': {1: {'precision': 'FP16', 'dims': (1, 1024, 1, 1)}}}
