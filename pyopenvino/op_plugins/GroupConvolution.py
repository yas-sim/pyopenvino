# GroupConvolution
from ctypes.wintypes import DWORD
import math
import common_def
import numpy as np

def name():
    print('GroupConvolution')


def disp_result(data):
    N,C,H,W = data.shape
    for c in range(C):
        print('C=', c)
        for h in range(H):
            for w in range(W):
                print('{:6.3f},'.format(data[0,c,h,w]), end='')
            print()

# ---------------------------------------------------------------------------------------

def calc_output_shape_group_conv(input_dim:tuple, kernel_dim:tuple, strides:tuple, pads_begin:tuple, pads_end:tuple, rounding_type:str, auto_pad:str):
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
            oh = math.ceil(h/sh)
            ow = math.ceil(w/sw)
    
    return (oh, ow)


def kernel_GroupConvolution_numpy(inputs, strides, dilation, pads_begin, pads_end, auto_pad):
    input                   = inputs[0]      # [N, GROUPS * C_IN, Y, X]  1,32,150,150
    kernel                  = inputs[1]      # [GROUPS, C_OUT, C_IN, Y, X]  32,1,1,3,3
    n, c, h, w              = input.shape
    grp, ch_o, ch_i, kh, kw = kernel.shape
    sh, sw                  = strides
    pbh, pbw                = pads_begin
    peh, pew                = pads_end
    dh, dw                  = dilation

    # output feature map size
    oh, ow = calc_output_shape_group_conv((h, w), (kh, kw), (sh, sw), pads_begin, pads_end, 'floor', auto_pad)

    input = np.pad(input, [(0,0), (0,0), (pbh, peh), (pbw, pew)], 'constant')
    output = np.zeros((n, grp*ch_o, oh, ow), dtype=input.dtype)    # [ N, GROUPS * C_OUT, Y, X ]

    n, c, h, w = input.shape   # Input image (padded)

    for ci in range(ch_i):  # C_IN
        for gp in range(grp): # GROUPS
            for co in range(ch_o):
                for dy in range(oh):
                    for dx in range(ow):
                        flt = kernel[gp, co, ci, :, :]
                        patch = input[0, gp*ci+gp, dy*sh:dy*sh+kh, dx*sw:dx*sw+kw]
                        output[0, gp*co+gp, dy, dx] = np.sum(patch*flt)
    return output

# ---------------------------------------------------------------------------------------

def kernel_GroupConvolution_naive(inputs, strides, dilation, pads_begin, pads_end, auto_pad):
    input                   = inputs[0]      # [N, GROUPS * C_IN, Y, X]
    kernel                  = inputs[1]      # [GROUPS, C_OUT, C_IN, Y, X]
    n, c, h, w              = input.shape
    grp, ch_o, ch_i, kh, kw = kernel.shape
    sh, sw                  = strides
    pbh, pbw                = pads_begin
    peh, pew                = pads_end
    dh, dw                  = dilation

    # output feature map size
    oh, ow = calc_output_shape_group_conv((h, w), (kh, kw), (sh, sw), pads_begin, pads_end, 'floor', auto_pad)

    input = np.pad(input, [(0,0), (0,0), (pbh, peh), (pbw, pew)], 'constant')
    output = np.zeros((n, grp*ch_o, oh, ow), dtype=input.dtype)    # [ N, GROUPS * C_OUT, Y, X ]

    n, c, h, w = input.shape   # Input image (padded)

    for ci in range(ch_i):  # C_IN
        for gp in range(grp): # GROUPS
            for co in range(ch_o):
                for dy in range(oh):
                    for dx in range(ow):
                        for fy in range(kh):
                            for fx in range(kw):
                                flt = kernel[gp, co, ci, fy, fx]
                                dt  = input[0, gp*ci+gp, dy*sh+fy*dh, dx*sw+fx*dw]
                                output[0, gp*co+gp, dy, dx] += flt*dt
    return output

# ---------------------------------------------------------------------------------------
def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        assert data.dtype == common_def.type_convert_tbl[input_port['precision']]
        assert data.shape == input_port['dims']

    strides    = common_def.string_to_tuple(node['data']['strides'])
    dilation   = common_def.string_to_tuple(node['data']['dilations'])
    pads_begin = common_def.string_to_tuple(node['data']['pads_begin'])
    pads_end   = common_def.string_to_tuple(node['data']['pads_end'])
    auto_pad   = node['data']['auto_pad']

    if kernel_type == 'naive':
        res = kernel_GroupConvolution_naive(inputs, strides, dilation, pads_begin, pads_end, auto_pad)
    else:
        res = kernel_GroupConvolution_numpy(inputs, strides, dilation, pads_begin, pads_end, auto_pad)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise', 'type': 'GroupConvolution', 'version': 'opset1', 
# 'data': {'auto_pad': 'same_upper', 'dilations': '1, 1', 'pads_begin': '1, 1', 'pads_end': '1, 1', 'strides': '1, 1'}, 
# 'input': {0: {'precision': 'FP16', 'dims': (1, 32, 150, 150)}, 
#           1: {'precision': 'FP16', 'dims': (32, 1, 1, 3, 3)}}, 
# 'output': {2: {'precision': 'FP16', 'dims': (1, 32, 150, 150)}}}
