# Convolution
from ctypes.wintypes import DWORD
import math
import common_def
import numpy as np

def name():
    print('Convolution')


def disp_result(data):
    N,C,H,W = data.shape
    for c in range(C):
        print('C=', c)
        for h in range(H):
            for w in range(W):
                print('{:6.3f},'.format(data[0,c,h,w]), end='')
            print()


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
            oh = math.ceil(h/sh)
            ow = math.ceil(w/sw)
    
    return (oh, ow)


# ---------------------------------------------------------------------------------------

# Referred from 'deep-learning-from-scratch' project
# https://github.com/oreilly-japan/deep-learning-from-scratch

def im2col(input, kh, kw, strides, pads_begin, pads_end, oh, ow):
    pbh, pbw       = pads_begin
    peh, pew       = pads_end
    n, c, h, w     = input.shape
    sh, sw         = strides

    img = np.pad(input, [(0,0), (0,0), (pbh, peh), (pbw, pew)], 'constant')
    col = np.zeros((n, c, kh, kw, oh, ow), dtype=np.float32)

    for y in range(kh):
        for x in range(kw):
            col[:, :, y, x, :, :] = img[:, :, y:y+sh*oh:sh, x:x+sw*ow:sw]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n*oh*ow, -1)
    return col

def kernel_Convolution_im2col(inputs, strides, dilation, pads_begin, pads_end, auto_pad):
    input          = inputs[0]
    kernel         = inputs[1]
    n, c, h, w     = input.shape   # Input image
    kn, kc, kh, kw = kernel.shape  # Kernel
    sh, sw         = strides

    # output feature map size
    oh, ow = calc_output_shape((h, w), (kh, kw), (sh, sw), pads_begin, pads_end, 'floor', auto_pad)

    col = im2col(input, kh, kw, strides, pads_begin, pads_end, oh, ow)
    col_W = kernel.reshape(kn, -1).T
    output = np.dot(col, col_W)
    output = output.reshape(n, oh, ow, -1).transpose(0, 3, 1, 2)
    output = output
    return output

# ---------------------------------------------------------------------------------------

def kernel_Convolution_numpy(inputs, strides, dilation, pads_begin, pads_end, auto_pad):
    input          = inputs[0]
    kernel         = inputs[1]
    n, c, h, w     = input.shape   # Input image
    kn, kc, kh, kw = kernel.shape  # Kernel
    sh, sw         = strides
    pbh, pbw       = pads_begin
    peh, pew       = pads_end
    dh, dw         = dilation

    # output feature map size
    oh, ow = calc_output_shape((h, w), (kh, kw), (sh, sw), pads_begin, pads_end, 'floor', auto_pad)

    input = np.pad(input, [(0,0), (0,0), (pbh, peh), (pbw, pew)], 'constant')
    output = np.zeros((n, kn, oh, ow), dtype=np.float32)

    n, c, h, w = input.shape   # Input image (padded)

    for fc in range(kn):  # Number of filters
        for dy in range(oh):
            for dx in range(ow):
                patch = input[0, :, dy*sh:dy*sh+kh:dh, dx*sw:dx*sw+kw:dw]
                output[0, fc, dy, dx] = np.sum(patch*kernel[fc])
    return output

# ---------------------------------------------------------------------------------------

def kernel_Convolution_naive(inputs, strides, dilation, pads_begin, pads_end, auto_pad):
    input          = inputs[0]
    kernel         = inputs[1]
    n, c, h, w     = input.shape   # Input image
    kn, kc, kh, kw = kernel.shape  # Kernel
    sh, sw         = strides
    pbh, pbw       = pads_begin
    peh, pew       = pads_end
    dh, dw         = dilation

    # output feature map size
    oh, ow = calc_output_shape((h, w), (kh, kw), (sh, sw), pads_begin, pads_end, 'floor', auto_pad)

    input = np.pad(input, [(0,0), (0,0), (pbh, peh), (pbw, pew)], 'constant')
    output = np.zeros((n, kn, oh, ow), dtype=np.float32)

    n, c, h, w = input.shape   # Input image (padded)

    for fc in range(kn):  # Number of filters
        for dy in range(oh):
            for dx in range(ow):
                for cc in range(kc):
                    for fy in range(kh):
                        for fx in range(kw):
                            flt = kernel[fc, cc, fy, fx]
                            dt  = input[0, cc, dy*sh+fy*dh, dx*sw+fx*dw]
                            output[0, fc, dy, dx] += flt * dt
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
        res = kernel_Convolution_naive(inputs, strides, dilation, pads_begin, pads_end, auto_pad)
    elif kernel_type == 'special':
        res = kernel_Convolution_im2col(inputs, strides, dilation, pads_begin, pads_end, auto_pad)
    else:
        res = kernel_Convolution_numpy(inputs, strides, dilation, pads_begin, pads_end, auto_pad)

    output_port_id = next(iter(node['output']))     # Get output port number
    output_precision = common_def.type_convert_tbl[node['output'][output_port_id]['precision']]
    res = res.astype(output_precision)
    res = { output_port_id:res }
    return res

# {'name': 'StatefulPartitionedCall/sequential/conv2d_1/Conv2D', 'type': 'Convolution', 'version': 'opset1', 
# 'data': {'strides': '1, 1', 'dilations': '1, 1', 'pads_begin': '0, 0', 'pads_end': '0, 0', 'auto_pad': 'valid'}, 
# 'input': {0: {'precision': 'FP32', 'dims': (1, 32, 13, 13)}, 
#           1: {'precision': 'FP32', 'dims': (64, 32, 3, 3)}}, 
# 'output': {2: {'precision': 'FP32', 'dims': (1, 64, 11, 11)}}}

#{'name': 'StatefulPartitionedCall/sequential/conv2d/Conv2D', 'type': 'Convolution', 'version': 'opset1', 
#   'data': {'strides': '1, 1', 'dilations': '1, 1', 'pads_begin': '0, 0', 'pads_end': '0, 0', 'auto_pad': 'valid'}, 
#   'input': {0: {'precision': 'FP32', 'dims': (1, 1, 28, 28)}, 
#             1: {'precision': 'FP32', 'dims': (32, 1, 3, 3)}},
#   'output': {2: {'precision': 'FP32', 'dims': (1, 32, 26, 26)}}}
