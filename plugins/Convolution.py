# Convolution
import common_def
import numpy as np

def disp_result(data):
    N,C,H,W = data.shape
    for c in range(C):
        print('C=', c)
        for h in range(H):
            for w in range(W):
                print('{:6.3f},'.format(data[0,c,h,w]), end='')
            print()

def name():
    print('Convolution')

def conv2d(inputs, strides, dilation, pads_begin, pads_end, auto_pad):
    input          = inputs[0]
    kernel         = inputs[1]
    n, c, h, w     = input.shape   # Input image
    kn, kc, kh, kw = kernel.shape  # Kernel
    filters        = kn

    output = np.zeros((n, filters, h-kh+1, w-kw+1), np.float32)

    for fc in range(filters):
        for dy in range(0, h-kh+1, strides[0]):
            for dx in range(0, w-kw+1, strides[1]):
                cnv = 0
                for cc in range(c):
                    for fy in range(kh):
                        for fx in range(kw):
                            flt = kernel[fc, cc, fy, fx]		# fx and fy are swapped (matmul)
                            dt  = input[0, cc, dx+fx, dy+fy]
                            cnv += flt * dt						# Convolution
                output[0, fc, dy, dx] = cnv
    return output

def compute(node:dict, inputs:dict=None, debug:bool=False):
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        if data.dtype != common_def.type_convert_tbl[input_port['precision']]:
            print('input data precision mismatch')
            return None
        if data.shape != input_port['dims']:
            print('input data shape mismatch')
            return None
    strides =  common_def.string_to_tuple(node['data']['strides'])
    dilation = common_def.string_to_tuple(node['data']['dilations'])
    pads_begin = common_def.string_to_tuple(node['data']['pads_begin'])
    pads_end = common_def.string_to_tuple(node['data']['pads_end'])
    auto_pad = True if node['data']['auto_pad']=='valid' else False
    res = conv2d(inputs, strides, dilation, pads_begin, pads_end, auto_pad)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

#Convolution
#{'name': 'StatefulPartitionedCall/sequential/conv2d/Conv2D', 'type': 'Convolution', 'version': 'opset1', 
#   'data': {'strides': '1, 1', 'dilations': '1, 1', 'pads_begin': '0, 0', 'pads_end': '0, 0', 'auto_pad': 'valid'}, 
#   'input': {0: {'precision': 'FP32', 'dims': (1, 1, 28, 28)}, 
#             1: {'precision': 'FP32', 'dims': (32, 1, 3, 3)}},
#   'output': {2: {'precision': 'FP32', 'dims': (1, 32, 26, 26)}}}
