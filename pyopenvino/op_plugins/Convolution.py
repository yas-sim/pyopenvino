# Convolution
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


def kernel_conv2d_numpy(inputs, strides, dilation, pads_begin, pads_end, auto_pad):
    input          = inputs[0]
    kernel         = inputs[1]
    n, c, h, w     = input.shape   # Input image
    kn, kc, kh, kw = kernel.shape  # Kernel
    sh, sw         = strides
    pb0, pb1       = pads_begin
    pe0, pe1       = pads_end

    # output feature map size
    oh = (h-kh)//sh + 1
    ow = (w-kw)//sw + 1

    output = np.zeros((n, kn, oh+pb1+pe1, ow+pb0+pe0), dtype=np.float32)

    for fc in range(kn):  # Number of filters
        for dy in range(oh):
            for dx in range(ow):
                patch = input[0, :, dy*sh:dy*sh+kh, dx*sw:dx*sw+kw]
                output[0, fc, dy+pb1, dx+pb0] = np.sum(patch*kernel[fc])
    return output


def kernel_conv2d_naive(inputs, strides, dilation, pads_begin, pads_end, auto_pad):
    input          = inputs[0]
    kernel         = inputs[1]
    n, c, h, w     = input.shape   # Input image
    kn, kc, kh, kw = kernel.shape  # Kernel
    sh, sw         = strides
    pb0, pb1       = pads_begin
    pe0, pe1       = pads_end

    # output feature map size
    oh = (h-kh)//sh + 1
    ow = (w-kw)//sw + 1

    output = np.zeros((n, kn, oh+pb1+pe1, ow+pb0+pe0), dtype=np.float32)

    for fc in range(kn):  # Number of filters
        for dy in range(oh):
            for dx in range(ow):
                for cc in range(kc):
                    for fy in range(kh):
                        for fx in range(kw):
                            flt = kernel[fc, cc, fy, fx]
                            dt  = input[0, cc, dy*sh+fy, dx*sw+fx]
                            output[0, fc, dy+pb1, dx+pb0] += flt * dt
    return output


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
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

    print(kernel_type)
    if kernel_type == 'numpy':
        res = kernel_conv2d_numpy(inputs, strides, dilation, pads_begin, pads_end, auto_pad)
    else:
        res = kernel_conv2d_naive(inputs, strides, dilation, pads_begin, pads_end, auto_pad)

    output_port_id = next(iter(node['output']))     # Get output port number
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
