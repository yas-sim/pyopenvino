# MaxPool
import numpy as np
import common_def

def name():
    print('MaxPool')


def kernel_MaxPool_numpy(inputs:dict, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad):
    input0 = inputs[0]
    n,c,h,w = input0.shape
    sh, sw  = strides
    kh, kw  = kernel

    # output feature map size
    oh = (h-kh)//sh + 1
    ow = (w-kw)//sw + 1

    res = np.zeros((n, c, oh, ow), dtype=np.float32)

    for bn in range(n):
        for ch in range(c):
            for y in range(oh):
                for x in range(ow):
                    patch = input0[bn, ch, y*sh:y*sh+kh, x*sw:x*sw+kw]
                    max_val = np.max(patch)
                    res[bn, ch, y, x] = max_val
    return res


def kernel_MaxPool_naive(inputs:dict, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad):
    input0 = inputs[0]
    n,c,h,w = input0.shape
    sh, sw  = strides
    kh, kw  = kernel

    # output feature map size
    oh = (h-kh)//sh + 1
    ow = (w-kw)//sw + 1

    res = np.zeros((n, c, oh, ow), dtype=np.float32)

    for bn in range(n):
        for ch in range(c):
            for y in range(oh):
                for x in range(ow):
                    max_val = 0
                    for ky in range(kh):
                        for kx in range(kw):
                            val = input0[bn, ch, y*sh + ky, x*sw + kx]
                            if max_val < val:
                                max_val = val
                    res[bn, ch, y, x] = max_val
    return res


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

    strides = common_def.string_to_tuple(node['data']['strides'])
    pads_begin = common_def.string_to_tuple(node['data']['pads_begin'])
    pads_end = common_def.string_to_tuple(node['data']['pads_end'])
    kernel = common_def.string_to_tuple(node['data']['kernel'])
    rounding_type = node['data']['rounding_type']
    auto_pad = node['data']['auto_pad']

    #res = kernel_MaxPool_numpy(inputs, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad)
    res = kernel_MaxPool_naive(inputs, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'StatefulPartitionedCall/sequential/max_pooling2d/MaxPool', 'type': 'MaxPool', 'version': 'opset1', 
#  'data': {'strides': '2, 2', 'pads_begin': '0, 0', 'pads_end': '0, 0', 'kernel': '2, 2', 'rounding_type': 'floor', 'auto_pad': 'valid'}, 
#  'input': {0: {'precision': 'FP32', 'dims': (1, 32, 26, 26)}}, 
#  'output': {1: {'precision': 'FP32', 'dims': (1, 32, 13, 13)}}}
