# DetectionOutput

import numpy as np
import common_def

def name():
    print('DetectionOutput')


def kernel_DetectionOutput_naive(inputs, clip, width, height, step, step_h, step_w, offset, variance, img_h, img_w):
    grid_h, grid_w   = [ int(item) for item in inputs[0] ]
    image_h, image_w = [ int(item) for item in inputs[1] ]

    return res


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    debug = True
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        assert data.dtype == common_def.type_convert_tbl[input_port['precision']]
        assert data.shape == input_port['dims']

    clip = common_def.string_to_boolean(node['data']['clip'])             if 'clip' in node['data'] else True
    width = common_def.string_to_tuple_float(node['data']['width'])       if 'width' in node['data'] else [1.0]
    height = common_def.string_to_tuple_float(node['data']['height'])     if 'height' in node['data'] else [1.0]
    step = int(node['data']['step'])                                      if 'step' in node['data'] else 0.0
    step_h = int(node['data']['step_h'])                                  if 'step_h' in node['data'] else 0.0
    step_w = int(node['data']['step_w'])                                  if 'step_w' in node['data'] else 0.0
    offset = float(node['data']['offset'])
    variance = common_def.string_to_tuple_float(node['data']['variance']) if 'variance' in node['data'] else []
    img_h = float(node['data']['img_h'])                                  if 'img_h' in node['data'] else 0.0
    img_w = float(node['data']['img_w'])                                  if 'img_w' in node['data'] else 0.0

    res = kernel_DetectionOutput_naive(inputs, clip, width, height, step, step_h, step_w, offset, variance, img_h, img_w)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    return res

# {'name': 'PriorBoxClustered_0/naked_not_unsqueezed', 'type': 'PriorBoxClustered', 'version': 'opset1', 
# 'data': {'clip': 'false', 'height': '30, 42.4264, 84.8528', 'offset': '0.5', 'step': '0', 'step_h': '0', 'step_w': '0', 'variance': '0.1, 0.1, 0.2, 0.2', 'width': '30, 84.8528, 42.4264'}, 
# 'input': {0: {'precision': 'I64', 'dims': (2,)},  # 19, 19
#           1: {'precision': 'I64', 'dims': (2,)}},  # 300, 300
# 'output': {2: {'precision': 'FP32', 'dims': (2, 4332)}}}  # 4*height*width*priors_per_point   4332 = 4*(19*19*3)
