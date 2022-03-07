# DetectionOutput

import numpy as np
import common_def

def name():
    print('DetectionOutput')

# Calculate IOU for non-maximum suppression
def iou(a, b):
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    iou_x1 = np.maximum(a[0], b[0])
    iou_y1 = np.maximum(a[1], b[1])
    iou_x2 = np.minimum(a[2], b[2])
    iou_y2 = np.minimum(a[3], b[3])

    iou_w = iou_x2 - iou_x1
    iou_h = iou_y2 - iou_y1

    if iou_w < 0 or iou_h < 0:
        return 0.0

    area_iou = iou_w * iou_h
    iou = area_iou / (area_a + area_b - area_iou)

    return iou

def nms():
    '''
    # Do non-maximum suppression to reject the redundant objects on the overlap region
    for obj_id1, obj1 in enumerate(objects[:-2]):
        for obj_id2, obj2 in enumerate(objects[obj_id1+1:]):
            if obj1[6] == True and obj2[6]==True:
                IOU = iou(obj1[0:3+1], obj2[0:3+1])
                if IOU>0.5:
                    if obj1[4]<obj2[4]:
                        obj1[6] = False
                    else:
                        obj2[6] = False
    '''

def kernel_DetectionOutput_naive(inputs, num_classes, background_label_id, top_k, variance_encoded_in_target, keep_top_k, code_type, share_location,
                                        nms_threshold, confidence_threshold, clip_after_nms, clip_before_nms, decrease_label_id, normalized,
                                        input_height, input_width, objectness_score):
    box_logits = inputs[0]  # [ N, num_prior_boxes * num_loc_classes * 4 ]
    class_pred = inputs[1]  # [N, num_prior_boxes * prior_box_size ]
    proposals  = inputs[2]  # [ priors_batch_size, 1, num_prior_boxes * prior_box_size] or [priors_batch_size, 2, num_prior_boxes * prior_box_size ]


    return res

# 'input': {0: {'precision': 'FP16', 'dims': (1, 7668)},         # Box logits [ N, num_prior_boxes * num_loc_classes * 4 ]
#           1: {'precision': 'FP16', 'dims': (1, 174447)},       # class predictions [N, num_prior_boxes * prior_box_size ]
#           2: {'precision': 'FP32', 'dims': (1, 2, 7668)}},     # proposals [ priors_batch_size, 1, num_prior_boxes * prior_box_size] or [priors_batch_size, 2, num_prior_boxes * prior_box_size ]


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    debug = True
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        assert data.dtype == common_def.type_convert_tbl[input_port['precision']]
        assert data.shape == input_port['dims']

    num_classes                = int(node['data']['num_classes'])
    background_label_id        = int(node['data']['background_label_id'])                                 if 'background_label_id' in node['data'] else 0
    top_k                      = int(node['data']['top_k'])                                               if 'top_k' in node['data'] else -1
    variance_encoded_in_target = common_def.string_to_boolean(node['data']['variance_encoded_in_target']) if 'variance_encoded_in_target' in node['data'] else False
    keep_top_k                 = common_def.string_to_tuple(node['data']['keep_top_k'])
    code_type                  = node['data']['code_type']                                                if 'code_type' in node['data'] else 'caffe.PriorBoxParameter.CORNER'
    share_location             = common_def.string_to_boolean(node['data']['share_location'])             if 'share_location' in node['data'] else True
    nms_threshold              = float(node['data']['nms_threshold'])
    confidence_threshold       = float(node['data']['confidence_threshold'])                              if 'confidence_threshold' in node['data'] else 0
    clip_after_nms             = common_def.string_to_boolean(node['data']['clip_after_nms'])             if 'clip_after_nms' in node['data'] else False
    clip_before_nms            = common_def.string_to_boolean(node['data']['clip_before_nms'])            if 'clip_before_nms' in node['data'] else False
    decrease_label_id          = common_def.string_to_boolean(node['data']['decrease_label_id'])          if 'decrease_label_id' in node['data'] else False
    normalized                 = common_def.string_to_boolean(node['data']['normalized'])                 if 'normalized' in node['data'] else False
    input_height               = int(node['data']['input_height'])                                        if 'input_height' in node['data'] else 1
    input_width                = int(node['data']['input_width'])                                         if 'width' in node['data'] else 1
    objectness_score           = float(node['data']['objectness_score'])                                  if 'objectness_score' in node['data'] else 0

    res = kernel_DetectionOutput_naive(inputs, num_classes, background_label_id, top_k, variance_encoded_in_target, keep_top_k, code_type, share_location,
                                        nms_threshold, confidence_threshold, clip_after_nms, clip_before_nms, decrease_label_id, normalized,
                                        input_height, input_width, objectness_score)

    output_port_id = next(iter(node['output']))     # Get output port number
    res = { output_port_id:res }
    print('*'*40, 'Implementation has not completed yet (WIP)')
    assert False
    return res

# {'name': 'DetectionOutput', 'type': 'DetectionOutput', 'version': 'opset1', 
# 'data': {'background_label_id': '0', 'clip_after_nms': 'true', 'clip_before_nms': 'false', 'code_type': 'caffe.PriorBoxParameter.CENTER_SIZE', 
#           'confidence_threshold': '0.30000001192092896', 'decrease_label_id': 'false', 'input_height': '1', 'input_width': '1', 'keep_top_k': '100', 
#           'nms_threshold': '0.60000002384185791', 'normalized': 'true', 'num_classes': '91', 'objectness_score': '0', 'share_location': 'true', 
#           'top_k': '100', 'variance_encoded_in_target': 'false'}, 
# 'input': {0: {'precision': 'FP16', 'dims': (1, 7668)},         # Box logits [ N, num_prior_boxes * num_loc_classes * 4 ]
#           1: {'precision': 'FP16', 'dims': (1, 174447)},       # class predictions [N, num_prior_boxes * prior_box_size ]
#           2: {'precision': 'FP32', 'dims': (1, 2, 7668)}},     # proposals [ priors_batch_size, 1, num_prior_boxes * prior_box_size] or [priors_batch_size, 2, num_prior_boxes * prior_box_size ]
# 'output': {3: {'precision': 'FP16', 'dims': (1, 1, 100, 7)}}}
