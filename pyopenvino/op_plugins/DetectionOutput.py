# DetectionOutput

import math
from types import new_class
import numpy as np
import common_def

def name():
    print('DetectionOutput')

# Calculate IOU for non-maximum suppression
def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)

    iou_x1 = max(ax0, bx0)
    iou_y1 = max(ay0, by0)
    iou_x2 = min(ax1, bx1)
    iou_y2 = min(ay1, by1)

    iou_w = iou_x2 - iou_x1
    iou_h = iou_y2 - iou_y1

    if iou_w < 0 or iou_h < 0:
        return 0.0

    try:
        area_iou = iou_w * iou_h
        iou = area_iou / (area_a + area_b - area_iou)
    except ZeroDivisionError as e:
        iou = 1e9
    return iou

import cv2

def nms(decoded_bboxes, class_num, confidence, nms_threshold):
    num_boxes = decoded_bboxes.shape[0]
    keep = [ True ] * num_boxes
    for box1_id in range(0, num_boxes-1):
        for box2_id in range(box1_id+1, num_boxes):
            iou_val = iou(decoded_bboxes[box1_id], decoded_bboxes[box2_id]) 
            if iou_val > nms_threshold:
                assert box1_id != box2_id
                if confidence[box1_id] < confidence[box2_id]:
                    keep[box1_id] = False
                else:
                    keep[box2_id] = False

    num_keep = np.count_nonzero(keep)
    new_decoded_bboxes = np.zeros((num_keep, 4), dtype=np.float32)
    new_confidence     = np.zeros((num_keep,), dtype=np.float32)
    new_class_num      = np.zeros((num_keep,), dtype=np.float32)
    idx = 0
    for i in range(num_boxes):
        if keep[i] == True:
            new_decoded_bboxes[idx] = decoded_bboxes[i]
            new_confidence[idx]     = confidence[i]
            new_class_num[idx]      = class_num[i]
            idx += 1

    return new_decoded_bboxes, new_confidence, new_class_num 


# Pick prior_boxes which has higher class confidence data than threshold (conf>threshold)
#  - Calculate class-by-class confidence for each pboxes with softmax
#  - Pick only pboxes which have higher confidence than threshold
def screen_out_prior_boxes(confidence, class_num, box_logits_,  proposals_p, proposals_v, confidence_threshold):
    num_prior_boxes   = box_logits_.shape[0]
    prior_box_size    = proposals_p.shape[1]

    new_box_logits  = np.array([], dtype=np.float32)
    new_proposals_p = np.array([], dtype=np.float32)
    new_proposals_v = np.array([], dtype=np.float32)
    new_class_num   = np.array([], dtype=np.float32)
    new_confidence  = np.array([], dtype=np.float32)

    for pbox_idx in range(num_prior_boxes):
        clsid = class_num[pbox_idx]
        conf = confidence[pbox_idx]
        if conf > confidence_threshold:
            if clsid != 0:          # reject background
                new_box_logits  = np.append(new_box_logits, box_logits_[pbox_idx, :])
                new_proposals_p = np.append(new_proposals_p, proposals_p[pbox_idx, :])
                new_proposals_v = np.append(new_proposals_v, proposals_v[pbox_idx, :])
                new_confidence  = np.append(new_confidence, conf)
                new_class_num   = np.append(new_class_num, clsid)

    new_box_logits  = new_box_logits.reshape((-1,4))
    new_proposals_p = new_proposals_p.reshape((-1, prior_box_size))
    new_proposals_v = new_proposals_v.reshape((-1, prior_box_size))

    return new_confidence, new_class_num, new_box_logits, new_proposals_p, new_proposals_v


def decode_bboxes(box_logits_, proposals_p, proposals_v, num_prior_boxes, n, num_loc_classes, offset, normalized, input_width, input_height, code_type, variance_encoded_in_target, clip_before_nms):
    decoded_bboxes = np.zeros((num_prior_boxes, 4), dtype=np.float32)

    for pbox_idx in range(num_prior_boxes):
        prior_xmin = proposals_p[pbox_idx, 0 + offset]
        prior_ymin = proposals_p[pbox_idx, 1 + offset]
        prior_xmax = proposals_p[pbox_idx, 2 + offset]
        prior_ymax = proposals_p[pbox_idx, 3 + offset]
        loc_xmin = box_logits_[pbox_idx, 0]
        loc_ymin = box_logits_[pbox_idx, 1]
        loc_xmax = box_logits_[pbox_idx, 2]
        loc_ymax = box_logits_[pbox_idx, 3]
        if normalized == False:
            prior_xmin /= input_width
            prior_ymin /= input_height
            prior_xmax /= input_width
            prior_ymax /= input_height
        if code_type=='caffe.PriorBoxParameter.CORNER':
            if variance_encoded_in_target == True:
                new_xmin = prior_xmin + loc_xmin
                new_ymin = prior_ymin + loc_ymin
                new_xmax = prior_xmax + loc_xmax
                new_ymax = prior_ymax + loc_ymax
            else:
                new_xmin = prior_xmin + proposals_v[pbox_idx, 0] * loc_xmin
                new_ymin = prior_ymin + proposals_v[pbox_idx, 1] * loc_ymin
                new_xmax = prior_xmax + proposals_v[pbox_idx, 2] * loc_xmax
                new_ymax = prior_ymax + proposals_v[pbox_idx, 3] * loc_ymax
        elif code_type =='caffe.PriorBoxParameter.CENTER_SIZE':
            prior_width  = prior_xmax - prior_xmin
            prior_height = prior_ymax - prior_ymin
            prior_cx     = (prior_xmin + prior_xmax) / 2
            prior_cy     = (prior_ymin + prior_ymax) / 2
            if variance_encoded_in_target == True:
                decode_bbox_cx = loc_xmin * prior_width  + prior_cx
                decode_bbox_cy = loc_ymin * prior_height + prior_cy
                decode_bbox_width  = math.exp(loc_xmax) * prior_width
                decode_bbox_height = math.exp(loc_ymax) * prior_height
            else:
                decode_bbox_cx = proposals_v[pbox_idx, 0] * loc_xmin * prior_width  + prior_cx
                decode_bbox_cy = proposals_v[pbox_idx, 1] * loc_ymin * prior_height + prior_cy
                decode_bbox_width  = math.exp(proposals_v[pbox_idx, 2] * loc_xmax) * prior_width
                decode_bbox_height = math.exp(proposals_v[pbox_idx, 3] * loc_ymax) * prior_height
            
            new_xmin = decode_bbox_cx - decode_bbox_width  / 2
            new_ymin = decode_bbox_cy - decode_bbox_height / 2
            new_xmax = decode_bbox_cx + decode_bbox_width  / 2
            new_ymax = decode_bbox_cy + decode_bbox_height / 2
        
        decoded_bboxes[pbox_idx, 0] = new_xmin
        decoded_bboxes[pbox_idx, 1] = new_ymin
        decoded_bboxes[pbox_idx, 2] = new_xmax
        decoded_bboxes[pbox_idx, 3] = new_ymax

    return decoded_bboxes


def clip_bounding_boxes(bboxes):
    for bbox in bboxes:
        bbox[0] = max(0, min(1, bbox[0]))
        bbox[1] = max(0, min(1, bbox[1]))
        bbox[2] = max(0, min(1, bbox[2]))
        bbox[3] = max(0, min(1, bbox[3]))


def kernel_DetectionOutput_naive(inputs, num_classes, background_label_id, top_k, variance_encoded_in_target, keep_top_k, code_type, share_location,
                                        nms_threshold, confidence_threshold, clip_after_nms, clip_before_nms, decrease_label_id, normalized,
                                        input_height, input_width, objectness_score):

    box_logits = inputs[0]  #    [ N, num_prior_boxes * num_loc_classes * 4 ]               (1,7668)    = (1, 1917*1*4)
    class_pred = inputs[1]  #    [ N, num_prior_boxes * num_classes]                        (1, 174447) = (1, 1917*91 )
    proposals  = inputs[2]  #    [ priors_batch_size, 1, num_prior_boxes * prior_box_size ] 
                            # or [ priors_batch_size, 2, num_prior_boxes * prior_box_size ] (1, 2, 7668) = (1, 2, 1917*4)  box_proposals[ cx, cy, w, h ]
                            #  Size of the second dimension depends on variance_encoded_in_target. If variance_encoded_in_target is equal to 0, 
                            # the second dimension equals to 2 and variance values are provided for each boxes coordinates. If variance_encoded_in_target is 
                            # equal to 1, the second dimension equals to 1 and this tensor contains proposals boxes only.
                            #  prior_box_size is equal to 4 when normalized is set to 1 or it’s equal to 5 otherwise.

    assert proposals.shape[1] == 2
                                                                         # memo for debug
    N = box_logits.shape[0]                                              # 1
    priors_batch_size = proposals.shape[0]                               # 1
    prior_box_size = 4 if normalized == True else 5                      # 4
    num_prior_boxes = int(proposals.shape[2] / prior_box_size)           # 1917
    num_loc_classes = num_classes if share_location==False else 1        # 1
    proposals_2nd_dim = 2 if variance_encoded_in_target == False else 1  # 2
    offset = 0 if normalized == True else 1

    assert N == 1
    assert num_loc_classes == 1

    box_logits_  = box_logits.reshape((num_prior_boxes, 4))
    class_pred_  = class_pred.reshape((num_prior_boxes, num_classes))
    proposals_p  = proposals[:,0,:].reshape((num_prior_boxes, prior_box_size))
    proposals_v  = proposals[:,1,:].reshape((num_prior_boxes, prior_box_size))

    # Calculate the best confidence value and class id for each prior box
    class_num = np.zeros((num_prior_boxes,), dtype=np.float32)
    confidence = np.zeros((num_prior_boxes,), dtype=np.float32)
    for pbox_idx in range(num_prior_boxes):
        pred = class_pred_[pbox_idx,:]
        #softmax = np.exp(pred)/np.sum(np.exp(pred))
        m = np.argsort(pred)[::-1]
        class_num[pbox_idx] = m[0]          # class id
        confidence[pbox_idx] = pred[m[0]]   # confidence value

    # Reject low confidence prior boxes
    result = screen_out_prior_boxes(confidence, class_num, box_logits_, proposals_p, proposals_v, confidence_threshold)
    confidence, class_num, new_box_logits, new_proposals_p, new_proposals_v = result
    new_num_prior_boxes = len(new_box_logits)

    # Calculate bounding box coordinates (0.0-1.0, normalized) from prior boxes and box logits
    decoded_bboxes = decode_bboxes(new_box_logits, new_proposals_p, new_proposals_v, new_num_prior_boxes, 0, num_loc_classes, offset, normalized,
                                    input_width, input_height, code_type, variance_encoded_in_target, clip_before_nms)

    if clip_before_nms == True:
        clip_bounding_boxes(decoded_bboxes)

    if decrease_label_id == True:
        #result = nms_mxnet(decoded_bboxes, class_num, confidence, nms_threshold)  # MxNet style
        result = nms(decoded_bboxes, class_num, confidence, nms_threshold)
        decoded_bboxes, confidence, class_num = result
    else:
        #result = nms_caffe(decoded_bboxes, class_num, confidence, nms_threshold)  # Caffe style
        result = nms(decoded_bboxes, class_num, confidence, nms_threshold)
        decoded_bboxes, confidence, class_num = result

    if clip_after_nms == True:
        clip_bounding_boxes(decoded_bboxes)

    assert normalized == True
    #if normalized == False:
    #    normalize_boxes(input_height, input_width)

    # Calculate output shape
    output_shape = (1, 1, N * num_classes * num_prior_boxes, 7)
    if keep_top_k[0] > 0:
        output_shape = (1, 1, N * keep_top_k[0], 7)
    elif keep_top_k[0] == -1:
        if top_k > 0:
            output_shape = (1, 1, N * top_k * num_classes, 7)

    # Construct the final result [[idx, clsid, conf, xmin, ymin, xmax, ymax], ...]
    res = np.zeros(output_shape, dtype=np.float32)
    sorted_idx = np.argsort(confidence)[::-1]    # High -> Low order
    num_bboxes = len(decoded_bboxes)
    max_record = res.shape[2]
    for n in range(min(max_record, num_bboxes)):
        idx = sorted_idx[n]
        class_id = class_num[idx]
        conf     = confidence[idx]
        xmin     = decoded_bboxes[idx][0]
        ymin     = decoded_bboxes[idx][1]
        xmax     = decoded_bboxes[idx][2]
        ymax     = decoded_bboxes[idx][3]
        record = np.array([n, class_id, conf, xmin, ymin, xmax, ymax], dtype=np.float32)
        res[0, 0, n, :] = record

    # Add record terminator
    if num_bboxes < max_record:
        record = np.array([-1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        res[0, 0, num_bboxes, :] = record

    return res

# 'input': {0: {'precision': 'FP16', 'dims': (1, 7668)},         # Box logits [ N, num_prior_boxes * num_loc_classes * 4 ]
#           1: {'precision': 'FP16', 'dims': (1, 174447)},       # class predictions [N, num_prior_boxes * prior_box_size ]
#           2: {'precision': 'FP32', 'dims': (1, 2, 7668)}},     # proposals [ priors_batch_size, 1, num_prior_boxes * prior_box_size] or [priors_batch_size, 2, num_prior_boxes * prior_box_size ]


def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
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
