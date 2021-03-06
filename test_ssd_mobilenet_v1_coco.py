# pyOpenVINO test program :-)

import time

import cv2
import numpy as np
from pyopenvino.inference_engine import IECore

model = 'models/ssd_mobilenet_v1_coco'

ie = IECore()                                        # Create core object
net = ie.read_network(model+'.xml', model+'.bin')    # Read model file
print('inputs:', net.inputs)
print('outputs:', net.outputs)
output_node_name = net.outputs[0]['name']
exenet = ie.load_network(net, 'CPU')                 # Unlike genuine OpenVINO, pyOpenVINO doesn't have device plugins. This function just schedules tasks for runtime)

# Read an image file to infer
cv2img = cv2.imread('resources/guinea-pig.jpg')
inblob = cv2.resize(cv2img, (300,300))
inblob = inblob.transpose((2,0,1))
inblob = inblob.reshape(1,3,300,300).astype(np.float32)

# Display read image
cv2img = cv2.resize(cv2img, (300,300))
cv2.imshow('input image', cv2img)
cv2.waitKey(1*1000)
cv2.destroyAllWindows()

exenet.kernel_type = 'special'    # Set kernel implementation type ('naive', 'numpy' or 'special')
atime = 0
nitr = 1

print('Kernel type:', exenet.kernel_type)
for i in range(nitr):
    stime = time.time()
    res = exenet.infer({'image_tensor':inblob}, verbose=True)    # Run inference
    etime = time.time()
    atime += etime-stime

print(atime/nitr, 'sec/inf')

# Decode and display SSD result
img_h, img_w = cv2img.shape[:1+1]
for record in res['detection_boxes:0'].reshape(100,7):
    n, class_id, conf, xmin, ymin, xmax, ymax = record
    if conf>0.5:
        x0 = int(xmin * img_w)
        y0 = int(ymin * img_h)
        x1 = int(xmax * img_w)
        y1 = int(ymax * img_h)
        cv2.rectangle(cv2img, (x0, y0), (x1, y1), (255,255,0), 2)
        cv2.putText(cv2img, str(int(class_id)), (x0, y0), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 2)
        print('({}, {}, {}, ({}, {}), ({}, {}))'.format(n, class_id, conf, xmin, ymin, xmax, ymax))

cv2.imshow('result', cv2img)
cv2.waitKey(0)
