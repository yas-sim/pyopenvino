# pyOpenVINO test program :-)

import time

import cv2
import numpy as np
from pyopenvino.inference_engine import IECore

model = 'models/googlenet-v1'

ie = IECore()                                        # Create core object
net = ie.read_network(model+'.xml', model+'.bin')    # Read model file
print('inputs:', net.inputs)
print('outputs:', net.outputs)
output_node_name = net.outputs[0]['name']
exenet = ie.load_network(net, 'CPU')                 # Unlike genuine OpenVINO, pyOpenVINO doesn't have device plugins. This function just schedules tasks for runtime)

# Read an image file to infer
cv2img = cv2.imread('resources/guinea-pig.jpg')
inblob = cv2.resize(cv2img, (224,224))
inblob = inblob.transpose((2,0,1))
inblob = inblob.reshape(1,3,224,224).astype(np.float32)

# Display read image
cv2img = cv2.resize(cv2img, (224,224))
cv2.imshow('input image', cv2img)
cv2.waitKey(1*1000)
cv2.destroyAllWindows()

exenet.kernel_type = 'special'    # Set kernel implementation type ('naive', 'numpy' or 'special')

atime = 0
nitr = 1

print('Kernel type:', exenet.kernel_type)
for i in range(nitr):
    stime = time.time()
    res = exenet.infer({'data':inblob}, verbose=True)    # Run inference
    etime = time.time()
    atime += etime-stime

print(atime/nitr, 'sec/inf')

#print('Raw result:', res)
m = np.argsort(res[output_node_name][0])[::-1]    # Sort results
print('Result:', m[:10])
