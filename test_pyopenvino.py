# pyOpenVINO test program :-)

import cv2
import numpy as np
from pyopenvino.inference_engine import IECore

import time

ie = IECore()
net = ie.read_network('models/mnist.xml', 'models/mnist.bin')
print('inputs:', net.inputs)
print('outputs:', net.outputs)
output_node_name = net.outputs[0]['name']
exenet = ie.load_network(net, 'CPU')

cv2img = cv2.imread('resources/mnist2.png')
inblob = cv2.split(cv2img)[0]
inblob = inblob.reshape(1,1,28,28).astype(np.float32)

cv2img = cv2.resize(cv2img, (0,0), fx=4.0, fy=4.0)
cv2.imshow('input image', cv2img)
cv2.waitKey(1*1000)
cv2.destroyAllWindows()

atime = 0
nitr = 1
exenet.kernel_type = 'numpy'    # Set kernel implementation type ('naive', 'numpy' or 'special')
for i in range(nitr):
    stime = time.time()
    res = exenet.infer({'conv2d_input':inblob}, verbose=True)
    etime = time.time()
    atime += etime-stime

print(atime/nitr, 'sec/inf')

m = np.argsort(res[output_node_name][0])[::-1]
print(m, res)
