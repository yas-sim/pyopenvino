# lesser-openvino.py test program :-)

import cv2
import numpy as np
from pyopenvino.inference_engine import IECore

import time

ie = IECore()
net = ie.read_network('models/mnist.xml', 'models/mnist.bin')
exenet = net.load_network(net, 'CPU')

cv2img = cv2.imread('resources/mnist2.png')
inblob = cv2.split(cv2img)[0]
inblob = inblob.reshape(1,1,28,28).astype(np.float32)

atime = 0
nitr = 1
for i in range(nitr):
    stime = time.time()
    res = exenet.infer({'conv2d_input':inblob})
    etime = time.time()
    atime += etime-stime

print(atime/nitr, 'sec/inf')

m = np.argsort(res['Func/StatefulPartitionedCall/output/_11:0'][0])[::-1]

print(m)
print(res)