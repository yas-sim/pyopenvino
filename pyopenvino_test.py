# lesser-openvino.py test program :-)

import cv2
import numpy as np
from pyopenvino.inference_engine import IECore

ie = IECore()
net = ie.read_network('mnist.xml', 'mnist.bin')
exenet = net.load_network(net, 'CPU')

cv2img = cv2.imread('mnist2.png')
inblob = cv2.split(cv2img)[0]
inblob = inblob.reshape(1,1,28,28).astype(np.float32)

res = exenet.infer({'conv2d_input':inblob})

m = np.argsort(res['Func/StatefulPartitionedCall/output/_11:0'][0])[::-1]

print(m)
print(res)
