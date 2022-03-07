# pyOpenVINO test program :-)

import time

import cv2
import numpy as np
from pyopenvino.inference_engine import IECore

def run_test(model:str, input_data, kernel_type, niter:int=1):
    ie = IECore()                                        # Create core object
    net = ie.read_network(model+'.xml', model+'.bin')    # Read model file
    print('model:', model)
    print('inputs:', net.inputs)
    print('outputs:', net.outputs)
    input_node_name = net.inputs[0]['name']
    output_node_name = net.outputs[0]['name']
    exenet = ie.load_network(net, 'CPU')                 # Unlike genuine OpenVINO, pyOpenVINO doesn't have device plugins. This function just schedules tasks for runtime)

    print('Kernel type:', kernel_type)
    exenet.kernel_type = kernel_type    # Set kernel implementation type ('naive', 'numpy' or 'special')

    atime = 0
    for i in range(niter):
        stime = time.time()
        res = exenet.infer({input_node_name:input_data}, verbose=False)    # Run inference
        etime = time.time()
        atime += etime-stime

    print(atime/niter, 'sec/inf')

    return res[output_node_name]



#---------------------------------------------------------------
# MNIST

def test_mnist():
    cv2img = cv2.imread('resources/mnist2.png')

    # Display read image
    disp_img = cv2.resize(cv2img, (0, 0), fx=4, fy=4)
    cv2.imshow('input image', disp_img)
    cv2.waitKey(1*1000)
    cv2.destroyAllWindows()

    # Read an image file to infer
    cv2img = cv2.imread('resources/mnist2.png')
    inblob = cv2.split(cv2img)[0]
    inblob = inblob.reshape(1,1,28,28).astype(np.float32)

    model = 'models/mnist'
    for kernel_type in kernel_types:
        res = run_test(model, inblob, kernel_type, niter=1)
        m = np.argsort(res[0])[::-1]    # Sort results
        print('Result:', m[:10])
        assert m[0]==2 and m[1]==0 and m[2]==1
        print()


#---------------------------------------------------------------
# MNIST_BN (BatchNorm)

def test_mnist_bn():
    cv2img = cv2.imread('resources/mnist2.png')

    # Display read image
    disp_img = cv2.resize(cv2img, (0, 0), fx=4, fy=4)
    cv2.imshow('input image', disp_img)
    cv2.waitKey(1*1000)
    cv2.destroyAllWindows()

    # Read an image file to infer
    cv2img = cv2.imread('resources/mnist2.png')
    inblob = cv2.split(cv2img)[0]
    inblob = inblob.reshape(1,1,28,28).astype(np.float32)

    model = 'models/mnist_bn'
    for kernel_type in kernel_types:
        res = run_test(model, inblob, kernel_type, niter=1)
        m = np.argsort(res[0])[::-1]    # Sort results
        print('Result:', m[:10])
        assert m[0]==2 and m[1]==8 and m[2]==7
        print()


#---------------------------------------------------------------
# Googlenet-v1

def test_googlenet_v1():
    cv2img = cv2.imread('resources/guinea-pig.jpg')

    # Display read image
    disp_img = cv2.resize(cv2img, (300, 300))
    cv2.imshow('input image', disp_img)
    cv2.waitKey(1*1000)
    cv2.destroyAllWindows()

    inblob = cv2.resize(cv2img, (224,224))
    inblob = inblob.transpose((2,0,1))
    inblob = inblob.reshape(1,3,224,224).astype(np.float32)

    model = 'models/googlenet-v1'
    for kernel_type in kernel_types:
        res = run_test(model, inblob, kernel_type, niter=1)
        m = np.argsort(res[0])[::-1]    # Sort results
        print('Result:', m[:10])
        assert m[0]==338 and 359 in m[:3] and 358 in m[:3]
        print()

#---------------------------------------------------------------

kernel_types = [ 'special', 'numpy' ]
kernel_types += [ 'naive' ]

test_mnist()
test_mnist_bn()
test_googlenet_v1()

print('Integrity test completed')
