# pyOpenVINO Integrity Test Program

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
# ssd_mobilenet_v1_coco

def test_ssd_mobilenet_v1():
    cv2img = cv2.imread('resources/guinea-pig.jpg')

    # Display read image
    disp_img = cv2.resize(cv2img, (300, 300))
    cv2.imshow('input image', disp_img)
    cv2.waitKey(1*1000)
    cv2.destroyAllWindows()

    inblob = cv2.resize(cv2img, (300,300))
    inblob = inblob.transpose((2,0,1))
    inblob = inblob.reshape(1,3,300,300).astype(np.float32)

    model = 'models/ssd_mobilenet_v1_coco'

    for kernel_type in kernel_types:
        res = run_test(model, inblob, kernel_type, niter=1)

        expected = (0.0, 16.0, 0.7186840772628784, 0.032441407442092896, 0.40934476256370544, 0.890156626701355, 0.9684370756149292)
        record = res.reshape(100,7)[0] # Checks only the 1st object
        n, class_id, conf, xmin, ymin, xmax, ymax = record
        if conf>0.5:
            canvas = disp_img.copy()
            img_h, img_w = canvas.shape[:1+1]
            x0 = int(xmin * img_w)
            y0 = int(ymin * img_h)
            x1 = int(xmax * img_w)
            y1 = int(ymax * img_h)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (255,255,0), 2)
            cv2.putText(canvas, str(int(class_id)), (x0, y0), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 2)
            compare = np.allclose(expected, record, rtol=0.01)
            print('Expected: ({}, {}, {}, ({}, {}), ({}, {}))'.format(*expected))
            print('Result  : ({}, {}, {}, ({}, {}), ({}, {}))'.format(*record))
            print('Compare :', 'OK' if compare else 'UNMATCH')
            cv2.imshow('result', canvas)
            cv2.waitKey(3*1000)
            cv2.destroyWindow('result')
            assert compare
        else:
            print('ERROR: No objects found.')
            assert False
        print()
    cv2.destroyAllWindows()

#---------------------------------------------------------------

kernel_types = [ 'special', 'numpy' ]
kernel_types += [ 'naive' ]

test_mnist()
test_mnist_bn()
test_googlenet_v1()
test_ssd_mobilenet_v1()

print('Integrity test completed')
