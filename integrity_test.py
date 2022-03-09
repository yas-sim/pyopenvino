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

kernel_types.remove('special')   # Currently, 'special' Convolution kernel with uneven padding doesn't work properly
test_ssd_mobilenet_v1()

print('Integrity test completed')


# Expected result
"""
model: models/mnist
inputs: [{'name': 'conv2d_input', 'type': 'Parameter', 'version': 'opset1', 'data': {'element_type': 'f32', 'shape': (1, 1, 28, 28)}, 'output': {0: {'precision': 'FP32', 'dims': (1, 1, 28, 28)}}}]
outputs: [{'name': 'Func/StatefulPartitionedCall/output/_11:0', 'type': 'Result', 'version': 'opset1', 'input': {0: {'precision': 'FP32', 'dims': (1, 10)}}}]
Kernel type: special
0.008022308349609375 sec/inf
Result: [2 0 1 7 8 6 3 4 5 9]

model: models/mnist
inputs: [{'name': 'conv2d_input', 'type': 'Parameter', 'version': 'opset1', 'data': {'element_type': 'f32', 'shape': (1, 1, 28, 28)}, 'output': {0: {'precision': 'FP32', 'dims': (1, 1, 28, 28)}}}]
outputs: [{'name': 'Func/StatefulPartitionedCall/output/_11:0', 'type': 'Result', 'version': 'opset1', 'input': {0: {'precision': 'FP32', 'dims': (1, 10)}}}]
Kernel type: numpy
0.1832447052001953 sec/inf
Result: [2 0 1 7 8 6 3 4 5 9]

model: models/mnist
inputs: [{'name': 'conv2d_input', 'type': 'Parameter', 'version': 'opset1', 'data': {'element_type': 'f32', 'shape': (1, 1, 28, 28)}, 'output': {0: {'precision': 'FP32', 'dims': (1, 1, 28, 28)}}}]
outputs: [{'name': 'Func/StatefulPartitionedCall/output/_11:0', 'type': 'Result', 'version': 'opset1', 'input': {0: {'precision': 'FP32', 'dims': (1, 10)}}}]
Kernel type: naive
2.470170259475708 sec/inf
Result: [2 0 1 7 8 6 3 4 5 9]

model: models/mnist_bn
inputs: [{'name': 'conv2d_input', 'type': 'Parameter', 'version': 'opset1', 'data': {'element_type': 'f32', 'shape': (1, 1, 28, 28)}, 'output': {0: {'precision': 'FP32', 'dims': (1, 1, 28, 28)}}}]
outputs: [{'name': 'Func/StatefulPartitionedCall/output/_23:0', 'type': 'Result', 'version': 'opset1', 'input': {0: {'precision': 'FP32', 'dims': (1, 10)}}}]
Kernel type: special
0.11962580680847168 sec/inf
Result: [2 8 7 0 3 1 5 9 6 4]

model: models/mnist_bn
inputs: [{'name': 'conv2d_input', 'type': 'Parameter', 'version': 'opset1', 'data': {'element_type': 'f32', 'shape': (1, 1, 28, 28)}, 'output': {0: {'precision': 'FP32', 'dims': (1, 1, 28, 28)}}}]
outputs: [{'name': 'Func/StatefulPartitionedCall/output/_23:0', 'type': 'Result', 'version': 'opset1', 'input': {0: {'precision': 'FP32', 'dims': (1, 10)}}}]
Kernel type: numpy
1.1129910945892334 sec/inf
Result: [2 8 7 0 3 1 5 9 6 4]

model: models/mnist_bn
inputs: [{'name': 'conv2d_input', 'type': 'Parameter', 'version': 'opset1', 'data': {'element_type': 'f32', 'shape': (1, 1, 28, 28)}, 'output': {0: {'precision': 'FP32', 'dims': (1, 1, 28, 28)}}}]
outputs: [{'name': 'Func/StatefulPartitionedCall/output/_23:0', 'type': 'Result', 'version': 'opset1', 'input': {0: {'precision': 'FP32', 'dims': (1, 10)}}}]
Kernel type: naive
66.6951036453247 sec/inf
Result: [2 8 7 0 3 1 5 9 6 4]

model: models/googlenet-v1
inputs: [{'name': 'data', 'type': 'Parameter', 'version': 'opset1', 'data': {'element_type': 'f32', 'shape': (1, 3, 224, 224)}, 'output': {0: {'precision': 'FP32', 'dims': (1, 3, 224, 224)}}}]
outputs: [{'name': 'prob/sink_port_0', 'type': 'Result', 'version': 'opset1', 'input': {0: {'precision': 'FP32', 'dims': (1, 1000)}}}]
Kernel type: special
0.46231722831726074 sec/inf
Result: [338 359 358 356 362 333 377 215 357 336]

model: models/googlenet-v1
inputs: [{'name': 'data', 'type': 'Parameter', 'version': 'opset1', 'data': {'element_type': 'f32', 'shape': (1, 3, 224, 224)}, 'output': {0: {'precision': 'FP32', 'dims': (1, 3, 224, 224)}}}]
outputs: [{'name': 'prob/sink_port_0', 'type': 'Result', 'version': 'opset1', 'input': {0: {'precision': 'FP32', 'dims': (1, 1000)}}}]
Kernel type: numpy
21.25524115562439 sec/inf
Result: [338 359 358 356 362 333 377 215 357 336]

model: models/googlenet-v1
inputs: [{'name': 'data', 'type': 'Parameter', 'version': 'opset1', 'data': {'element_type': 'f32', 'shape': (1, 3, 224, 224)}, 'output': {0: {'precision': 'FP32', 'dims': (1, 3, 224, 224)}}}]
outputs: [{'name': 'prob/sink_port_0', 'type': 'Result', 'version': 'opset1', 'input': {0: {'precision': 'FP32', 'dims': (1, 1000)}}}]
Kernel type: naive
1494.766568183899 sec/inf
Result: [338 358 359 356 362 333 357 377 336 215]

model: models/ssd_mobilenet_v1_coco
inputs: [{'name': 'image_tensor', 'type': 'Parameter', 'version': 'opset1', 'data': {'element_type': 'f32', 'shape': (1, 3, 300, 300)}, 'output': {0: {'precision': 'FP32', 'dims': (1, 3, 300, 300)}}}]
outputs: [{'name': 'detection_boxes:0', 'type': 'Result', 'version': 'opset1', 'input': {0: {'precision': 'FP32', 'dims': (1, 1, 100, 7)}}}]
Kernel type: numpy
49.93236422538757 sec/inf
Expected: (0.0, 16.0, 0.7186840772628784, (0.032441407442092896, 0.40934476256370544), (0.890156626701355, 0.9684370756149292))
Result  : (0.0, 16.0, 0.7186850309371948, (0.032441481947898865, 0.40934476256370544), (0.8901565074920654, 0.9684370756149292))
Compare : OK

model: models/ssd_mobilenet_v1_coco
inputs: [{'name': 'image_tensor', 'type': 'Parameter', 'version': 'opset1', 'data': {'element_type': 'f32', 'shape': (1, 3, 300, 300)}, 'output': {0: {'precision': 'FP32', 'dims': (1, 3, 300, 300)}}}]
outputs: [{'name': 'detection_boxes:0', 'type': 'Result', 'version': 'opset1', 'input': {0: {'precision': 'FP32', 'dims': (1, 1, 100, 7)}}}]
Kernel type: naive
1374.4146270751953 sec/inf
Expected: (0.0, 16.0, 0.7186840772628784, (0.032441407442092896, 0.40934476256370544), (0.890156626701355, 0.9684370756149292))
Result  : (0.0, 16.0, 0.7186837196350098, (0.032441459596157074, 0.40934476256370544), (0.8901565074920654, 0.9684370160102844))
Compare : OK

Integrity test completed
"""
