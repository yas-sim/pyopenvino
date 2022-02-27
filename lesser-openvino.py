# lesser OpenVINO

import sys, os
import struct
import pickle

import glob
import importlib

import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et

import common_def


class plugins:
    def __init__(self):
        self.plugins={}

    def import_plugin(self, plugin_path, file_path, plugin_name=None):
        path, fname = os.path.split(file_path)
        bname, ext = os.path.splitext(fname)
        if plugin_name is None:
            plugin_name = bname
        plugin_path = plugin_path + '.' + bname
        module = importlib.import_module(plugin_path)
        setattr(self, plugin_name, module)
        self.plugins[plugin_name] = module

    def import_plugins(self, plugin_path:str):
        plugins = glob.glob(os.path.join(plugin_path, '**', '*.py'), recursive=True)
        for plugin in plugins:
            self.import_plugin(plugin_path, plugin)

def read_IR_Model(model):
    bname, ext = os.path.splitext(model)
    xmlFile = bname + '.xml'
    binFile = bname + '.bin'
    if not os.path.isfile(xmlFile) or not os.path.isfile(binFile):
        print('model {} is not found'.format(model))
        return None, None

    xml = et.parse(bname+'.xml')

    with open(bname+'.bin', 'rb') as f:
        bin = f.read()

    return xml, bin

def parse_IR_XML(xml:et.ElementTree):

    root = xml.getroot()
    if root.tag != 'net':
        print('not an OpenVINO IR file')
        return -1

    dict_layers = {}
    layers = root.findall('./layers/layer')
    for layer in layers:
        dict_layer = {}
        layer_id = layer.attrib['id']
        dict_layer = { key:val for key, val in layer.attrib.items() if key!='id' }
        data = layer.find('data')
        if data is not None:
            dict_layer['data'] = data.attrib
            if 'shape' in dict_layer['data']:
                dict_layer['data']['shape'] = common_def.string_to_tuple(dict_layer['data']['shape'])
            if 'stride' in dict_layer['data']:
                dict_layer['data']['stride'] = common_def.string_to_tuple(dict_layer['data']['stride'])

        input = layer.find('input')
        if input is not None:
            dict_layer['input'] = {}
            for port in input.findall('port'):
                port_id = port.attrib['id']
                port_prec = port.attrib['precision']
                dims = port.findall('./dim')
                list_dims = []
                for dim in dims:
                    list_dims.append(int(dim.text))
                dict_layer['input'][int(port_id)] = { 'precision': port_prec, 'dims':tuple(list_dims) }

        output = layer.find('output')
        if output is not None:
            dict_layer['output'] = {}
            for port in output.findall('port'):
                port_id = port.attrib['id']
                port_prec = port.attrib['precision']
                dims = port.findall('./dim')
                list_dims = []
                for dim in dims:
                    list_dims.append(int(dim.text))
                dict_layer['output'][int(port_id)] = { 'precision': port_prec, 'dims':tuple(list_dims) }
        dict_layers[int(layer_id)] = dict_layer

    list_edges = []
    edges = root.findall('./edges/edge')
    for edge in edges:
        attr = edge.attrib
        list_edges.append((int(attr['from-layer']), int(attr['from-port']), int(attr['to-layer']), int(attr['to-port'])))

    return dict_layers, list_edges

def build_graph(ir_layers:dict, ir_edges:list):
    G = nx.DiGraph()
    for node_id, node_info in ir_layers.items():
        G.add_node(node_id)
        for key, val in node_info.items():
            G.nodes[node_id][key] = val
    for edge in ir_edges:  # edge = (from-layer,from-port, to-layer, to-port)
        G.add_edge(edge[0], edge[2])
        G.edges[(edge[0], edge[2])]['connection'] = edge
        #nx.set_edge_attributes(G, values={(edge[0], edge[2]):{'connection':edge}})
    if not nx.is_directed_acyclic_graph(G):
        print('Graph is not an directed acyclic graph')
        return None
    return G


def set_constants_to_graph(G, bin:bytes):
    consts = find_node_by_type(G, 'Const')
    for const in consts:
        node = G.nodes[const[0]]
        data = node['data']
        offset    = int(data['offset'])
        size      = int(data['size'])
        precision = data['element_type'].upper()
        blobBin = bin[offset:offset+size]                       # cut out the weight for this blob from the weight buffer
        formatstring = '<' + common_def.format_config[precision][0] * (len(blobBin)//common_def.format_config[precision][1])
        decoded_data = struct.unpack(formatstring, blobBin)     # decode the buffer
        node['const'] = { 'data':decoded_data, 'element_info':precision, 'size':size, 'decode_info':common_def.format_config[precision] }

def find_node_by_type(G:nx.DiGraph, type:str):
    results = []
    for node in G.nodes():
        if G.nodes[node]['type'] == type:
            results.append((node, G.nodes[node]['name']))
    return results

def schedule_tasks(G):
    def search_predecessors(G, node_ids):
        for node_id in node_ids:
            predecessors = list(G.pred[node_id])
            search_predecessors(G, predecessors)
            if node_id not in task_list:
                task_list.append(node_id)
    outputs = find_node_by_type(G, 'Result')
    task_list = []
    outputs = [ key for key,_ in outputs]
    search_predecessors(G, outputs)
    return task_list

def prepare_inputs(task:str, G:nx.DiGraph):
    predecessors = list(G.pred[task])
    inputs = {}
    for predecessor in predecessors:
        edge = G.edges[(predecessor, task)]  # edge = { 'connection':(from-layer, from-port, to-layer, to-port)}
        connection = edge['connection']
        source_node = G.nodes[connection[0]]
        source_port = connection[1]
        data = source_node['output'][source_port]['data']
        sink_port = connection[3]
        inputs[sink_port] = data
    return inputs

def compare_results(node_name:str, result:np.array, GT:dict):
    print('{} : '.format(node_name), end='')
    if node_name not in GT:
        print('skip')
        return
    GT_data = GT[node_name][2]
    print('{} / {}'.format(result.shape, GT_data.shape), end='')
    if np.allclose(result, GT_data, rtol=0.001):
        print(' OOO match')
    else:
        print(' XXX unmatch')

def run_tasks(task_list:list, G:nx.DiGraph, p:plugins):
    for task in task_list:      # task_list = [ 0, 1, 2, ... ]  Numbers are the node_id
        node = G.nodes[task]
        node_type = node['type']
        node_name = G.nodes[task]['name']
        inputs = {}
        if 'input' in node:     # Prepare input data for computation
            inputs = prepare_inputs(task, G)

        res = p.plugins[node_type].compute(node, inputs, debug=False)

        # Set computation result to output ports
        if len(res)>0:
            for port_id, data in res.items():
                G.nodes[task]['output'][port_id]['data'] = data
                #print(G.nodes[task]['output'])
                compare_results(node_name, data, fmap)
                if 'StatefulPartitionedCall/sequential/conv2d_1/Conv2D' == node_name:
                    disp_result(data)

def run_infer(inputs:dict, task_list:list, G:nx.DiGraph, p:plugins):
    # Set input data for inference
    for node_name, val in inputs.items():
        for node in G.nodes:
            if G.nodes[node]['name'] == node_name:
                G.nodes[node]['param'] = val

    run_tasks(task_list, G, p)

    outputs = find_node_by_type(G, 'Result')
    res = {}
    for output in outputs:
        node_id = output[0]
        result = G.nodes[node_id]['result']
        node_name = G.nodes[node_id]['name']
        res[node_name] = result

    return res

'''
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(-1, 1, 28, 28, 1)
test_images = test_images.reshape(-1, 1, 28, 28, 1)
with open('mnist-train.pickle', 'wb') as f:
    pickle.dump(train_images, f)
with open('mnist-test-image.pickle', 'wb') as f:
    pickle.dump(test_images, f)
'''
with open('mnist-test-image.pickle', 'rb') as f:
    test_images = pickle.load(f)
#print(test_images.shape)
'''
img = test_images[0]
cv2img = img.reshape(1,28,28).transpose((1,2,0)).astype(np.uint8)
cv2.imshow('image', cv2img)
cv2.waitKey(0)
#cv2img = cv2.merge([cv2img, cv2img, cv2img])
cv2.imwrite('mnist.png', cv2img)
'''

def disp_result(data):
    N,C,H,W = data.shape
    for c in range(C):
        print('C=', c)
        for h in range(H):
            for w in range(W):
                print('{:6.3f},'.format(data[0,c,h,w]), end='')
            print()

with open('mnist_featmap.pickle', 'rb') as f:
    fmap = pickle.load(f)


def dump_graph(G:nx.DiGraph):
    for node_id, node_contents in G.nodes.items():
        print('node id=', node_id)
        print_dict(node_contents)
    for edge_id, edge_contents in G.edges.items():
        print('edge_id=', edge_id)
        print(' '*2, edge_contents)


def main():
    xml, bin = read_IR_Model('mnist.xml')
    if xml is None or bin is None:
        print('failed to read model file')
        return -1

    layers, edges = parse_IR_XML(xml)
    #print(layers, edges)

    G=build_graph(layers, edges)
    #dump_graph(G)
    #find_node_by_type(G, 'Parameter')
    #find_node_by_type(G, 'Const')
    #find_node_by_type(G, 'Result')
    set_constants_to_graph(G, bin)
    task_list = schedule_tasks(G)
    #print(task_list)

    p = plugins()
    p.import_plugins('plugins')

    inblob = test_images[0]
    res = run_infer({'conv2d_input':inblob}, task_list, G, p)

    print(res)

if __name__ == '__main__':
    sys.exit(main())

'''
conv2d_input
StatefulPartitionedCall/sequential/conv2d/Conv2D
StatefulPartitionedCall/sequential/conv2d/BiasAdd/Add
StatefulPartitionedCall/sequential/conv2d/Relu
StatefulPartitionedCall/sequential/max_pooling2d/MaxPool
StatefulPartitionedCall/sequential/conv2d_1/Conv2D
StatefulPartitionedCall/sequential/conv2d_1/BiasAdd/Add
StatefulPartitionedCall/sequential/conv2d_1/Relu
StatefulPartitionedCall/sequential/max_pooling2d_1/MaxPool
StatefulPartitionedCall/sequential/target_conv_layer/Conv2D
StatefulPartitionedCall/sequential/target_conv_layer/BiasAdd/Add
StatefulPartitionedCall/sequential/target_conv_layer/Relu
StatefulPartitionedCall/sequential/target_conv_layer/Relu/Transpose
StatefulPartitionedCall/sequential/flatten/Reshape
StatefulPartitionedCall/sequential/dense/MatMul
StatefulPartitionedCall/sequential/dense/BiasAdd/Add
StatefulPartitionedCall/sequential/dense/Relu
StatefulPartitionedCall/sequential/dense_1/MatMul
StatefulPartitionedCall/sequential/dense_1/BiasAdd/Add
'''