# lesser OpenVINO

import sys, os
import struct
import pickle
import glob
import importlib
import argparse

if os.name == 'nt':
    import enable_escape_sequence_win

import cv2
import numpy as np
import networkx as nx
import xml.etree.ElementTree as et

import common_def

# -------------------------------------------------------------------------------------------------------

class IECore:
    def __init__(self):
        # Load ops plug-ins
        self.plugins = plugins()
        self.plugins.load_plugins('plugins')

    def read_network(self, xmlpath:str, binpath:str):
        net = CNNNetwork(self)
        net.read_IR_Model(xmlpath)
        if net.xml is None or net.bin is None:
            print('failed to read model file')
            return -1
        net.parse_IR_XML()
        self.G = net.build_graph()
        net.set_constants_to_graph()
        return net

# -------------------------------------------------------------------------------------------------------

class CNNNetwork:
    def __init__(self, iecore:IECore):
        self.ie = iecore
        self.xml = None
        self.bin = None
        self.layers = None
        self.edges = None

    def read_IR_Model(self, model):
        bname, ext = os.path.splitext(model)
        xmlFile = bname + '.xml'
        binFile = bname + '.bin'
        if not os.path.isfile(xmlFile) or not os.path.isfile(binFile):
            print('model {} is not found'.format(model))
            return

        self.xml = et.parse(bname+'.xml')

        with open(bname+'.bin', 'rb') as f:
            self.bin = f.read()


    # Parse IR XML and generate 'layer(dict)' and 'edge(list)' data structure
    def parse_IR_XML(self):
        root = self.xml.getroot()
        if root.tag != 'net':
            print('not an OpenVINO IR file')
            return

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

        self.layers = dict_layers
        self.edges = list_edges

    # Build directional acyclic graph (nx.DiGraph) using 'networkx' as the representation of the DL model
    def build_graph(self):
        self.G = nx.DiGraph()
        for node_id, node_info in self.layers.items():
            self.G.add_node(node_id)
            for key, val in node_info.items():
                self.G.nodes[node_id][key] = val
        for edge in self.edges:  # edge = (from-layer,from-port, to-layer, to-port)
            self.G.add_edge(edge[0], edge[2])
            self.G.edges[(edge[0], edge[2])]['connection'] = edge
        if not nx.is_directed_acyclic_graph(self.G):
            print('Graph is not an directed acyclic graph')
            return

    # Set 'Const' data from '.bin' file data
    # Cut out binary weight/bias/constant data, decode it, and store the result to the DiGraph nodes
    def set_constants_to_graph(self):
        consts = self.find_node_by_type('Const')
        for const in consts:
            node = self.G.nodes[const[0]]
            data = node['data']
            offset    = int(data['offset'])
            size      = int(data['size'])
            precision = data['element_type'].upper()
            blobBin = self.bin[offset:offset+size]                       # cut out the weight for this blob from the weight buffer
            formatstring = '<' + common_def.format_config[precision][0] * (len(blobBin)//common_def.format_config[precision][1])
            decoded_data = struct.unpack(formatstring, blobBin)          # decode the buffer
            node['const'] = { 'data':decoded_data, 'element_info':precision, 'size':size, 'decode_info':common_def.format_config[precision] }

    # Find a node from DiGraph
    def find_node_by_type(self, type:str):
        results = []
        for node in self.G.nodes():
            if self.G.nodes[node]['type'] == type:
                results.append((node, self.G.nodes[node]['name']))
        return results

    def load_network(self, net:nx.DiGraph, device:str='CPU', num_infer:int=1):
        exenet = Executable_Network(self)
        exenet.schedule_tasks()
        return exenet

# -------------------------------------------------------------------------------------------------------

class Executable_Network:
    def __init__(self, cnnnetwork:CNNNetwork):
        self.cnnnet = cnnnetwork

    def schedule_tasks(self):
        def search_predecessors(G, node_ids, task_list:list):
            for node_id in node_ids:
                predecessors = list(G.pred[node_id])
                search_predecessors(G, predecessors, task_list)
                if node_id not in task_list:
                    task_list.append(node_id)
        outputs = self.cnnnet.find_node_by_type('Result')
        self.task_list = []
        outputs = [ key for key,_ in outputs]
        search_predecessors(self.cnnnet.G, outputs, self.task_list)

    # Gather and prepare the input data which needed to run a task
    def prepare_inputs_for_task(self, task:str):
        predecessors = list(self.cnnnet.G.pred[task])
        inputs = {}
        for predecessor in predecessors:
            edge = self.cnnnet.G.edges[(predecessor, task)]  # edge = { 'connection':(from-layer, from-port, to-layer, to-port)}
            connection = edge['connection']
            source_node = self.cnnnet.G.nodes[connection[0]]
            source_port = connection[1]
            data = source_node['output'][source_port]['data']
            sink_port = connection[3]
            inputs[sink_port] = data
        return inputs

    # Run tasks in the order specified in the task_list
    def run_tasks(self):
        G = self.cnnnet.G
        p = self.cnnnet.ie.plugins
        for task in self.task_list:      # task_list = [ 0, 1, 2, ... ]  Numbers are the node_id
            node = G.nodes[task]
            node_type = node['type']
            node_name = G.nodes[task]['name']
            inputs = {}
            if 'input' in node:     # Prepare input data for computation
                inputs = self.prepare_inputs_for_task(task)

            res = p.plugins[node_type].compute(node, inputs, debug=False)  # Run a task (op)

            # Set computation result to output ports
            if len(res)>0:
                for port_id, data in res.items():
                    G.nodes[task]['output'][port_id]['data'] = data
                    # Result compare
                    #print(G.nodes[task]['output'])
                    #if fmap is not None:
                    #    self.compare_results(node_name, data)
                    #    #if 'StatefulPartitionedCall/sequential/conv2d_1/Conv2D' == node_name:
                    #    #if 'StatefulPartitionedCall/sequential/conv2d/Conv2D' == node_name:
                    #    #    disp_result(data)

    # Run inference
    def infer(self, inputs:dict):
        G = self.cnnnet.G
        # Set input data for inference
        for node_name, val in inputs.items():
            for node in G.nodes:
                if G.nodes[node]['name'] == node_name:
                    G.nodes[node]['param'] = val

        self.run_tasks()

        outputs = self.cnnnet.find_node_by_type('Result')
        res = {}
        for output in outputs:
            node_id = output[0]
            result = G.nodes[node_id]['result']
            node_name = G.nodes[node_id]['name']
            res[node_name] = result

        return res

# -------------------------------------------------------------------------------------------------------

# Class for operator plugins
class plugins:
    def __init__(self):
        self.plugins={}

    # Import an ops plugin (.py)
    def import_plugin(self, plugin_path, file_path, plugin_name=None):
        path, fname = os.path.split(file_path)
        bname, ext = os.path.splitext(fname)
        if plugin_name is None:
            plugin_name = bname
        plugin_path = plugin_path + '.' + bname
        module = importlib.import_module(plugin_path)
        setattr(self, plugin_name, module)
        self.plugins[plugin_name] = module

    # Search ops plugins (.py) and import all
    def load_plugins(self, plugin_path:str):
        plugins = glob.glob(os.path.join(plugin_path, '**', '*.py'), recursive=True)
        for plugin in plugins:
            self.import_plugin(plugin_path, plugin)


# -------------------------------------------------------------------------------------------------------

# DEBUG: Compare data for accuracy checking (with a pre-generated dict data {'node-name': np.array(featmap), ...})
def compare_results(node_name:str, result:np.array, GT:dict):
    if node_name not in GT:
        print('{} : Skipped'.format(node_name))
        return
    GT_data = GT[node_name][2]
    match = np.allclose(result, GT_data, rtol=0.001)
    if match:
        col = '\x1b[32m'
    else:
        col = '\x1b[31m'
    print('{}{} : {} / {}\x1b[37m\n'.format(col, node_name, result.shape, GT_data.shape), end='')


# DEBUG: Display np.ndarray data for debug purpose
def disp_result(data):
    N,C,H,W = data.shape
    for c in range(C):
        print('C=', c)
        for h in range(H):
            for w in range(W):
                print('{:6.3f},'.format(data[0,c,h,w]), end='')
            print()


# DEBUG: Dump network graph for debug purpose
def dump_graph(G:nx.DiGraph):
    for node_id, node_contents in G.nodes.items():
        print('node id=', node_id)
        print_dict(node_contents)
    for edge_id, edge_contents in G.edges.items():
        print('edge_id=', edge_id)
        print(' '*2, edge_contents)
