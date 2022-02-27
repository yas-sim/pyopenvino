# lesser OpenVINO

import sys, os
import struct

import glob
import importlib

import networkx as nx
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et

def print_dict(dic:dict, indent_level=0, indent_step=4):
    for key, val in dic.items():
        print(' ' * indent_step * indent_level, key, ': ', end='')
        if type(val) is dict:
            print()
            print_dict(val, indent_level+1)
        else:
            print(val)

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

    def string_to_tuple(string:str):
        tmp_list = [ int(item) for item in string.split(',') ]
        return tuple(tmp_list)

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
                dict_layer['data']['shape'] = string_to_tuple(dict_layer['data']['shape'])
            if 'stride' in dict_layer['data']:
                dict_layer['data']['stride'] = string_to_tuple(dict_layer['data']['stride'])

        output = layer.find('output')
        if output is not None:
            port = output.find('port')
            dict_layer['output'] = { 'port':port.attrib }
            dims = port.findall('./dim')
            list_dims = []
            for dim in dims:
                list_dims.append(int(dim.text))
            dict_layer['output'] = { 'dims':tuple(list_dims) }
        dict_layers[int(layer_id)] = dict_layer

    list_edges = []
    edges = root.findall('./edges/edge')
    for edge in edges:
        attr = edge.attrib
        list_edges.append((int(attr['from-layer']), int(attr['from-port']), int(attr['to-layer']), int(attr['to-port'])))
    '''
    print_dict(dict_layers)
    print(list_edges)
    '''
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
        return -1
    '''
    nx.draw_planar(G)
    plt.show()
    '''
    '''
    print(G.nodes[0]['name'])
    print(G.nodes.data())
    '''
    return G


def set_constants_to_graph(G, bin:bytes):
    format_config = { 'FP32': ['f', 4], 'FP16': ['e', 2], 'F32' : ['f', 4], 'F16' : ['e', 2],
                      'I64' : ['q', 8], 'I32' : ['i', 4], 'I16' : ['h', 2], 'I8'  : ['b', 1], 'U8'  : ['B', 1] }
    consts = find_node_by_type(G, 'Const')
    for const in consts:
        node = G.nodes[const[0]]
        data = node['data']
        offset    = int(data['offset'])
        size      = int(data['size'])
        precision = data['element_type'].upper()
        blobBin = bin[offset:offset+size]                       # cut out the weight for this blob from the weight buffer
        formatstring = '<' + format_config[precision][0] * (len(blobBin)//format_config[precision][1])
        decoded_data = struct.unpack(formatstring, blobBin)     # decode the buffer
        node['const'] = { 'data':decoded_data, 'element_info':precision, 'size':size, 'decode_info':format_config[precision] }

def find_node_by_type(G:nx.DiGraph, type:str):
    results = []
    for node in G.nodes():
        if G.nodes[node]['type'] == type:
            results.append((node, G.nodes[node]['name']))
    #print(results)
    return results

def schedule_tasks(G):
    def search_predecessors(G, node_ids):
        for node_id in node_ids:
            predecessors = list(G.pred[node_id])
            #print(node_id, '->', predecessors)
            search_predecessors(G, predecessors)
            if node_id not in task_list:
                task_list.append(node_id)
    outputs = find_node_by_type(G, 'Result')
    task_list = []
    outputs = [ key for key,_ in outputs]
    search_predecessors(G, outputs)
    #print(task_list)
    return task_list

def run_tasks(task_list:list, G:nx.DiGraph, p:plugins):
    for task in task_list:
        node = G.nodes[task]
        node_type = node['type']
        p.plugins[node_type].name()

def run_infer(inputs:dict, task_list:list, G:nx.DiGraph, p:plugins):
    for node_name, val in inputs:
        for node in G.nodes:
            if node['name'] == node_name:
                node['param'] = val

def main():
    xml, bin = read_IR_Model('mnist.xml')
    if xml is None or bin is None:
        print('failed to read model file')
        return -1

    layers, edges = parse_IR_XML(xml)
    print(layers, edges)

    G=build_graph(layers, edges)
    #find_node_by_type(G, 'Parameter')
    #find_node_by_type(G, 'Const')
    #find_node_by_type(G, 'Result')
    set_constants_to_graph(G, bin)
    task_list = schedule_tasks(G)
    #print(task_list)

    p = plugins()
    p.import_plugins('plugins')
    #p.plugin_test.test_func('shimura')
    #p.plugins['plugin_test'].test_func('shimura')
    
    run_infer({'conv2d_input':inblob}, task_list, G, p)

if __name__ == '__main__':
    sys.exit(main())
