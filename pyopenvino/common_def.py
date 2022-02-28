
import numpy as np
import networkx as nx

format_config = { 'FP32': ['f', 4], 'FP16': ['e', 2], 'F32' : ['f', 4], 'F16' : ['e', 2],
                  'I64' : ['q', 8], 'I32' : ['i', 4], 'I16' : ['h', 2], 'I8'  : ['b', 1], 'U8'  : ['B', 1] }

type_convert_tbl = { 'f32':np.float32, 'f16':np.float16, 'i64':np.int64, 'i32':np.int32, 'i16':np.int16, 'i8':np.int8, 'u8':np.uint8,
                    'FP32':np.float32, 'FP16':np.float16, 'I64':np.int64 }


def string_to_tuple(string:str):
    tmp_list = [ int(item) for item in string.split(',') ]
    return tuple(tmp_list)

# -------------------------------------------------------------------------------------------------------

# DEBUG: Print dict object
def print_dict(dic:dict, indent_level=0, indent_step=4):
    for key, val in dic.items():
        print(' ' * indent_step * indent_level, key, ': ', end='')
        if type(val) is dict:
            print()
            print_dict(val, indent_level+1)
        else:
            print(val)


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
