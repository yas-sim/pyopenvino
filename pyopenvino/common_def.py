import platform
import math
import numpy as np
import networkx as nx

if platform.system() == 'Windows':
    from ctypes import windll, wintypes, byref

from functools import reduce

format_config = { 'FP32': ['f', 4], 'FP16': ['e', 2], 'F32' : ['f', 4], 'F16' : ['e', 2],
                  'I64' : ['q', 8], 'I32' : ['i', 4], 'I16' : ['h', 2], 'I8'  : ['b', 1], 'U8'  : ['B', 1] }

type_convert_tbl = { 'f32':np.float32, 'f16':np.float16, 'i64':np.int64, 'i32':np.int32, 'i16':np.int16, 'i8':np.int8, 'u8':np.uint8,
                    'FP32':np.float32, 'FP16':np.float16, 'I64':np.int64 }

def string_to_boolean(bool_val:str):
    bool_val = bool_val.upper()
    res = True if bool_val in ['TRUE', '1'] else False
    return res

def string_to_tuple(string:str) -> tuple:
    tmp_list = [ int(item) for item in string.split(',') ]
    return tuple(tmp_list)

def string_to_tuple_float(string:str) -> tuple:
    tmp_list = [ float(item) for item in string.split(',') ]
    return tuple(tmp_list)

# Enable escape sequence on Windows (for coloring text)
def enable_escape_sequence():
    if platform.system() != 'Windows':
        return
    INVALID_HANDLE_VALUE = -1
    STD_INPUT_HANDLE = -10
    STD_OUTPUT_HANDLE = -11
    STD_ERROR_HANDLE = -12
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    ENABLE_LVB_GRID_WORLDWIDE = 0x0010

    hOut = windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    if hOut == INVALID_HANDLE_VALUE:
        return False
    dwMode = wintypes.DWORD()
    if windll.kernel32.GetConsoleMode(hOut, byref(dwMode)) == 0:
        return False
    dwMode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
    # dwMode.value |= ENABLE_LVB_GRID_WORLDWIDE
    if windll.kernel32.SetConsoleMode(hOut, dwMode) == 0:
        return False
    return True

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
    rtol = 1
    if node_name not in GT:
        print('{} : Skipped'.format(node_name))
        return
    GT_data = GT[node_name][2].astype(result.dtype)
    match = np.allclose(result, GT_data, rtol=rtol)
    if match:
        col = '\x1b[32m'
    else:
        col = '\x1b[31m'
    print('{}{} : {} / {}\x1b[37m\n'.format(col, node_name, result.shape, GT_data.shape), end='')
    if not match:
        '''
        # item by item comparison
        a = result.ravel()
        b = GT_data.ravel()
        for c, d in zip(a, b):
            if abs(c-d)/c > 0.1:
                print(c, d)
        '''
        print(np.isclose(result, GT_data, rtol=rtol))
        print(np.count_nonzero(np.where(np.isclose(result, GT_data), 1, 0)))
        print('* Result')
        #disp_result(result)
        print(result)
        print('* GT')
        #disp_result(GT_data)
        print(GT_data)
        assert False


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


# ---------------------------------------------------------------------------------------


